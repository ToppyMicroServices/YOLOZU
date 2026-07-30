import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.predictions import validate_predictions_entries
from yolozu.predictions.predictions import CURRENT_ENTRY_SCHEMA_VERSION
from yolozu.predictions.schema_governance import CURRENT_SCHEMA_VERSION

_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"))


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or name (e.g., yolo26n.pt)")
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/)")
    parser.add_argument("--split", default=None, help="Dataset split (default: auto-detect)")
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Explicit local image file or directory. Directory images are expanded in sorted order; "
            "cannot be combined with --max-images. Defaults to selected dataset manifest images."
        ),
    )
    parser.add_argument("--output", required=True, help="Where to write predictions JSON")
    parser.add_argument("--image-size", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold (default: 0.7)")
    parser.add_argument("--stride", type=int, default=32, help="Stride used by preprocessing (recorded in export_settings).")
    parser.add_argument(
        "--letterbox",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Record whether letterbox preprocessing was applied (default: true).",
    )
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300)")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap selected manifest images and actual inference inputs; cannot be combined with --source.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "Recorded backend batch preference (default: 1). Local image paths "
            "are submitted one at a time to preserve result-path identity."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Device for inference (default: cuda)")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference where supported")
    parser.add_argument(
        "--end2end",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use end2end (NMS-free) head when supported (default: true)",
    )
    parser.add_argument(
        "--protocol",
        choices=("nms_applied", "e2e_nms_free"),
        default=None,
        help="Evaluation protocol annotation. Default resolves from --end2end.",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap as {schema_version, predictions:[...], meta:{...}}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write schema-valid output without the external YOLO runtime.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation.")
    return parser.parse_args(argv)


def _result_path(result):
    for key in ("path", "orig_path"):
        value = getattr(result, key, None)
        if value:
            return str(value)
    return None


def _default_wrap_meta(*, adapter: str, config: str, images: int) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter": adapter,
        "config": config,
        "images": int(images),
        "tta": {
            "enabled": False,
            "seed": None,
            "flip_prob": 0.0,
            "norm_only": False,
            "warnings": [],
            "summary": None,
        },
        "ttt": {
            "enabled": False,
            "method": "none",
            "steps": 0,
            "batch_size": 1,
            "lr": 0.0,
            "update_filter": "none",
            "include": None,
            "exclude": None,
            "max_batches": 0,
            "seed": None,
            "mim": {"mask_prob": 0.0, "patch_size": 16, "mask_value": 0.0},
            "report": None,
        },
    }


def _sha256_if_file(value: str) -> str | None:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _manifest_image_path(*, dataset: str, image: str) -> Path:
    image_path = Path(image).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()

    dataset_path = _resolve_local_path(dataset)
    base = dataset_path if dataset_path.is_dir() else dataset_path.parent
    # build_manifest() can return a relative path that already includes the
    # dataset's relative prefix (for example, data/smoke/images/val/a.jpg).
    # Accept that resolved form only when it still identifies a manifest-derived
    # location, rather than treating an unrelated same-named CWD file as input.
    cwd_candidate = _resolve_local_path(image_path)
    dataset_arg = Path(dataset).expanduser()
    manifest_base = dataset_arg if dataset_path.is_dir() else dataset_arg.parent
    if not manifest_base.is_absolute() and manifest_base.parts:
        prefix = manifest_base.parts
        if image_path.parts[: len(prefix)] == prefix:
            return cwd_candidate

    dataset_candidate = (base / image_path).resolve()
    if dataset_candidate.is_file():
        return dataset_candidate

    try:
        cwd_candidate.relative_to(base)
        return cwd_candidate
    except ValueError:
        pass

    return dataset_candidate


def _expand_explicit_source(value: str) -> list[Path]:
    source_path = _resolve_local_path(value)
    if source_path.is_file():
        if source_path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise SystemExit(f"--source must be an image file or image directory: {source_path}")
        return [source_path]
    if not source_path.is_dir():
        raise SystemExit(f"source not found: {source_path}")

    images = sorted(
        {
            path.resolve()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        },
        key=str,
    )
    if not images:
        raise SystemExit(f"--source directory contains no supported images: {source_path}")
    return images


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.source is not None and args.max_images is not None:
        raise SystemExit("--source cannot be combined with --max-images")

    output_path = Path(args.output).expanduser()
    if output_path.exists() or output_path.is_symlink():
        if output_path.is_dir():
            raise SystemExit(f"--output must be a file path: {output_path}")
        output_path.unlink()

    manifest = build_manifest(args.dataset, split=args.split)
    manifest_records = list(manifest["images"])
    path_to_manifest: dict[str, str] = {}
    for rec in manifest_records:
        key = str(rec.get("image") or "")
        if not key:
            continue
        full = str(_manifest_image_path(dataset=str(args.dataset), image=key))
        path_to_manifest[full] = key

    if args.source is not None:
        source_mode = "explicit_source"
        selected_paths = _expand_explicit_source(str(args.source))
        selected_inputs = [path_to_manifest.get(str(path), str(path)) for path in selected_paths]
    else:
        source_mode = "dataset_manifest"
        records = manifest_records
        if args.max_images is not None:
            records = records[: max(0, int(args.max_images))]
        selected_inputs = [str(record.get("image") or "") for record in records]
        if any(not image for image in selected_inputs):
            raise SystemExit("selected dataset manifest record is missing its image identifier")
        selected_paths = [
            _manifest_image_path(dataset=str(args.dataset), image=image)
            for image in selected_inputs
        ]

    if not selected_paths:
        raise SystemExit("no input images selected")
    missing_inputs = [str(path) for path in selected_paths if not path.is_file()]
    if missing_inputs:
        raise SystemExit(f"selected input image not found: {missing_inputs[0]}")

    selected_input_count = len(selected_paths)
    runtime_sources = [str(path) for path in selected_paths]

    results = None
    runtime_error = None
    if not args.dry_run:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise SystemExit("ultralytics package is required (pip install ultralytics) unless --dry-run is set") from exc
        model = YOLO(args.model)
        def _identity_safe_results():
            for runtime_source in runtime_sources:
                yield from model.predict(
                    source=runtime_source,
                    imgsz=int(args.image_size),
                    conf=float(args.conf),
                    iou=float(args.iou),
                    max_det=int(args.max_det),
                    batch=1,
                    device=args.device,
                    half=bool(args.half),
                    end2end=bool(args.end2end),
                    stream=True,
                    verbose=False,
                )

        results = _identity_safe_results()
    else:
        runtime_error = "dry_run"

    outputs = []
    inference_calls = 0
    if args.dry_run:
        outputs = [{"image": image, "detections": []} for image in selected_inputs]
    else:
        try:
            for result in results or []:
                if inference_calls >= selected_input_count:
                    raise RuntimeError(
                        "Ultralytics returned more results than selected input images "
                        f"({inference_calls + 1} > {selected_input_count})"
                    )

                reported_path = _result_path(result)
                if not reported_path:
                    raise RuntimeError(
                        "Ultralytics result is missing path/orig_path for the selected input: "
                        f"expected {selected_paths[inference_calls]}"
                    )
                resolved_result = _resolve_local_path(reported_path)
                if resolved_result != selected_paths[inference_calls]:
                    raise RuntimeError(
                        "Ultralytics result order/path does not match the selected input list: "
                        f"expected {selected_paths[inference_calls]}, got {resolved_result}"
                    )

                dets = []
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xywhn = boxes.xywhn
                    conf = boxes.conf
                    cls = boxes.cls
                    if xywhn is not None and conf is not None and cls is not None:
                        xywhn_list = xywhn.detach().cpu().tolist()
                        conf_list = conf.detach().cpu().tolist()
                        cls_list = cls.detach().cpu().tolist()
                        for bbox, score, class_id in zip(xywhn_list, conf_list, cls_list):
                            if len(bbox) != 4:
                                continue
                            dets.append(
                                {
                                    "class_id": int(class_id),
                                    "score": float(score),
                                    "bbox": {
                                        "cx": float(bbox[0]),
                                        "cy": float(bbox[1]),
                                        "w": float(bbox[2]),
                                        "h": float(bbox[3]),
                                    },
                                }
                            )

                outputs.append({"image": selected_inputs[inference_calls], "detections": dets})
                inference_calls += 1
        except Exception as exc:
            raise SystemExit(f"Ultralytics inference failed: {exc}") from exc

        if inference_calls != selected_input_count or len(outputs) != selected_input_count:
            raise SystemExit(
                "Ultralytics result count does not match selected input count: "
                f"results={inference_calls}, selected={selected_input_count}"
            )

    result_count = int(inference_calls)
    runtime_executed = bool(
        not args.dry_run
        and inference_calls == selected_input_count
        and len(outputs) == selected_input_count
    )

    for output in outputs:
        output["schema_version"] = CURRENT_ENTRY_SCHEMA_VERSION
    validate_predictions_entries(outputs, strict=bool(args.strict))

    protocol_id = str(args.protocol) if args.protocol else ("e2e_nms_free" if bool(args.end2end) else "nms_applied")

    if args.wrap:
        meta = _default_wrap_meta(adapter="ultralytics", config=str(args.model), images=len(outputs))
        meta["extra"] = {
            "exporter": "ultralytics",
            "protocol_id": protocol_id,
            "imgsz": int(args.image_size),
            "dataset": str(args.dataset),
            "split": manifest["split"],
            "source": str(args.source) if args.source is not None else None,
            "source_mode": source_mode,
            "max_images": args.max_images,
            "model": str(args.model),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "batch": int(args.batch),
            "runtime_batch_mode": "identity_safe_single_source",
            "device": str(args.device),
            "half": bool(args.half),
            "end2end": bool(args.end2end),
            "export_settings": {
                "imgsz": int(args.image_size),
                "score_threshold": float(args.conf),
                "iou_threshold": float(args.iou),
                "max_detections": int(args.max_det),
                "bbox_format": "cxcywh_norm",
                "protocol": protocol_id,
                "nms": {"applied": protocol_id == "nms_applied"},
                "preprocess": {
                    "method": "letterbox" if bool(args.letterbox) else "resize",
                    "input_color": "RGB",
                    "normalize": "0_1",
                    "resize_interp": "linear",
                    "letterbox_fill": [114, 114, 114],
                    "stride": int(args.stride),
                },
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "dry_run": bool(args.dry_run),
            "runtime_error": runtime_error,
            "runtime_executed": runtime_executed,
            "execution_status": "dry_run" if args.dry_run else "completed",
            "inference_calls": int(inference_calls),
            "selected_input_count": int(selected_input_count),
            "selected_inputs": list(selected_inputs),
            "result_count": result_count,
            "model_provenance": {
                "model": str(args.model),
                "model_sha256": _sha256_if_file(str(args.model)),
            },
        }
        try:
            import torch  # type: ignore

            meta["extra"]["torch"] = {"version": getattr(torch, "__version__", None), "cuda": bool(torch.cuda.is_available())}
        except Exception:
            meta["extra"]["torch"] = None
        try:
            import ultralytics  # type: ignore

            meta["extra"]["ultralytics"] = {"version": getattr(ultralytics, "__version__", None)}
        except Exception:
            meta["extra"]["ultralytics"] = None

        payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "predictions": outputs,
            "meta": meta,
        }
    else:
        payload = outputs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
