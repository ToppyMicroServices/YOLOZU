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


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or name (e.g., yolo26n.pt)")
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/)")
    parser.add_argument("--split", default=None, help="Dataset split (default: auto-detect)")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source override (directory or file). Defaults to dataset images/<split>.",
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
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference (default: 1)")
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
    parser.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
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


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    manifest = build_manifest(args.dataset, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]
    image_paths = [record["image"] for record in records]
    path_to_manifest: dict[str, str] = {}
    base_to_manifest: dict[str, str] = {}
    for rec in records:
        key = str(rec.get("image") or "")
        if not key:
            continue
        full = str((Path(args.dataset) / key).resolve())
        path_to_manifest[full] = key
        base_to_manifest.setdefault(Path(key).name, key)
    images_dir = Path(args.dataset) / "images" / manifest["split"]
    source = args.source or str(images_dir)
    if not Path(source).exists():
        raise SystemExit(f"source not found: {source}")

    results = None
    runtime_error = None
    if not args.dry_run:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise SystemExit("ultralytics package is required (pip install ultralytics) unless --dry-run is set") from exc
        model = YOLO(args.model)
        results = model.predict(
            source=source,
            imgsz=int(args.image_size),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            batch=int(args.batch),
            device=args.device,
            half=bool(args.half),
            end2end=bool(args.end2end),
            stream=True,
            verbose=False,
        )
    else:
        runtime_error = "dry_run"

    outputs = []
    inference_calls = 0
    if args.dry_run:
        outputs = [{"image": str(rec.get("image") or ""), "detections": []} for rec in records]
    else:
        for result in results or []:
            inference_calls += 1
            image_path = _result_path(result)
            if image_path is None:
                if image_paths:
                    image_path = image_paths[len(outputs)]
                else:
                    image_path = ""
            else:
                resolved = str(Path(image_path).resolve())
                image_path = path_to_manifest.get(resolved) or base_to_manifest.get(Path(image_path).name) or image_path

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

            outputs.append({"image": image_path, "detections": dets})

    runtime_executed = bool(not args.dry_run and inference_calls > 0)
    if not args.dry_run and not runtime_executed:
        raise SystemExit("Ultralytics inference did not execute for any input image")

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
            "max_images": args.max_images,
            "model": str(args.model),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "batch": int(args.batch),
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

        payload = {"predictions": outputs, "meta": meta}
    else:
        payload = outputs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
