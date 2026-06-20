#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.letterbox import compute_letterbox, input_xyxy_to_orig_xyxy, orig_xyxy_to_cxcywh_norm
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenVINO detection artifact and export YOLOZU predictions JSON. "
            "The declared default decode path expects one combined output shaped "
            "(N,6) or (1,N,6): [x1,y1,x2,y2,score,class_id]."
        )
    )
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--model", required=False, help="Path to OpenVINO IR .xml model (required unless --dry-run).")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name (default: CPU).")
    parser.add_argument("--imgsz", "--input-size", dest="imgsz", type=int, default=640, help="Square input size (default: 640).")
    parser.add_argument("--input-name", default=None, help="Optional OpenVINO input tensor name.")
    parser.add_argument("--combined-output", default=None, help="Optional output tensor name for combined detections.")
    parser.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for decoded rows (default: xyxy_score_class).",
    )
    parser.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in input pixels (abs) or normalized [0,1] wrt input size (default: norm).",
    )
    parser.add_argument("--min-score", type=float, default=0.001, help="Score threshold (default: 0.001).")
    parser.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (default: 300).")
    parser.add_argument("--output", default="reports/predictions_openvino.json", help="Where to write predictions JSON.")
    parser.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    parser.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without loading OpenVINO.")
    parser.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")
    return parser.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _openvino_core() -> Any:
    try:
        from openvino import Core  # type: ignore
    except Exception:
        try:
            from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openvino is required for non-dry OpenVINO prediction export") from exc
    return Core()


def _split_combined_output(values: Any, *, fmt: str) -> tuple[Any, Any, Any]:
    if fmt != "xyxy_score_class":
        raise ValueError(f"unsupported combined format: {fmt}")
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for OpenVINO decode")
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"unsupported OpenVINO combined output shape: {arr.shape}; expected (N,6) or (1,N,6)")
    return arr[:, :4], arr[:, 4], arr[:, 5]


def _tensor_names(tensor: Any) -> set[str]:
    names = set()
    try:
        names.update(str(name) for name in tensor.get_names())
    except Exception:
        # Some OpenVINO tensor-like objects do not expose the names collection.
        pass
    try:
        names.add(str(tensor.get_any_name()))
    except Exception:
        # Anonymous tensors are matched by other available names or by position.
        pass
    return names


def _select_input(compiled: Any, input_name: str | None) -> Any:
    inputs = list(compiled.inputs)
    if not inputs:
        raise ValueError("OpenVINO model has no inputs")
    if input_name:
        for tensor in inputs:
            if input_name in _tensor_names(tensor):
                return tensor
        raise ValueError(f"missing OpenVINO input tensor: {input_name}")
    if len(inputs) != 1:
        names = ", ".join(sorted(name for tensor in inputs for name in _tensor_names(tensor)))
        raise ValueError(f"OpenVINO model has multiple inputs ({names}); pass --input-name")
    return inputs[0]


def _output_dict(raw: Any, compiled: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            names = _tensor_names(key)
            if not names:
                names = {str(key)}
            for name in names:
                out[name] = value
        return out
    outputs = list(compiled.outputs)
    if len(outputs) == 1:
        names = _tensor_names(outputs[0]) or {"output0"}
        for name in names:
            out[name] = raw
        return out
    raise ValueError("OpenVINO inference returned unnamed multiple outputs")


def _select_output(raw: Any, compiled: Any, *, combined_output: str | None) -> Any:
    outputs = _output_dict(raw, compiled)
    if combined_output:
        if combined_output not in outputs:
            raise ValueError(f"missing OpenVINO combined output key: {combined_output}")
        return outputs[combined_output]
    unique_values = list({id(value): value for value in outputs.values()}.values())
    if len(unique_values) == 1:
        return unique_values[0]
    keys = ", ".join(sorted(outputs))
    raise ValueError(f"OpenVINO model has multiple outputs ({keys}); pass --combined-output")


def _load_image(image_path: str, *, input_size: int) -> tuple[Any, tuple[int, int], Any]:
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for OpenVINO preprocessing")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for OpenVINO preprocessing") from exc

    orig_w, orig_h = get_image_size(image_path)
    letterbox = compute_letterbox(orig_w=orig_w, orig_h=orig_h, input_size=input_size)
    img = Image.open(image_path).convert("RGB")
    if img.size != (letterbox.new_w, letterbox.new_h):
        img = img.resize((letterbox.new_w, letterbox.new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(img, (int(letterbox.pad_x), int(letterbox.pad_y)))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr, (orig_w, orig_h), letterbox


def _meta(*, args: argparse.Namespace, model_path: Path | None, images: int, split: str) -> dict[str, Any]:
    return {
        "timestamp": _now_utc(),
        "adapter": "openvino",
        "config": str(model_path) if model_path is not None else "openvino",
        "images": int(images),
        "extra": {
            "exporter": "openvino",
            "protocol_id": "yolo26",
            "dataset": str(_resolve_path(args.dataset) or args.dataset),
            "split": split,
            "max_images": args.max_images,
            "model": None if model_path is None else str(model_path),
            "model_sha256": None if model_path is None or not model_path.exists() else _sha256(model_path),
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "input_name": args.input_name,
            "combined_output": args.combined_output,
            "combined_format": args.combined_format,
            "boxes_scale": args.boxes_scale,
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
            "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    dataset_root = _resolve_path(args.dataset)
    if dataset_root is None:
        raise SystemExit("--dataset is required")
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest.get("images", []))
    if args.max_images is not None:
        records = records[: int(args.max_images)]
    model_path = _resolve_path(args.model)
    predictions: list[dict[str, Any]] = []

    if args.dry_run:
        predictions = [{"image": str(record["image"]), "detections": []} for record in records]
    else:
        if model_path is None:
            raise SystemExit("--model is required unless --dry-run is set")
        if not model_path.exists():
            raise SystemExit(f"OpenVINO model not found: {model_path}")
        core = _openvino_core()
        model = core.read_model(str(model_path))
        compiled = core.compile_model(model, str(args.device))
        input_tensor = _select_input(compiled, args.input_name)
        for record in records:
            image_path = str(record["image"])
            image, (orig_w, orig_h), letterbox = _load_image(image_path, input_size=int(args.imgsz))
            raw = compiled({input_tensor: image})
            selected = _select_output(raw, compiled, combined_output=args.combined_output)
            boxes, scores, class_ids = _split_combined_output(selected, fmt=str(args.combined_format))
            order = np.argsort(-np.asarray(scores))[: max(int(args.topk), 0)]
            detections: list[dict[str, Any]] = []
            for idx in order:
                score = float(scores[idx])
                if score < float(args.min_score):
                    continue
                x1, y1, x2, y2 = [float(value) for value in np.asarray(boxes[idx]).tolist()]
                if args.boxes_scale == "norm":
                    x1, y1, x2, y2 = x1 * args.imgsz, y1 * args.imgsz, x2 * args.imgsz, y2 * args.imgsz
                orig_xyxy = input_xyxy_to_orig_xyxy((x1, y1, x2, y2), letterbox=letterbox, orig_w=orig_w, orig_h=orig_h)
                bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)
                detections.append({"class_id": int(class_ids[idx]), "score": score, "bbox": bbox})
            predictions.append({"image": image_path, "detections": detections})

    validate_predictions_entries(predictions, strict=args.strict)
    payload = {
        "predictions": predictions,
        "meta": _meta(args=args, model_path=model_path, images=len(records), split=str(manifest.get("split") or "")),
    } if args.wrap else predictions
    out_path = _resolve_path(args.output) or (repo_root / "reports" / "predictions_openvino.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
