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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode ExecuTorch runtime outputs (or dry-run) and export YOLOZU predictions JSON. "
            "Non-dry mode requires --runtime-output-json with rows shaped "
            "[x1,y1,x2,y2,score,class_id]."
        )
    )
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--model", default=None, help="Path to ExecuTorch .pte model (required unless --dry-run).")
    parser.add_argument(
        "--runtime-output-json",
        default=None,
        help=(
            "JSON artifact produced by an ExecuTorch runtime pass. Supported forms: "
            "{image: [[x1,y1,x2,y2,score,class_id], ...]} or "
            "[{image: <path>, detections|output: [...]}]. Required for non-dry decode."
        ),
    )
    parser.add_argument(
        "--boxes-scale",
        choices=("norm", "abs"),
        default="norm",
        help="Whether runtime output boxes are normalized [0,1] or absolute image pixels (default: norm).",
    )
    parser.add_argument("--min-score", type=float, default=0.001, help="Score threshold metadata (default: 0.001).")
    parser.add_argument("--topk", type=int, default=300, help="Top-k metadata (default: 300).")
    parser.add_argument("--output", default="reports/predictions_executorch.json", help="Where to write predictions JSON.")
    parser.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    parser.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
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


def _load_runtime_outputs(path: Path) -> dict[str, list[Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read --runtime-output-json: {path}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
        payload = payload["outputs"]

    outputs: dict[str, list[Any]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(value, list):
                raise ValueError(f"runtime output for {key!r} must be a list of detection rows")
            outputs[str(key)] = value
        return outputs

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("runtime output list entries must be objects")
            image = item.get("image")
            rows = item.get("detections", item.get("output"))
            if not isinstance(image, str):
                raise ValueError("runtime output entry missing string image")
            if not isinstance(rows, list):
                raise ValueError(f"runtime output entry for {image!r} missing detections/output list")
            outputs[image] = rows
        return outputs

    raise ValueError("--runtime-output-json must be an object or list")


def _lookup_rows(outputs: dict[str, list[Any]], image: str) -> list[Any]:
    if image in outputs:
        return outputs[image]
    name = Path(image).name
    if name in outputs:
        return outputs[name]
    rel_parts = Path(image).parts[-3:]
    rel = "/".join(rel_parts)
    if rel in outputs:
        return outputs[rel]
    return []


def _decode_rows(rows: list[Any], *, image: str, boxes_scale: str, min_score: float, topk: int) -> list[dict[str, Any]]:
    width, height = get_image_size(image)
    decoded: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError(f"unsupported ExecuTorch detection row at {image}#{idx}: expected [x1,y1,x2,y2,score,class_id]")
        try:
            x1, y1, x2, y2, score, class_id = [float(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric ExecuTorch detection row at {image}#{idx}") from exc
        if score < float(min_score):
            continue
        if boxes_scale == "abs":
            cx = ((x1 + x2) / 2.0) / float(width)
            cy = ((y1 + y2) / 2.0) / float(height)
            bw = max(0.0, x2 - x1) / float(width)
            bh = max(0.0, y2 - y1) / float(height)
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
        decoded.append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "bbox": {"cx": float(cx), "cy": float(cy), "w": float(bw), "h": float(bh)},
            }
        )
    decoded.sort(key=lambda item: float(item["score"]), reverse=True)
    return decoded[: max(0, int(topk))]


def _default_wrap_meta(*, adapter: str, config: str, images: int) -> dict[str, object]:
    return {
        "timestamp": _now_utc(),
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.max_images is not None and int(args.max_images) < 0:
        raise SystemExit("--max-images must be >= 0")
    if int(args.topk) <= 0:
        raise SystemExit("--topk must be >= 1")
    if float(args.min_score) < 0.0 or float(args.min_score) > 1.0:
        raise SystemExit("--min-score must be in [0, 1]")

    dataset_root = Path(args.dataset).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    model_path = _resolve_path(args.model)
    runtime_output_path = _resolve_path(args.runtime_output_json)

    runtime_outputs: dict[str, list[Any]] = {}
    if not args.dry_run:
        if model_path is None:
            raise SystemExit("--model is required unless --dry-run is set")
        if not model_path.exists():
            raise SystemExit(f"executorch model not found: {model_path}")
        if runtime_output_path is None:
            raise SystemExit("--runtime-output-json is required for non-dry ExecuTorch decode")
        if not runtime_output_path.exists():
            raise SystemExit(f"ExecuTorch runtime output JSON not found: {runtime_output_path}")
        try:
            runtime_outputs = _load_runtime_outputs(runtime_output_path)
        except ValueError as exc:
            raise SystemExit(f"ExecuTorch decoder error: {exc}") from exc

    predictions = []
    for record in records:
        rows = [] if args.dry_run else _lookup_rows(runtime_outputs, str(record["image"]))
        try:
            detections = [] if args.dry_run else _decode_rows(
                rows,
                image=str(record["image"]),
                boxes_scale=str(args.boxes_scale),
                min_score=float(args.min_score),
                topk=int(args.topk),
            )
        except ValueError as exc:
            raise SystemExit(f"ExecuTorch decoder error: {exc}") from exc
        predictions.append({"image": record["image"], "detections": detections})

    validate_predictions_entries(predictions, strict=bool(args.strict))

    meta = _default_wrap_meta(
        adapter="executorch",
        config=(str(model_path) if model_path is not None else "executorch"),
        images=len(predictions),
    )
    meta["extra"] = {
        "exporter": "executorch",
        "protocol_id": "yolo26",
        "dataset": str(dataset_root),
        "split": manifest["split"],
        "max_images": args.max_images,
        "model": (None if model_path is None else str(model_path)),
        "model_sha256": (None if model_path is None or not model_path.exists() else _sha256(model_path)),
        "runtime_output_json": None if runtime_output_path is None else str(runtime_output_path),
        "runtime_output_sha256": None if runtime_output_path is None or not runtime_output_path.exists() else _sha256(runtime_output_path),
        "runtime_decode": {
            "contract": "combined_xyxy_score_class",
            "row_shape": "[x1,y1,x2,y2,score,class_id]",
            "boxes_scale": str(args.boxes_scale),
        },
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
    }

    payload = {"predictions": predictions, "meta": meta} if args.wrap else predictions
    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
