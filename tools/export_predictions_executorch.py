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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ExecuTorch inference (or dry-run) and export YOLOZU predictions JSON."
    )
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    parser.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--model", default=None, help="Path to ExecuTorch .pte model (required unless --dry-run).")
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

    model_path: Path | None = None
    if args.model:
        model_path = Path(args.model).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()

    predictions = [{"image": record["image"], "detections": []} for record in records]

    runtime_warning: str | None = None
    if not args.dry_run:
        if model_path is None:
            raise SystemExit("--model is required unless --dry-run is set")
        if not model_path.exists():
            raise SystemExit(f"executorch model not found: {model_path}")
        try:
            import executorch  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "executorch runtime is required for non-dry-run mode (pip install executorch). "
                "Use --dry-run for contract validation without runtime dependencies."
            ) from exc
        runtime_warning = (
            "generic ExecuTorch decode path is not configured in this skeleton exporter; "
            "emitted empty detections with valid interface contract"
        )

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
        "min_score": float(args.min_score),
        "topk": int(args.topk),
        "dry_run": bool(args.dry_run),
        "runtime_warning": runtime_warning,
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
