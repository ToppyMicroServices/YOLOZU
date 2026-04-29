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
    p = argparse.ArgumentParser(
        description=(
            "Run a TorchScript detection artifact and export YOLOZU predictions JSON. "
            "The declared decode path expects a combined tensor shaped (N,6) or "
            "(1,N,6): [x1,y1,x2,y2,score,class_id]."
        )
    )
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--model", required=False, help="Path to TorchScript model (required unless --dry-run).")
    p.add_argument("--device", default="cpu", help="Torch device string (default: cpu).")
    p.add_argument("--imgsz", "--input-size", dest="imgsz", type=int, default=640, help="Square input size (default: 640).")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional dict output key for the combined detection tensor; omitted means single tensor output.",
    )
    p.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for decoded rows (default: xyxy_score_class).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in input pixels (abs) or normalized [0,1] wrt input size (default: norm).",
    )
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold (default: 0.001).")
    p.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (default: 300).")
    p.add_argument("--output", default="reports/predictions_torchscript.json", help="Where to write predictions JSON.")
    p.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    p.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")
    return p.parse_args(argv)


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


def _split_combined_output(values: Any, *, fmt: str) -> tuple[Any, Any, Any]:
    if fmt != "xyxy_score_class":
        raise ValueError(f"unsupported combined format: {fmt}")
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for TorchScript decode")
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"unsupported TorchScript combined output shape: {arr.shape}; expected (N,6) or (1,N,6)")
    return arr[:, :4], arr[:, 4], arr[:, 5]


def _tensor_to_numpy(value: Any) -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for TorchScript export") from exc

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0].detach().cpu().numpy()
        raise ValueError("TorchScript output tuple/list must contain exactly one combined detection tensor")
    return value


def _select_output(raw: Any, *, combined_output: str | None) -> Any:
    if isinstance(raw, dict):
        if combined_output:
            if combined_output not in raw:
                raise ValueError(f"missing TorchScript combined output key: {combined_output}")
            return _tensor_to_numpy(raw[combined_output])
        tensor_items = list(raw.values())
        if len(tensor_items) == 1:
            return _tensor_to_numpy(tensor_items[0])
        keys = ", ".join(str(k) for k in raw)
        raise ValueError(f"TorchScript dict output has multiple keys ({keys}); pass --combined-output")
    return _tensor_to_numpy(raw)


def _load_image_tensor(image_path: str, *, input_size: int, device: str):
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for TorchScript preprocessing")
    try:
        from PIL import Image
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow and torch are required for TorchScript preprocessing") from exc

    orig_w, orig_h = get_image_size(image_path)
    letterbox = compute_letterbox(orig_w=orig_w, orig_h=orig_h, input_size=input_size)

    img = Image.open(image_path).convert("RGB")
    if img.size != (letterbox.new_w, letterbox.new_h):
        img = img.resize((letterbox.new_w, letterbox.new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(img, (int(letterbox.pad_x), int(letterbox.pad_y)))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return torch.from_numpy(arr).to(device), (orig_w, orig_h), letterbox


def _default_tta_meta() -> dict[str, Any]:
    return {
        "enabled": False,
        "seed": None,
        "flip_prob": 0.0,
        "norm_only": False,
        "warnings": [],
        "summary": None,
    }


def _default_ttt_meta() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "none",
        "steps": 0,
        "batch_size": 0,
        "lr": 0.0,
        "update_filter": "none",
        "include": None,
        "exclude": None,
        "max_batches": 0,
        "seed": None,
        "mim": {"mask_prob": 0.0, "patch_size": 0, "mask_value": 0.0},
        "report": None,
    }


def _meta(*, args: argparse.Namespace, model_path: Path | None, images: int, split: str) -> dict[str, Any]:
    return {
        "timestamp": _now_utc(),
        "adapter": "torchscript",
        "config": str(model_path) if model_path is not None else "torchscript",
        "images": int(images),
        "tta": _default_tta_meta(),
        "ttt": _default_ttt_meta(),
        "extra": {
            "exporter": "torchscript",
            "protocol_id": "yolo26",
            "dataset": str(_resolve_path(args.dataset) or args.dataset),
            "split": split,
            "max_images": args.max_images,
            "model": None if model_path is None else str(model_path),
            "model_sha256": None if model_path is None or not model_path.exists() else _sha256(model_path),
            "device": str(args.device),
            "imgsz": int(args.imgsz),
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

    if args.max_images is not None and int(args.max_images) < 0:
        raise SystemExit("--max-images must be >= 0")
    if int(args.topk) <= 0:
        raise SystemExit("--topk must be >= 1")
    if float(args.min_score) < 0.0 or float(args.min_score) > 1.0:
        raise SystemExit("--min-score must be in [0, 1]")
    if int(args.imgsz) <= 0:
        raise SystemExit("--imgsz must be >= 1")

    dataset_root = _resolve_path(args.dataset)
    if dataset_root is None:
        raise SystemExit("--dataset is required")
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    predictions: list[dict[str, Any]] = []
    model_path = _resolve_path(args.model)
    if args.dry_run:
        predictions = [{"image": record["image"], "detections": []} for record in records]
    else:
        if model_path is None:
            raise SystemExit("--model is required unless --dry-run is set")
        if not model_path.exists():
            raise SystemExit(f"TorchScript model not found: {model_path}")
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for TorchScript exporter") from exc

        module = torch.jit.load(str(model_path), map_location=str(args.device))
        module.eval()
        with torch.no_grad():
            for record in records:
                image_path = record["image"]
                x, (orig_w, orig_h), letterbox = _load_image_tensor(
                    image_path,
                    input_size=int(args.imgsz),
                    device=str(args.device),
                )
                raw = module(x)
                selected = _select_output(raw, combined_output=args.combined_output)
                boxes_t, scores_t, class_t = _split_combined_output(selected, fmt=str(args.combined_format))
                boxes = np.asarray(boxes_t)
                scores = np.asarray(scores_t).astype(float)
                class_ids = np.asarray(class_t).astype(int)
                idx = [i for i, s in enumerate(scores.tolist()) if float(s) >= float(args.min_score)]
                idx.sort(key=lambda i: float(scores[i]), reverse=True)
                idx = idx[: max(0, int(args.topk))]

                detections = []
                for i in idx:
                    x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
                    if args.boxes_scale == "norm":
                        x1 *= float(args.imgsz)
                        y1 *= float(args.imgsz)
                        x2 *= float(args.imgsz)
                        y2 *= float(args.imgsz)
                    orig_xyxy = input_xyxy_to_orig_xyxy(
                        (x1, y1, x2, y2),
                        letterbox=letterbox,
                        orig_w=orig_w,
                        orig_h=orig_h,
                    )
                    bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)
                    detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})
                predictions.append({"image": image_path, "detections": detections})

    validate_predictions_entries(predictions, strict=bool(args.strict))
    payload = (
        {"predictions": predictions, "meta": _meta(args=args, model_path=model_path, images=len(predictions), split=manifest["split"])}
        if args.wrap
        else predictions
    )
    out_path = _resolve_path(args.output)
    if out_path is None:
        raise SystemExit("--output is required")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
