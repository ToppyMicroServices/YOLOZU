#!/usr/bin/env python3
from __future__ import annotations

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
from yolozu.imports import project_yolox_exp
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run YOLOX inference (or dry-run) and export YOLOZU predictions.json.")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root")
    p.add_argument("--split", default=None, help="Dataset split (default: auto)")
    p.add_argument("--output", required=True, help="Output predictions JSON path")
    p.add_argument("--max-images", type=int, default=None, help="Cap number of images")
    p.add_argument("--exp", default=None, help="YOLOX exp file path")
    p.add_argument("--weights", default=None, help="YOLOX checkpoint path")
    p.add_argument("--device", default="cuda", help="Torch device for YOLOX inference")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--score-thr", type=float, default=0.01, help="Score threshold")
    p.add_argument("--nms-thr", type=float, default=0.65, help="NMS IoU threshold")
    p.add_argument("--topk", type=int, default=300, help="Top-k detections per image")
    p.add_argument(
        "--protocol",
        choices=("nms_applied", "e2e_nms_free"),
        default="nms_applied",
        help="Evaluation protocol annotation",
    )
    p.add_argument("--strict", action="store_true", help="Strict schema validation")
    p.add_argument("--dry-run", action="store_true", help="Write schema-valid empty detections")
    return p.parse_args(argv)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _xyxy_to_cxcywh_norm(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> dict[str, float] | None:
    if width <= 0 or height <= 0:
        return None
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    if w <= 0.0 or h <= 0.0:
        return None
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return {"cx": cx / float(width), "cy": cy / float(height), "w": w / float(width), "h": h / float(height)}


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()
    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]

    exp_path = Path(args.exp).expanduser() if args.exp else None
    if exp_path is not None and not exp_path.is_absolute():
        exp_path = (Path.cwd() / exp_path).resolve()
    weights_path = Path(args.weights).expanduser() if args.weights else None
    if weights_path is not None and not weights_path.is_absolute():
        weights_path = (Path.cwd() / weights_path).resolve()

    exp_params: dict[str, Any] | None = None
    exp_error: str | None = None
    if exp_path is not None:
        try:
            exp_cfg = project_yolox_exp(config=exp_path)
            exp_params = exp_cfg.to_dict()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:  # pragma: no cover
            exp_error = str(exc)

    outputs: list[dict[str, Any]] = []
    runtime_error: str | None = None

    run_native = (not bool(args.dry_run)) and exp_path is not None and weights_path is not None

    if run_native:
        try:
            import cv2  # type: ignore
            import torch  # type: ignore
            from yolox.data.data_augment import preproc  # type: ignore
            from yolox.exp import get_exp  # type: ignore
            from yolox.utils import postprocess  # type: ignore

            exp = get_exp(str(exp_path), None)
            model = exp.get_model()
            ckpt = torch.load(str(weights_path), map_location="cpu")
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
            model.to(args.device)
            model.eval()

            num_classes = int(getattr(exp, "num_classes", 80))

            for rec in records:
                image_rel = str(rec.get("image") or "")
                image_abs = dataset_root / image_rel
                width = int(rec.get("width") or 0)
                height = int(rec.get("height") or 0)

                img = cv2.imread(str(image_abs))
                if img is None:
                    outputs.append({"image": image_rel, "detections": []})
                    continue

                padded, ratio = preproc(img, (int(args.imgsz), int(args.imgsz)))
                tensor = torch.from_numpy(padded).unsqueeze(0).float().to(args.device)
                with torch.no_grad():
                    pred = model(tensor)
                    dets = postprocess(
                        pred,
                        num_classes=num_classes,
                        conf_thre=float(args.score_thr),
                        nms_thre=float(args.nms_thr),
                        class_agnostic=False,
                    )

                detections: list[dict[str, Any]] = []
                one = dets[0] if isinstance(dets, (list, tuple)) and len(dets) > 0 else None
                if one is not None:
                    arr = one.detach().cpu().numpy()
                    arr = arr[: int(args.topk)]
                    for row in arr:
                        x1, y1, x2, y2, obj, cls_conf, cls_id = [float(v) for v in row[:7]]
                        score = float(obj * cls_conf)
                        if score < float(args.score_thr):
                            continue
                        if float(ratio) > 0:
                            x1, y1, x2, y2 = x1 / float(ratio), y1 / float(ratio), x2 / float(ratio), y2 / float(ratio)
                        bbox = _xyxy_to_cxcywh_norm(x1, y1, x2, y2, width, height)
                        if bbox is None:
                            continue
                        detections.append({"class_id": int(cls_id), "score": score, "bbox": bbox})
                outputs.append({"image": image_rel, "detections": detections})
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:  # pragma: no cover
            runtime_error = str(exc)
            outputs = [{"image": str(rec.get("image") or ""), "detections": []} for rec in records]
    else:
        outputs = [{"image": str(rec.get("image") or ""), "detections": []} for rec in records]

    validate_predictions_entries(outputs, strict=bool(args.strict))

    meta = _default_wrap_meta(adapter="yolox", config=str(exp_path or "yolox"), images=len(outputs))
    meta["extra"] = {
        "exporter": "yolox",
        "protocol_id": str(args.protocol),
        "dataset": str(dataset_root),
        "split": manifest["split"],
        "exp": str(exp_path) if exp_path else None,
        "weights": str(weights_path) if weights_path else None,
        "weights_sha256": _sha256(weights_path),
        "exp_error": exp_error,
        "runtime_error": runtime_error,
        "dry_run": bool(args.dry_run),
        "export_settings": {
            "imgsz": int(args.imgsz),
            "score_threshold": float(args.score_thr),
            "iou_threshold": float(args.nms_thr),
            "max_detections": int(args.topk),
            "bbox_format": "cxcywh_norm",
            "protocol": str(args.protocol),
            "nms": {"applied": args.protocol == "nms_applied"},
            "preprocessing": {
                "method": "letterbox",
                "input_color": "BGR",
                "normalize": "0_1",
                "resize_interp": "linear",
                "letterbox_fill": [114, 114, 114],
            },
            "decode": {
                "type": "yolox_anchor_free",
                "strides": [8, 16, 32],
                "decode_space": "grid",
            },
            "exp_params": exp_params,
        },
        "runtime": {"platform": platform.system(), "python": sys.version},
    }

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"predictions": outputs, "meta": meta}, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
