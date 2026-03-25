import argparse
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
    parser = argparse.ArgumentParser(description="Run Detectron2 inference and export YOLOZU predictions.json.")
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root")
    parser.add_argument("--split", default=None, help="Dataset split (default: auto)")
    parser.add_argument("--config", required=True, help="Detectron2 config YAML path")
    parser.add_argument("--weights", required=True, help="Detectron2 checkpoint path")
    parser.add_argument("--output", required=True, help="Output predictions JSON path")
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of images")
    parser.add_argument("--score-thr", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--topk", type=int, default=300, help="Top-k detections per image")
    parser.add_argument("--device", default="cuda", help="Detectron2 model device")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size recorded in export_settings")
    parser.add_argument(
        "--protocol",
        choices=("nms_applied", "e2e_nms_free"),
        default="nms_applied",
        help="Protocol annotation in export settings",
    )
    parser.add_argument("--input-color", choices=("BGR", "RGB"), default="BGR", help="Input color order before framework pipeline")
    parser.add_argument("--normalize-scale", default="1.0", help="Normalization scale metadata (e.g. 1/255)")
    parser.add_argument("--normalize-mean", default="", help="Comma-separated mean values metadata")
    parser.add_argument("--normalize-std", default="", help="Comma-separated std values metadata")
    parser.add_argument("--resize-policy", default="config_shortest_edge", help="Resize policy metadata")
    parser.add_argument("--pad-policy", default="config_pad_to_stride", help="Pad policy metadata")
    parser.add_argument("--dry-run", action="store_true", help="Write schema-valid output without Detectron2 runtime")
    parser.add_argument("--strict", action="store_true", help="Strict schema validation")
    return parser.parse_args(argv)


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


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return value.detach().cpu().numpy().tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    try:
        return value.cpu().numpy().tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    try:
        return value.tolist()
    except (AttributeError, TypeError, ValueError):
        pass
    return []


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    manifest = build_manifest(dataset_root, split=args.split)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]

    outputs: list[dict[str, Any]] = []
    runtime_error: str | None = None

    predictor = None
    cv2 = None
    if not args.dry_run:
        try:
            import cv2 as _cv2  # type: ignore
            from detectron2.config import get_cfg  # type: ignore
            from detectron2.engine import DefaultPredictor  # type: ignore

            cfg = get_cfg()
            cfg.merge_from_file(str(Path(args.config).expanduser()))
            cfg.MODEL.WEIGHTS = str(Path(args.weights).expanduser())
            cfg.MODEL.DEVICE = str(args.device)
            try:
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.score_thr)
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = float(args.score_thr)
            except (AttributeError, TypeError, ValueError):
                pass
            predictor = DefaultPredictor(cfg)
            cv2 = _cv2
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            runtime_error = str(exc)

    for record in records:
        image_rel = str(record.get("image") or "")
        image_abs = dataset_root / image_rel
        detections: list[dict[str, Any]] = []
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)

        if predictor is not None and cv2 is not None:
            image = cv2.imread(str(image_abs))
            if image is None:
                outputs.append({"image": image_rel, "detections": []})
                continue
            result = predictor(image)
            instances = result.get("instances") if isinstance(result, dict) else None
            if instances is not None:
                try:
                    instances = instances.to("cpu")
                except (AttributeError, RuntimeError):
                    pass
                boxes_tensor = None
                if hasattr(instances, "pred_boxes") and getattr(instances, "pred_boxes") is not None:
                    boxes_tensor = getattr(instances.pred_boxes, "tensor", None)
                boxes = _tensor_to_list(boxes_tensor)
                scores = _tensor_to_list(getattr(instances, "scores", None))
                classes = _tensor_to_list(getattr(instances, "pred_classes", None))

                for box, score, class_id in zip(boxes, scores, classes):
                    if len(box) != 4:
                        continue
                    score_v = float(score)
                    if score_v < float(args.score_thr):
                        continue
                    bbox = _xyxy_to_cxcywh_norm(float(box[0]), float(box[1]), float(box[2]), float(box[3]), width, height)
                    if bbox is None:
                        continue
                    detections.append({"class_id": int(class_id), "score": score_v, "bbox": bbox})

                detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)[: int(args.topk)]

        outputs.append({"image": image_rel, "detections": detections})

    validate_predictions_entries(outputs, strict=bool(args.strict))

    meta = _default_wrap_meta(adapter="detectron2", config=str(args.config), images=len(outputs))
    meta["extra"] = {
        "exporter": "detectron2",
        "protocol_id": str(args.protocol),
        "dataset": str(dataset_root),
        "split": manifest["split"],
        "weights": str(args.weights),
        "device": str(args.device),
        "runtime_error": runtime_error,
        "dry_run": bool(args.dry_run),
        "export_settings": {
            "imgsz": int(args.imgsz),
            "score_threshold": float(args.score_thr),
            "iou_threshold": None,
            "max_detections": int(args.topk),
            "bbox_format": "cxcywh_norm",
            "protocol": str(args.protocol),
            "nms": {"applied": args.protocol == "nms_applied"},
            "preprocessing": {
                "source": "detectron2_config_pipeline",
                "input_color": str(args.input_color),
                "normalize_scale": str(args.normalize_scale),
                "normalize_mean": [x.strip() for x in str(args.normalize_mean).split(",") if x.strip()],
                "normalize_std": [x.strip() for x in str(args.normalize_std).split(",") if x.strip()],
                "resize_policy": str(args.resize_policy),
                "pad_policy": str(args.pad_policy),
            },
            "raw_output_bbox_format": "xyxy_abs",
        },
        "runtime": {"platform": platform.system()},
    }

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"predictions": outputs, "meta": meta}, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    if runtime_error and not args.dry_run:
        print(f"[warn] detectron2 runtime unavailable: {runtime_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
