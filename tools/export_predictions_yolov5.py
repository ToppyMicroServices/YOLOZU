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


def _xyxy_to_cxcywh_norm(xyxy: list[float], width: int, height: int) -> dict[str, float] | None:
    if len(xyxy) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return None
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return {"cx": cx / float(width), "cy": cy / float(height), "w": w / float(width), "h": h / float(height)}


def _load_json_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return [x for x in data["predictions"] if isinstance(x, dict)]
    raise SystemExit("unsupported --json format (expected list or wrapped predictions)")


def _parse_labels_txt(path: Path, conf_default: float) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if not path.exists():
        return detections
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.strip().split()
        if len(cols) < 5:
            continue
        try:
            class_id = int(float(cols[0]))
            cx = float(cols[1])
            cy = float(cols[2])
            w = float(cols[3])
            h = float(cols[4])
            score = float(cols[5]) if len(cols) > 5 else float(conf_default)
        except (TypeError, ValueError, IndexError):
            continue
        detections.append(
            {
                "class_id": class_id,
                "score": score,
                "bbox": {"cx": cx, "cy": cy, "w": w, "h": h},
            }
        )
    return detections


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert YOLOv5-style outputs into YOLOZU predictions.json.")
    parser.add_argument("--dataset", required=True, help="YOLO-format dataset root")
    parser.add_argument("--split", default=None, help="Dataset split (default: auto)")
    parser.add_argument("--labels-dir", default=None, help="YOLOv5 detect --save-txt labels directory")
    parser.add_argument("--json", default=None, help="Optional JSON detections with xyxy/conf/cls fields")
    parser.add_argument("--output", required=True, help="Output predictions.json")
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of images")
    parser.add_argument("--conf-default", type=float, default=1.0, help="Fallback confidence for txt rows without conf")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size recorded in export_settings")
    parser.add_argument("--letterbox", default=True, action=argparse.BooleanOptionalAction, help="Record letterbox preprocessing")
    parser.add_argument("--stride", type=int, default=32, help="Stride recorded in export_settings")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold recorded in export_settings")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold recorded in export_settings")
    parser.add_argument("--protocol", choices=("nms_applied", "e2e_nms_free"), default="nms_applied")
    parser.add_argument("--strict", action="store_true", help="Strict schema validation")
    args = parser.parse_args(argv)

    if not args.labels_dir and not args.json:
        raise SystemExit("one of --labels-dir or --json is required")

    manifest = build_manifest(args.dataset, split=args.split)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]

    labels_dir = Path(args.labels_dir).expanduser() if args.labels_dir else None
    if labels_dir is not None and not labels_dir.is_absolute():
        labels_dir = (Path.cwd() / labels_dir).resolve()

    json_entries: dict[str, list[dict[str, Any]]] = {}
    if args.json:
        src = Path(args.json).expanduser()
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        raw = _load_json_predictions(src)
        for entry in raw:
            image = str(entry.get("image") or "").strip()
            if not image:
                continue
            dets = entry.get("detections")
            if isinstance(dets, list):
                json_entries[image] = [d for d in dets if isinstance(d, dict)]

    outputs: list[dict[str, Any]] = []
    for record in records:
        image = str(record.get("image") or "")
        dets: list[dict[str, Any]] = []

        if labels_dir is not None:
            stem = Path(image).stem
            txt_path = labels_dir / f"{stem}.txt"
            dets.extend(_parse_labels_txt(txt_path, float(args.conf_default)))

        mapped = json_entries.get(image) or json_entries.get(Path(image).name)
        if mapped:
            width = int(record.get("width") or 0)
            height = int(record.get("height") or 0)
            for det in mapped:
                if "bbox" in det and isinstance(det["bbox"], dict):
                    bbox = det["bbox"]
                    if all(k in bbox for k in ("cx", "cy", "w", "h")):
                        try:
                            dets.append(
                                {
                                    "class_id": int(det.get("class_id", det.get("cls", -1))),
                                    "score": float(det.get("score", det.get("conf", 0.0))),
                                    "bbox": {
                                        "cx": float(bbox["cx"]),
                                        "cy": float(bbox["cy"]),
                                        "w": float(bbox["w"]),
                                        "h": float(bbox["h"]),
                                    },
                                }
                            )
                        except (TypeError, ValueError, KeyError):
                            continue
                        continue
                xyxy = det.get("xyxy")
                if isinstance(xyxy, list):
                    norm = _xyxy_to_cxcywh_norm([float(v) for v in xyxy], width, height)
                    if norm is None:
                        continue
                    try:
                        class_id = int(det.get("class_id", det.get("cls", -1)))
                        score = float(det.get("score", det.get("conf", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    dets.append({"class_id": class_id, "score": score, "bbox": norm})

        outputs.append({"image": image, "detections": dets})

    validate_predictions_entries(outputs, strict=bool(args.strict))

    meta = _default_wrap_meta(adapter="yolov5", config=str(args.labels_dir or args.json or ""), images=len(outputs))
    meta["extra"] = {
        "exporter": "yolov5",
        "protocol_id": str(args.protocol),
        "dataset": str(args.dataset),
        "split": manifest["split"],
        "export_settings": {
            "imgsz": int(args.imgsz),
            "score_threshold": float(args.conf),
            "iou_threshold": float(args.iou),
            "max_detections": 300,
            "bbox_format": "cxcywh_norm",
            "protocol": str(args.protocol),
            "nms": {"applied": args.protocol == "nms_applied"},
            "preprocess": {
                "method": "letterbox" if bool(args.letterbox) else "resize",
                "input_color": "RGB",
                "normalize": "0_1",
                "resize_interp": "linear",
                "letterbox_fill": [114, 114, 114],
                "stride": int(args.stride),
            },
        },
        "runtime": {"platform": platform.system()},
    }

    payload = {"predictions": outputs, "meta": meta}
    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
