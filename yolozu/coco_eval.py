"""COCO-style mAP evaluation (bbox).

Builds GT from YOLOZU dataset records, converts prediction entries to COCO
detections, then delegates to ``pycocotools.COCOeval`` for metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .image_size import get_image_size
from .image_keys import add_image_aliases, lookup_image_alias, require_image_key

__all__ = [
    "CocoIndex",
    "build_coco_ground_truth",
    "predictions_to_coco_detections",
    "evaluate_coco_map",
]


@dataclass(frozen=True)
class CocoIndex:
    image_key_to_id: dict[str, int]
    class_id_to_category_id: dict[int, int]


def build_coco_ground_truth(records: list[dict[str, Any]]) -> tuple[dict[str, Any], CocoIndex]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    max_class_id = -1
    for record in records:
        for label in record.get("labels", []) or []:
            try:
                max_class_id = max(max_class_id, int(label.get("class_id", -1)))
            except Exception:
                continue

    # COCO category ids are typically 1-based; keep that invariant.
    class_id_to_category_id = {cid: cid + 1 for cid in range(max_class_id + 1)}
    categories = [{"id": cid + 1, "name": str(cid)} for cid in range(max_class_id + 1)]

    image_key_to_id: dict[str, int] = {}
    ann_id = 1
    for idx, record in enumerate(records):
        image_id = int(idx + 1)
        image_key = require_image_key(record.get("image"), where=f"records[{idx}].image")
        image_path = Path(image_key)
        width, height = get_image_size(image_path)

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        add_image_aliases(image_key_to_id, image_key, image_id)

        for label in record.get("labels", []) or []:
            class_id = int(label["class_id"])
            category_id = class_id_to_category_id.get(class_id, class_id + 1)

            bbox = _yolo_norm_cxcywh_to_abs_xywh(
                (float(label["cx"]), float(label["cy"]), float(label["w"]), float(label["h"])),
                width=width,
                height=height,
            )
            x, y, w, h = bbox
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": float(max(0.0, w) * max(0.0, h)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, CocoIndex(image_key_to_id=image_key_to_id, class_id_to_category_id=class_id_to_category_id)


def predictions_to_coco_detections(
    predictions_entries: Iterable[dict[str, Any]],
    *,
    coco_index: CocoIndex,
    image_sizes: dict[int, tuple[int, int]],
    bbox_format: str = "cxcywh_norm",
) -> list[dict[str, Any]]:
    """Convert YOLOZU prediction entries to COCO detections list.

    bbox_format:
      - cxcywh_norm: bbox dict {cx,cy,w,h} in [0,1] relative to image size
      - cxcywh_abs:  bbox dict {cx,cy,w,h} in pixels
      - xywh_abs:    bbox dict {x,y,w,h} in pixels (top-left origin)
      - xyxy_abs:    bbox dict {x1,y1,x2,y2} in pixels
    """

    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(predictions_entries):
        where = f"predictions[{idx}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")

        image_key = require_image_key(entry.get("image"), where=f"{where}.image")
        image_id = lookup_image_alias(coco_index.image_key_to_id, image_key)
        if image_id is None:
            raise ValueError(f"prediction refers to unknown image: {image_key}")
        if image_id not in image_sizes:
            raise ValueError(f"missing image size for image_id={image_id}")

        width, height = image_sizes[image_id]
        detections = entry.get("detections", [])
        if detections is None:
            detections = []
        if not isinstance(detections, list):
            raise ValueError(f"{where}.detections must be a list")
        for det_idx, det in enumerate(detections):
            if not isinstance(det, dict):
                raise ValueError(f"{where}.detections[{det_idx}] must be an object")
            if "class_id" not in det:
                raise ValueError(f"missing class_id for image {image_key}")
            class_id = int(det["class_id"])
            category_id = coco_index.class_id_to_category_id.get(class_id, class_id + 1)
            score = float(det.get("score", 0.0))

            bbox = det.get("bbox")
            if bbox is None:
                raise ValueError(f"missing bbox for image {image_key}")

            x, y, w, h = _to_abs_xywh(bbox, width=width, height=height, bbox_format=bbox_format)
            out.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "score": score,
                }
            )

    return out


def evaluate_coco_map(gt: dict[str, Any], dt: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute COCO-style mAP using pycocotools if available."""

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pycocotools is required for COCO mAP evaluation. Install it (e.g. `python3 -m pip install pycocotools`)."
        ) from exc

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    if not dt:
        return {
            "metrics": {
                "map50_95": 0.0,
                "map50": 0.0,
                "map75": 0.0,
                "ar100": 0.0,
            },
            "stats": [],
        }

    coco_dt = coco_gt.loadRes(dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = list(getattr(coco_eval, "stats", []))
    # COCOeval.stats:
    #  0: AP@[.5:.95]  1: AP@.5  2: AP@.75  3: AP small  4: AP medium  5: AP large
    #  6: AR@1         7: AR@10  8: AR@100  9: AR small 10: AR medium 11: AR large
    metrics = {
        "map50_95": float(stats[0]) if len(stats) > 0 else None,
        "map50": float(stats[1]) if len(stats) > 1 else None,
        "map75": float(stats[2]) if len(stats) > 2 else None,
        "ar100": float(stats[8]) if len(stats) > 8 else None,
    }
    return {"metrics": metrics, "stats": stats}


def _yolo_norm_cxcywh_to_abs_xywh(
    bbox: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, w, h = bbox
    abs_w = w * width
    abs_h = h * height
    x = (cx * width) - abs_w / 2.0
    y = (cy * height) - abs_h / 2.0
    return float(x), float(y), float(abs_w), float(abs_h)


def _to_abs_xywh(bbox: Any, *, width: int, height: int, bbox_format: str) -> tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        if bbox_format == "cxcywh_norm":
            return _yolo_norm_cxcywh_to_abs_xywh(
                (float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])),
                width=width,
                height=height,
            )
        if bbox_format == "cxcywh_abs":
            cx, cy, w, h = float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])
            return float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)
        if bbox_format == "xywh_abs":
            return float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
        if bbox_format == "xyxy_abs":
            x1, y1, x2, y2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
            return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    if isinstance(bbox, list) and len(bbox) == 4:
        # Common export format: [x, y, w, h] in pixels.
        if bbox_format != "xywh_abs":
            raise ValueError("bbox is a list; use --bbox-format xywh_abs")
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    raise ValueError(f"unsupported bbox format ({bbox_format}) or shape")
