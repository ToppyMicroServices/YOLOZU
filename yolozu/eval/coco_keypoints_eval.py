"""COCO-style OKS mAP evaluation (keypoints).

Parallels ``coco_eval.py`` but uses ``iouType='keypoints'`` and builds
annotation entries that carry the ``keypoints`` / ``num_keypoints`` fields
required by ``pycocotools.COCOeval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from yolozu.core.image_size import get_image_size
from yolozu.core.image_keys import add_image_aliases, lookup_image_alias, require_image_key
from yolozu.core.keypoints import normalize_keypoints

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)

__all__ = [
    "CocoKeypointsIndex",
    "COCO17_KPT_OKS_SIGMAS",
    "build_coco_keypoints_ground_truth",
    "predictions_to_coco_keypoints",
    "evaluate_coco_oks_map",
]


@dataclass(frozen=True)
class CocoKeypointsIndex:
    image_key_to_id: dict[str, int]
    class_id_to_category_id: dict[int, int]
    keypoints_count: int


COCO17_KPT_OKS_SIGMAS: list[float] = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]


def build_coco_keypoints_ground_truth(
    records: list[dict[str, Any]],
    *,
    keypoints_format: str = "xy_norm",
    keypoint_names: list[str] | None = None,
) -> tuple[dict[str, Any], CocoKeypointsIndex]:
    """Build COCO ground truth dict for keypoints evaluation.

    keypoints_format:
      - xy_norm: x,y in [0,1] normalized to image size
      - xy_abs:  x,y in pixels
    """

    if keypoints_format not in ("xy_norm", "xy_abs"):
        raise ValueError("keypoints_format must be 'xy_norm' or 'xy_abs'")

    k_count = 0
    max_class_id = -1
    for record in records:
        for label in record.get("labels", []) or []:
            if not isinstance(label, dict) or label.get("keypoints") is None:
                continue
            try:
                kps = normalize_keypoints(label.get("keypoints"), where="label.keypoints")
            except ValueError:
                continue
            k_count = max(k_count, int(len(kps)))
            try:
                max_class_id = max(max_class_id, int(label.get("class_id", -1)))
            except _NUMERIC_ERRORS:
                continue

    if keypoint_names is None:
        keypoint_names = [f"kp{i}" for i in range(k_count)]
    if len(keypoint_names) != int(k_count):
        raise ValueError(f"keypoint_names must have length {k_count}, got {len(keypoint_names)}")

    # COCO category ids are typically 1-based; keep that invariant.
    class_id_to_category_id = {cid: cid + 1 for cid in range(max_class_id + 1)} if max_class_id >= 0 else {}
    categories = [
        {"id": cid + 1, "name": str(cid), "keypoints": list(keypoint_names), "skeleton": []}
        for cid in range(max_class_id + 1)
    ]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_key_to_id: dict[str, int] = {}

    def to_px(x: float, y: float, *, width: int, height: int) -> tuple[float, float]:
        if keypoints_format == "xy_abs":
            return float(x), float(y)
        return float(x) * float(width), float(y) * float(height)

    def vis(v: Any) -> int:
        if v is None:
            return 2
        try:
            fv = float(v)
        except _NUMERIC_ERRORS:
            return 2
        if fv <= 0.0:
            return 0
        if fv < 1.5:
            return 1
        return 2

    ann_id = 1
    for idx, record in enumerate(records):
        image_id = int(idx + 1)
        image_key = require_image_key(record.get("image"), where=f"records[{idx}].image")
        image_path = Path(image_key)
        width, height = get_image_size(image_path)

        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        add_image_aliases(image_key_to_id, image_key, image_id)

        for label in record.get("labels", []) or []:
            if not isinstance(label, dict) or label.get("keypoints") is None:
                continue

            class_id = int(label.get("class_id", 0))
            category_id = class_id_to_category_id.get(class_id, class_id + 1)

            cx = float(label.get("cx", 0.0) or 0.0)
            cy = float(label.get("cy", 0.0) or 0.0)
            bw = float(label.get("w", 0.0) or 0.0)
            bh = float(label.get("h", 0.0) or 0.0)
            abs_w = bw * float(width)
            abs_h = bh * float(height)
            x = (cx * float(width)) - abs_w / 2.0
            y = (cy * float(height)) - abs_h / 2.0

            kps = normalize_keypoints(label.get("keypoints"), where="label.keypoints")
            out_kps: list[float] = []
            num_kps = 0
            for i in range(int(k_count)):
                if i < len(kps):
                    kp = kps[i]
                    xk, yk = to_px(float(kp["x"]), float(kp["y"]), width=width, height=height)
                    v = vis(kp.get("v"))
                else:
                    xk, yk, v = 0.0, 0.0, 0
                if v > 0:
                    num_kps += 1
                out_kps.extend([float(xk), float(yk), int(v)])

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(abs_w), float(abs_h)],
                    "area": float(max(0.0, abs_w) * max(0.0, abs_h)),
                    "iscrowd": 0,
                    "num_keypoints": int(num_kps),
                    "keypoints": out_kps,
                }
            )
            ann_id += 1

    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, CocoKeypointsIndex(
        image_key_to_id=image_key_to_id,
        class_id_to_category_id=class_id_to_category_id,
        keypoints_count=int(k_count),
    )


def predictions_to_coco_keypoints(
    predictions_entries: Iterable[dict[str, Any]],
    *,
    coco_index: CocoKeypointsIndex,
    image_sizes: dict[int, tuple[int, int]],
    keypoints_format: str = "xy_norm",
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert YOLOZU prediction entries to COCO keypoints detections list."""

    if keypoints_format not in ("xy_norm", "xy_abs"):
        raise ValueError("keypoints_format must be 'xy_norm' or 'xy_abs'")

    k_count = int(coco_index.keypoints_count)

    def to_px(x: float, y: float, *, width: int, height: int) -> tuple[float, float]:
        if keypoints_format == "xy_abs":
            return float(x), float(y)
        return float(x) * float(width), float(y) * float(height)

    def kp_score(kp: dict[str, Any]) -> float:
        s = kp.get("score")
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return float(s)
        v = kp.get("v")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
            if 0.0 <= fv <= 1.0:
                return fv
            if 0.0 <= fv <= 2.0:
                return fv / 2.0
        return 1.0

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
            if det.get("keypoints") is None:
                continue

            score = float(det.get("score", 0.0) or 0.0)
            if score < float(min_score):
                continue

            class_id = int(det.get("class_id", 0))
            category_id = coco_index.class_id_to_category_id.get(class_id, class_id + 1)

            bbox = det.get("bbox") or {}
            cx = float(bbox.get("cx", 0.0) or 0.0)
            cy = float(bbox.get("cy", 0.0) or 0.0)
            bw = float(bbox.get("w", 0.0) or 0.0)
            bh = float(bbox.get("h", 0.0) or 0.0)
            abs_w = bw * float(width)
            abs_h = bh * float(height)
            x = (cx * float(width)) - abs_w / 2.0
            y = (cy * float(height)) - abs_h / 2.0

            kps = normalize_keypoints(det.get("keypoints"), where="det.keypoints")
            out_kps: list[float] = []
            for i in range(int(k_count)):
                if i < len(kps):
                    kp = kps[i]
                    xk, yk = to_px(float(kp["x"]), float(kp["y"]), width=width, height=height)
                    sk = kp_score(kp)
                else:
                    xk, yk, sk = 0.0, 0.0, 0.0
                out_kps.extend([float(xk), float(yk), float(sk)])

            out.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(abs_w), float(abs_h)],
                    "keypoints": out_kps,
                    "score": float(score),
                }
            )

    return out


def evaluate_coco_oks_map(
    gt: dict[str, Any],
    dt: list[dict[str, Any]],
    *,
    sigmas: list[float] | None = None,
    max_dets: int = 20,
) -> dict[str, Any]:
    """Compute COCO-style OKS mAP using pycocotools if available."""

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pycocotools is required for OKS mAP evaluation. Install it (e.g. `python3 -m pip install pycocotools`)."
        ) from exc

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    if not dt:
        return {
            "metrics": {
                "oks_map50_95": 0.0,
                "oks_map50": 0.0,
                "oks_map75": 0.0,
                "oks_ar50_95": 0.0,
            },
            "stats": [],
        }

    coco_dt = coco_gt.loadRes(dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    if sigmas is not None:
        import numpy as np

        coco_eval.params.kpt_oks_sigmas = np.asarray(list(sigmas), dtype=np.float64)
    coco_eval.params.maxDets = [int(max_dets)]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = list(getattr(coco_eval, "stats", []))
    # For iouType="keypoints" stats indices are:
    # 0: AP@[.5:.95]  1: AP@.5  2: AP@.75  3: AP medium  4: AP large
    # 5: AR@[.5:.95]  6: AR@.5  7: AR@.75  8: AR medium  9: AR large
    metrics = {
        "oks_map50_95": float(stats[0]) if len(stats) > 0 else None,
        "oks_map50": float(stats[1]) if len(stats) > 1 else None,
        "oks_map75": float(stats[2]) if len(stats) > 2 else None,
        "oks_map_medium": float(stats[3]) if len(stats) > 3 else None,
        "oks_map_large": float(stats[4]) if len(stats) > 4 else None,
        "oks_ar50_95": float(stats[5]) if len(stats) > 5 else None,
        "oks_ar50": float(stats[6]) if len(stats) > 6 else None,
        "oks_ar75": float(stats[7]) if len(stats) > 7 else None,
        "oks_ar_medium": float(stats[8]) if len(stats) > 8 else None,
        "oks_ar_large": float(stats[9]) if len(stats) > 9 else None,
        "oks_max_dets": int(max_dets),
    }
    return {"metrics": metrics, "stats": stats}
