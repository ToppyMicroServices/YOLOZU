"""COCO dataset adapter (detection, instance segmentation, keypoints).

Supports the standard COCO directory layout::

    <root>/
       images/
           train2017/
           val2017/
       annotations/
           instances_train2017.json
           instances_val2017.json
           person_keypoints_train2017.json
           ...

Also supports YOLO-converted COCO layouts (images + labels directories).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .coco_convert import resolve_coco_file_path
from .registry import DatasetInfo, DatasetSample, register_adapter

__all__ = [
    "COCO_80_CLASSES",
    "COCOPaths",
    "resolve_coco_paths",
    "iter_coco_detection_samples",
    "COCOAdapter",
]


# -- class list (80 detection classes, 0-indexed) ---------------------------

COCO_80_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO uses non-contiguous category IDs (1-90). Map to 0-79.
_COCO_CAT_ID_TO_CLASS: dict[int, int] = {}
_COCO_CAT_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
for _idx, _cat_id in enumerate(_COCO_CAT_IDS):
    _COCO_CAT_ID_TO_CLASS[_cat_id] = _idx


# ---------------------------------------------------------------------------
# Layout resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class COCOPaths:
    """Resolved COCO directory layout."""

    root: Path
    images_root: Path
    annotations_dir: Path | None


def resolve_coco_paths(root: str | Path) -> COCOPaths:
    """Resolve COCO root to its canonical layout.

    Accepted structures::

        <root>/images/ + <root>/annotations/   (standard)
        <root>/train2017/ + <root>/annotations/ (images at root level)
        <root>/images/ (YOLO-converted, no annotations dir)
    """
    root_path = Path(root)

    images_root: Path | None = None
    annotations_dir: Path | None = None

    # Standard layout: root/images/ + root/annotations/
    if (root_path / "images").is_dir():
        images_root = root_path / "images"
    # Alternative: image split dirs directly under root
    elif any((root_path / d).is_dir() for d in ("train2017", "val2017", "test2017")):
        images_root = root_path

    if images_root is None:
        raise ValueError(f"COCO images directory not found under: {root_path}")

    if (root_path / "annotations").is_dir():
        annotations_dir = root_path / "annotations"

    return COCOPaths(root=root_path, images_root=images_root, annotations_dir=annotations_dir)


def _find_instances_json(annotations_dir: Path, split: str) -> Path | None:
    """Find the instances JSON for a given split."""
    candidates = [
        annotations_dir / f"instances_{split}.json",
        annotations_dir / f"instances_{split}2017.json",
        annotations_dir / f"{split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob for any instances_*.json
    for p in sorted(annotations_dir.glob("instances_*.json")):
        if split in p.stem:
            return p
    return None


def _find_split_images_dir(images_root: Path, split: str) -> Path | None:
    """Find the image directory for a split."""
    candidates = [
        images_root / split,
        images_root / f"{split}2017",
        images_root / f"{split}2014",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # images_root itself if it directly contains images
    if any(images_root.glob("*.jpg")):
        return images_root
    return None


# ---------------------------------------------------------------------------
# Detection sample iteration
# ---------------------------------------------------------------------------

def _build_category_map(instances: dict[str, Any]) -> dict[int, int]:
    """Build category_id -> contiguous class_id map from COCO JSON."""
    categories = instances.get("categories") or []
    if not categories:
        return _COCO_CAT_ID_TO_CLASS.copy()
    cat_ids = sorted(int(c["id"]) for c in categories if "id" in c)
    return {cat_id: idx for idx, cat_id in enumerate(cat_ids)}


def iter_coco_detection_samples(
    root: str | Path,
    *,
    split: str = "val",
    include_crowd: bool = False,
) -> Iterator[DatasetSample]:
    """Yield detection samples from a COCO-format dataset.

    Each sample's ``labels`` list contains YOLO-normalised bboxes::

        {"class_id": int, "cx": float, "cy": float, "w": float, "h": float}
    """
    paths = resolve_coco_paths(root)

    if paths.annotations_dir is None:
        raise ValueError(f"COCO annotations directory not found under: {paths.root}")

    instances_path = _find_instances_json(paths.annotations_dir, split)
    if instances_path is None:
        raise ValueError(
            f"COCO instances JSON not found for split={split!r} under: {paths.annotations_dir}"
        )

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    cat_map = _build_category_map(instances)

    # Build image_id -> image meta lookup
    image_meta: dict[int, dict[str, Any]] = {}
    for img in instances.get("images") or []:
        if not isinstance(img, dict) or "id" not in img:
            continue
        try:
            image_meta[int(img["id"])] = img
        except (ValueError, TypeError):
            continue

    # Group annotations by image
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in instances.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        if not include_crowd and int(ann.get("iscrowd", 0) or 0) == 1:
            continue
        try:
            img_id = int(ann["image_id"])
        except (KeyError, ValueError, TypeError):
            continue
        ann_by_image.setdefault(img_id, []).append(ann)

    # Find images directory for this split
    images_dir = _find_split_images_dir(paths.images_root, split)
    if images_dir is None:
        raise ValueError(f"COCO images directory for split={split!r} not found under: {paths.images_root}")

    # Yield samples
    for img_id, meta in sorted(image_meta.items(), key=lambda kv: str(kv[1].get("file_name", ""))):
        file_name = str(meta.get("file_name") or "").strip()
        if not file_name:
            continue
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        image_path = resolve_coco_file_path(images_dir, file_name)

        labels: list[dict[str, Any]] = []
        for ann in ann_by_image.get(img_id, []):
            try:
                cat_id = int(ann["category_id"])
            except (KeyError, ValueError, TypeError):
                continue
            class_id = cat_map.get(cat_id)
            if class_id is None:
                continue
            bbox = ann.get("bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, w, h = (float(v) for v in bbox)
            if w <= 0 or h <= 0:
                continue
            labels.append({
                "class_id": int(class_id),
                "cx": float((x + w / 2.0) / width),
                "cy": float((y + h / 2.0) / height),
                "w": float(w / width),
                "h": float(h / height),
            })

        sample_id = Path(file_name).stem
        yield DatasetSample(
            image_path=image_path,
            split=split,
            sample_id=sample_id,
            labels=labels,
            annotation_path=instances_path,
            extra={"image_hw": [height, width]},
        )


# ---------------------------------------------------------------------------
# Adapter class (registered in the global registry)
# ---------------------------------------------------------------------------

class COCOAdapter:
    """COCO dataset adapter for the unified registry."""

    format_name = "coco"

    def probe(self, root: Path) -> DatasetInfo | None:
        try:
            paths = resolve_coco_paths(root)
        except ValueError:
            return None

        splits: list[str] = []
        if paths.annotations_dir is not None:
            for p in sorted(paths.annotations_dir.glob("instances_*.json")):
                # instances_val2017.json -> val2017
                split_name = p.stem.replace("instances_", "")
                splits.append(split_name)

        if not splits:
            # Check for image-split directories
            for name in ("train2017", "val2017", "train", "val"):
                if _find_split_images_dir(paths.images_root, name) is not None:
                    splits.append(name)

        if not splits:
            return None

        return DatasetInfo(
            format_name=self.format_name,
            root=paths.root,
            splits=splits,
            task="detection",
            num_classes=80,
            class_names=COCO_80_CLASSES,
        )

    def iter_samples(
        self,
        root: Path,
        *,
        split: str = "val",
        **kwargs: Any,
    ) -> Iterator[DatasetSample]:
        yield from iter_coco_detection_samples(root, split=split, **kwargs)


# Auto-register on import.
register_adapter(COCOAdapter())
