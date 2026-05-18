"""COCO-to-YOLO label conversion utilities.

Converts COCO instances annotations into YOLO-format ``<class> cx cy w h``
label files, and builds a deterministic category ↔ class mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CocoCategoryMap",
    "build_category_map_from_coco",
    "resolve_coco_file_path",
    "yolo_line_from_coco_bbox",
    "convert_coco_instances_to_yolo_labels",
]


@dataclass(frozen=True)
class CocoCategoryMap:
    """Mapping between COCO category ids and contiguous YOLO class ids."""

    category_id_to_class_id: dict[int, int]
    class_id_to_category_id: dict[int, int]
    class_names: list[str]


def build_category_map_from_coco(annotations: dict) -> CocoCategoryMap:
    categories = annotations.get("categories") or []
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO annotations missing categories[]")

    # Sort by COCO category id for deterministic, explicit mapping.
    categories_sorted = sorted(categories, key=lambda c: int(c["id"]))
    category_id_to_class_id: dict[int, int] = {}
    class_id_to_category_id: dict[int, int] = {}
    class_names: list[str] = []
    for class_id, cat in enumerate(categories_sorted):
        cat_id = int(cat["id"])
        name = str(cat.get("name", cat_id))
        category_id_to_class_id[cat_id] = class_id
        class_id_to_category_id[class_id] = cat_id
        class_names.append(name)

    return CocoCategoryMap(
        category_id_to_class_id=category_id_to_class_id,
        class_id_to_category_id=class_id_to_category_id,
        class_names=class_names,
    )


def resolve_coco_file_path(images_dir: Path, file_name: str) -> Path:
    """Resolve a COCO image file_name under images_dir without traversal."""
    raw = str(file_name or "").strip()
    rel = Path(raw)
    if not raw or rel.is_absolute() or raw in {".", ".."} or ".." in rel.parts:
        raise ValueError(f"unsafe COCO file_name: {raw!r}")

    base = images_dir.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"unsafe COCO file_name: {raw!r}") from exc
    return candidate


def yolo_line_from_coco_bbox(
    *,
    class_id: int,
    coco_xywh: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> str | None:
    x, y, w, h = coco_xywh
    if image_w <= 0 or image_h <= 0:
        raise ValueError("invalid image size")
    if w <= 0.0 or h <= 0.0:
        return None

    cx = x + w / 2.0
    cy = y + h / 2.0

    cx_n = cx / float(image_w)
    cy_n = cy / float(image_h)
    w_n = w / float(image_w)
    h_n = h / float(image_h)

    # Basic sanity: keep normalized values within a reasonable envelope.
    if w_n <= 0.0 or h_n <= 0.0:
        return None
    if not (-0.5 <= cx_n <= 1.5 and -0.5 <= cy_n <= 1.5):
        return None

    return f"{int(class_id)} {cx_n:.6g} {cy_n:.6g} {w_n:.6g} {h_n:.6g}"


def convert_coco_instances_to_yolo_labels(
    *,
    instances_json: dict,
    images_dir: Path,
    labels_dir: Path,
    include_crowd: bool = False,
) -> CocoCategoryMap:
    images = instances_json.get("images") or []
    annotations = instances_json.get("annotations") or []
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("invalid COCO instances JSON (images/annotations)")

    cat_map = build_category_map_from_coco(instances_json)

    image_id_to_meta: dict[int, dict] = {}
    for img in images:
        image_id_to_meta[int(img["id"])] = img

    # Group annotations by image_id.
    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if not include_crowd and int(ann.get("iscrowd", 0)) == 1:
            continue
        img_id = int(ann["image_id"])
        ann_by_image.setdefault(img_id, []).append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_id, meta in image_id_to_meta.items():
        file_name = str(meta.get("file_name") or "")
        if not file_name:
            continue
        # Validate before writing labels so malformed COCO metadata cannot
        # leave partial outputs behind.
        _ = resolve_coco_file_path(images_dir, file_name)
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        stem = Path(file_name).stem

        # Write label file regardless of whether it has annotations (empty file is OK).
        lines: list[str] = []
        for ann in ann_by_image.get(img_id, []):
            cat_id = int(ann["category_id"])
            class_id = cat_map.category_id_to_class_id.get(cat_id)
            if class_id is None:
                continue
            bbox = ann.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            line = yolo_line_from_coco_bbox(
                class_id=class_id,
                coco_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                image_w=width,
                image_h=height,
            )
            if line is not None:
                lines.append(line)

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # Persist mapping for downstream inference implementations.
    mapping = {
        "category_id_to_class_id": cat_map.category_id_to_class_id,
        "class_id_to_category_id": cat_map.class_id_to_category_id,
        "class_names": cat_map.class_names,
    }
    (labels_dir / "classes.json").write_text(json.dumps(mapping, indent=2, sort_keys=True))
    (labels_dir / "classes.txt").write_text("\n".join(cat_map.class_names) + "\n")

    return cat_map
