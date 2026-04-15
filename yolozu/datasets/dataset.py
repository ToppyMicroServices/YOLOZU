"""YOLO-format dataset loading and manifest building.

Reads images + labels directories, COCO instances JSON descriptors,
Ultralytics-style data YAML, and optional sidecar metadata to
produce a unified list of sample records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yolozu.core.config import simple_yaml_load
from .coco_convert import build_category_map_from_coco
from yolozu.core.keypoints import normalize_keypoints

__all__ = [
    "load_yolo_dataset",
    "load_coco_instances_dataset",
    "build_manifest",
    "inspect_dataset_layout",
]


@dataclass(frozen=True)
class DatasetLayoutInfo:
    format_name: str
    root: str
    split: str | None = None
    images_dir: str | None = None
    labels_dir: str | None = None
    data_yaml: str | None = None
    instances_json: str | None = None
    label_format: str | None = None
    split_candidates: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format_name,
            "root": self.root,
        }
        if self.split is not None:
            payload["split"] = self.split
        if self.images_dir is not None:
            payload["images_dir"] = self.images_dir
        if self.labels_dir is not None:
            payload["labels_dir"] = self.labels_dir
        if self.data_yaml is not None:
            payload["data_yaml"] = self.data_yaml
        if self.instances_json is not None:
            payload["instances_json"] = self.instances_json
        if self.label_format is not None:
            payload["label_format"] = self.label_format
        if self.split_candidates:
            payload["split_candidates"] = list(self.split_candidates)
        return payload

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def _first_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _resolve_optional_path(value: Any, root: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return list(value)
        if isinstance(value[0], (str, Path)):
            out = []
            for item in value:
                if item is None:
                    out.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = root / path
                out.append(str(path))
            return out
        return value
    if isinstance(value, str) or isinstance(value, Path):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return str(path)
    return value


def _load_sidecar_metadata(meta_path: Path, root: Path) -> dict[str, Any]:
    """Load optional per-image metadata from labels/<split>/<image>.json.

    This loader is intentionally lightweight: it resolves relative paths but does
    not decode masks/depth arrays. The resulting keys are merged into the record.
    """

    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[str, Any] = {}

    # Optional image size hints (used for pixel conversion).
    image_hw = data.get("image_hw") or data.get("hw")
    image_size = data.get("image_size")
    if image_hw is not None:
        out["image_hw"] = image_hw
    if image_size is not None:
        out["image_size"] = image_size

    # Pose / intrinsics.
    if "pose" in data:
        out["pose"] = data.get("pose")
    r_gt = _first_key(data, ("R_gt", "R"))
    t_gt = _first_key(data, ("t_gt", "t"))
    if r_gt is not None:
        out["R_gt"] = r_gt
    if t_gt is not None:
        out["t_gt"] = t_gt
    if "pose" not in out and (r_gt is not None or t_gt is not None):
        out["pose"] = {"R": r_gt, "t": t_gt}

    if "intrinsics" in data:
        out["intrinsics"] = data.get("intrinsics")
    k_gt = _first_key(data, ("K_gt", "K"))
    if k_gt is not None:
        out["K_gt"] = k_gt
        out["intrinsics"] = k_gt

    # Optional paths (kept as paths/inline values).
    mask_value = _first_key(data, ("mask_path", "M_path", "M", "mask"))
    depth_value = _first_key(data, ("depth_path", "D_obj_path", "D_obj", "depth"))
    # Optional mask metadata (instance/semantic segmentation helpers).
    if "mask_format" in data:
        out["mask_format"] = data.get("mask_format")
    if "mask_instances" in data:
        out["mask_instances"] = data.get("mask_instances")
    if "mask_classes" in data:
        out["mask_classes"] = data.get("mask_classes")
    if "mask_class_id" in data:
        out["mask_class_id"] = data.get("mask_class_id")
    if "mask_class_map" in data:
        out["mask_class_map"] = data.get("mask_class_map")
    if mask_value is not None:
        resolved = _resolve_optional_path(mask_value, root)
        out["mask_path"] = resolved
        out["mask"] = resolved
    if depth_value is not None:
        resolved = _resolve_optional_path(depth_value, root)
        out["depth_path"] = resolved
        out["depth"] = resolved
        out["D_obj"] = resolved

    cad_value = _first_key(data, ("cad_points", "cad_path", "cad_points_path"))
    if cad_value is not None:
        out["cad_points"] = _resolve_optional_path(cad_value, root)

    return out


def _as_numpy_mask(mask: Any) -> np.ndarray:
    import numpy as np

    if isinstance(mask, np.ndarray):
        return mask
    return np.asarray(mask)


def _load_mask_value(value: Any) -> Any:
    import numpy as np
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            return None
        try:
            with Image.open(path) as img:
                return np.asarray(img)
        except OSError:
            return None
    return value


def _parse_color_key(key: Any) -> Any:
    if isinstance(key, (int, float)):
        return int(key)
    if isinstance(key, str):
        if "," in key:
            parts = [int(p.strip()) for p in key.split(",")]
            return tuple(parts)
        if key.startswith("#") and len(key) == 7:
            r = int(key[1:3], 16)
            g = int(key[3:5], 16)
            b = int(key[5:7], 16)
            return (r, g, b)
        if key.isdigit():
            return int(key)
    return key


def _bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    return x_min, y_min, x_max, y_max


def _connected_components(mask_bool: np.ndarray) -> list[tuple[int, int, int, int]]:
    import numpy as np

    h, w = mask_bool.shape
    visited = np.zeros((h, w), dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            y_min = y_max = y
            x_min = x_max = x
            while stack:
                cy, cx = stack.pop()
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def _label_from_bbox(*, class_id: int, bbox: tuple[int, int, int, int], h: int, w: int) -> dict[str, Any]:
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max + 1) / 2.0 / w
    cy = (y_min + y_max + 1) / 2.0 / h
    bw = (x_max - x_min + 1) / w
    bh = (y_max - y_min + 1) / h
    return {"class_id": int(class_id), "cx": float(cx), "cy": float(cy), "w": float(bw), "h": float(bh)}


def _mask_to_yolo_labels(
    mask: Any,
    *,
    mask_format: str | None = None,
    class_map: dict[str, Any] | None = None,
    mask_class_id: int | None = None,
    instances: bool = False,
) -> list[dict[str, Any]]:
    """Derive YOLO-style bbox labels from a mask image or array.

    Supported patterns:
    - color mask (RGB): each unique non-black color is treated as a class
      - optional mask_class_map maps colors (\"r,g,b\" or \"#RRGGBB\") -> class_id
    - instance mask (single-channel IDs): each non-zero ID is treated as an instance
      - class_id comes from mask_class_id (fallback) or mask_class_map (id -> class_id)
    - class-id mask (single-channel IDs): each non-zero value is treated as a class_id (or remapped via class_map)

    When instances=True, connected components are split into multiple boxes (useful for binary masks).
    """

    class_map = class_map or {}
    import numpy as np

    mask_np = _as_numpy_mask(mask)

    # RGB / RGBA color masks.
    if mask_format == "color" or (mask_np.ndim == 3 and mask_np.shape[-1] in (3, 4)):
        if mask_np.ndim != 3:
            return []
        rgb = mask_np[..., :3]
        flat = rgb.reshape(-1, 3)
        unique = np.unique(flat, axis=0)
        colors = [tuple(c.tolist()) for c in unique]
        colors = [c for c in colors if c != (0, 0, 0)]
        mapping: dict[tuple[int, int, int], int] = {}
        if class_map:
            for k, v in class_map.items():
                key = _parse_color_key(k)
                if isinstance(key, tuple) and len(key) == 3:
                    mapping[tuple(int(x) for x in key)] = int(v)
        else:
            for idx, color in enumerate(sorted(colors)):
                mapping[color] = int(idx)

        labels: list[dict[str, Any]] = []
        h, w = mask_np.shape[:2]
        for color, class_id in mapping.items():
            mask_bool = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
            if instances:
                for bbox in _connected_components(mask_bool):
                    labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
            else:
                bbox = _bbox_from_mask(mask_bool)
                if bbox is None:
                    continue
                labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
        return labels

    # Numeric masks: instance IDs or class IDs.
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    if mask_np.ndim != 2:
        return []
    mask_int = mask_np.astype(int)
    unique = np.unique(mask_int)
    ids = [int(v) for v in unique if int(v) != 0]
    h, w = mask_int.shape[:2]

    labels = []
    fmt = (mask_format or "").lower()
    is_instance_mask = fmt in ("instance", "instances")
    fallback_class = int(mask_class_id) if mask_class_id is not None else 0
    for mask_id in ids:
        class_id = fallback_class
        if class_map:
            class_id = int(class_map.get(str(mask_id), class_map.get(mask_id, class_id)))
        elif not is_instance_mask:
            class_id = int(mask_id)
        mask_bool = mask_int == int(mask_id)
        if instances and not is_instance_mask:
            for bbox in _connected_components(mask_bool):
                labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
            continue
        bbox = _bbox_from_mask(mask_bool)
        if bbox is None:
            continue
        labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
    return labels


def _derive_labels_from_mask(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("mask_path") is None and record.get("mask") is None:
        return []

    import numpy as np

    mask_value = record.get("mask") or record.get("mask_path")
    instances = bool(record.get("mask_instances", False))
    class_map = record.get("mask_class_map") or {}
    mask_classes = record.get("mask_classes")
    mask_format = record.get("mask_format")
    mask_class_id = record.get("mask_class_id")

    if isinstance(mask_value, (list, tuple)) and mask_value and isinstance(mask_value[0], (str, Path)):
        labels: list[dict[str, Any]] = []
        class_ids = list(mask_classes) if isinstance(mask_classes, (list, tuple)) else list(range(len(mask_value)))
        for idx, mask_path in enumerate(mask_value):
            mask_data = _load_mask_value(mask_path)
            if mask_data is None:
                continue
            class_id = class_ids[idx] if idx < len(class_ids) else idx
            mask_np = _as_numpy_mask(mask_data)
            if mask_np.ndim == 3 and mask_np.shape[-1] in (3, 4):
                mask_bool = np.any(mask_np[..., :3] != 0, axis=-1)
            elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
                mask_bool = mask_np[..., 0] != 0
            else:
                mask_bool = mask_np != 0
            if mask_bool.ndim != 2:
                continue
            h, w = mask_bool.shape
            if instances:
                for bbox in _connected_components(mask_bool):
                    labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
            else:
                bbox = _bbox_from_mask(mask_bool)
                if bbox is None:
                    continue
                labels.append(_label_from_bbox(class_id=int(class_id), bbox=bbox, h=h, w=w))
        return labels

    mask_data = _load_mask_value(mask_value)
    if mask_data is None:
        return []
    return _mask_to_yolo_labels(
        mask_data,
        mask_format=str(mask_format) if mask_format is not None else None,
        class_map=class_map if isinstance(class_map, dict) else {},
        mask_class_id=int(mask_class_id) if mask_class_id is not None else None,
        instances=instances,
    )


def _bbox_from_poly(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs or not ys:
        return None
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    w = float(x_max - x_min)
    h = float(y_max - y_min)
    if w <= 0.0 or h <= 0.0:
        return None
    cx = float(x_min + x_max) / 2.0
    cy = float(y_min + y_max) / 2.0
    return cx, cy, w, h


def load_yolo_dataset(
    images_dir,
    labels_dir,
    *,
    dataset_root: Path | None = None,
    label_format: str | None = None,
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    dataset_root = Path(dataset_root) if dataset_root is not None else labels_dir.parent.parent
    label_format = str(label_format or "detect").strip().lower()
    images: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"):
        images.extend(images_dir.glob(ext))
    # Also match uppercase extensions (common on Windows / cameras).
    for ext in ("*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP", "*.GIF"):
        for p in images_dir.glob(ext):
            if p not in images:
                images.append(p)
    images = sorted(images)
    records = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        meta_path = labels_dir / f"{image_path.stem}.json"
        labels = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                class_id = int(float(parts[0]))
                if label_format in ("segment", "seg", "polygon", "yolo-seg", "yolo_seg"):
                    values = [float(value) for value in parts[1:]]
                    # Support both:
                    #  A) class + poly(x1 y1 x2 y2 ...)
                    #  B) class + bbox(cx cy w h) + poly(...)
                    if len(values) >= 10 and (len(values) - 4) >= 6 and (len(values) - 4) % 2 == 0:
                        poly = values[4:]
                    else:
                        poly = values
                    bbox = _bbox_from_poly(poly)
                    if bbox is None:
                        raise ValueError(f"invalid segmentation label line: {line}")
                    cx, cy, bw, bh = bbox
                    label: dict[str, Any] = {
                        "class_id": class_id,
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(bw),
                        "h": float(bh),
                        "polygon": [float(v) for v in poly],
                    }
                    labels.append(label)
                    continue

                if len(parts) < 5:
                    raise ValueError(f"invalid label line: {line}")

                coords = [float(value) for value in parts[1:5]]
                label = {"class_id": class_id, "cx": coords[0], "cy": coords[1], "w": coords[2], "h": coords[3]}

                # Optional: YOLO pose-style keypoints appended to the label line.
                # Supported:
                #   - 5 + 2*K: x1 y1 x2 y2 ...
                #   - 5 + 3*K: x1 y1 v1 x2 y2 v2 ...
                extra = parts[5:]
                if extra:
                    try:
                        extra_f = [float(v) for v in extra]
                    except Exception as exc:
                        raise ValueError(f"invalid label keypoints: {line}") from exc
                    label["keypoints"] = normalize_keypoints(extra_f, where="label.keypoints")

                labels.append(label)
        record = {"image": str(image_path), "labels": labels}
        record.update(_load_sidecar_metadata(meta_path, dataset_root))
        if not record["labels"]:
            record["labels"] = _derive_labels_from_mask(record)
        records.append(record)
    return records


def _pick_split(dataset_root: Path, split: str | None) -> str:
    if split:
        return split
    images_root = dataset_root / "images"
    if not images_root.exists():
        return "train2017"

    # Prefer common validation splits when present.
    candidates = (
        "val2017",
        "val",
        "valid",
        "validation",
        "train2017",
        "train",
    )
    for candidate in candidates:
        if (images_root / candidate).exists():
            return candidate

    # Fallback: first directory under images/.
    try:
        child_entries = sorted(images_root.iterdir())
    except OSError:
        child_entries = []
    for child in child_entries:
        if child.is_dir():
            return child.name
    return "train2017"


def _resolve_ultralytics_data_yaml(
    config_path: Path,
    split: str | None,
) -> tuple[Path, Path, str, Path, str | None] | None:
    """Resolve images/labels directories from an Ultralytics-style data.yaml.

    Supports the common pattern:
      path: /abs/or/rel/root
      train: images/train
      val: images/val

    Where train/val point to `.../images/<split>` directories, and labels live under
    `.../labels/<split>`.
    """

    if not config_path.exists() or not config_path.is_file():
        return None
    if config_path.suffix.lower() not in (".yaml", ".yml"):
        return None

    text = config_path.read_text(encoding="utf-8")
    data: Any | None = None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = None
    if data is None:
        # Minimal fallback parser (keeps yolozu migrate usable even if PyYAML is missing).
        try:
            data = simple_yaml_load(text)
        except (OSError, ValueError):
            return None
    if not isinstance(data, dict):
        return None

    label_format: str | None = None
    raw_task = data.get("task") or data.get("label_format")
    if isinstance(raw_task, str) and raw_task.strip():
        lowered = raw_task.strip().lower()
        if lowered in ("detect", "det", "bbox"):
            label_format = "detect"
        elif lowered in ("pose", "keypoints"):
            label_format = "detect"
        elif lowered in ("segment", "seg", "polygon", "yolo-seg", "yolo_seg"):
            label_format = "segment"

    base_raw = data.get("path")
    base = Path(str(base_raw)) if isinstance(base_raw, str) and str(base_raw).strip() else config_path.parent
    if not base.is_absolute():
        base = (config_path.parent / base).resolve()

    def _resolve_images_dir(value: Any) -> Path | None:
        if not (isinstance(value, str) and str(value).strip()):
            return None
        p = Path(str(value).strip())
        if not p.is_absolute():
            p = base / p
        return p.resolve()

    def _infer_layout(images_dir: Path) -> tuple[Path, Path, str, Path] | None:
        if images_dir.parent.name != "images":
            return None
        dataset_root = images_dir.parent.parent
        split_name = images_dir.name
        labels_dir = dataset_root / "labels" / split_name
        return images_dir, labels_dir, split_name, dataset_root

    layouts: dict[str, tuple[Path, Path, str, Path]] = {}
    train_dir = _resolve_images_dir(data.get("train"))
    val_dir = _resolve_images_dir(data.get("val"))
    if train_dir is not None:
        layout = _infer_layout(train_dir)
        if layout is not None:
            layouts["train"] = layout
            layouts[layout[2]] = layout
    if val_dir is not None:
        layout = _infer_layout(val_dir)
        if layout is not None:
            layouts["val"] = layout
            layouts[layout[2]] = layout

    if split:
        key = str(split)
        if key in layouts:
            img, lbl, split_name, root = layouts[key]
            return img, lbl, split_name, root, label_format
        lowered = key.lower()
        if lowered in ("train", "tr"):
            resolved = layouts.get("train")
            if resolved is None:
                return None
            img, lbl, split_name, root = resolved
            return img, lbl, split_name, root, label_format
        if lowered in ("val", "valid", "validation", "eval"):
            resolved = layouts.get("val")
            if resolved is None:
                return None
            img, lbl, split_name, root = resolved
            return img, lbl, split_name, root, label_format

    resolved = layouts.get("val") or layouts.get("train")
    if resolved is None:
        return None
    img, lbl, split_name, root = resolved
    return img, lbl, split_name, root, label_format


def _resolve_dataset_json_layout(dataset_root: Path, split: str | None) -> tuple[Path, Path, str, str | None] | None:
    """Resolve images/labels directories from a dataset.json descriptor (optional).

    Some prepare tools write a dataset.json with absolute paths to avoid copying
    large image trees. When present, prefer it as a hint for build_manifest().
    """

    descriptor = dataset_root / "dataset.json"
    if not descriptor.exists():
        return None
    try:
        data = json.loads(descriptor.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    images_dir = data.get("images_dir")
    labels_dir = data.get("labels_dir")
    split_from_desc = data.get("split")
    label_format = data.get("label_format") or data.get("task")
    if not images_dir or not labels_dir:
        return None

    img = Path(images_dir)
    lbl = Path(labels_dir)
    if not img.is_absolute():
        img = dataset_root / img
    if not lbl.is_absolute():
        lbl = dataset_root / lbl

    effective_split = str(split_from_desc) if split_from_desc else None
    if split:
        effective_split = str(split)
        img_candidate = img.parent / effective_split
        lbl_candidate = lbl.parent / effective_split
        if img_candidate.exists():
            img = img_candidate
        if lbl_candidate.exists():
            lbl = lbl_candidate
    if effective_split is None:
        # Best-effort: infer from labels dir name.
        effective_split = lbl.name

    label_format_out = None
    if isinstance(label_format, str) and label_format.strip():
        label_format_out = label_format.strip()

    return img, lbl, effective_split, label_format_out


def _available_yolo_splits(dataset_root: Path) -> list[str]:
    images_root = dataset_root / "images"
    if not images_root.exists() or not images_root.is_dir():
        return []
    splits: list[str] = []
    try:
        for child in sorted(images_root.iterdir()):
            if child.is_dir():
                splits.append(child.name)
    except OSError:
        return []
    return splits


def _available_coco_splits(dataset_root: Path) -> list[str]:
    ann_root = dataset_root / "annotations"
    if not ann_root.exists() or not ann_root.is_dir():
        return []
    splits: list[str] = []
    try:
        for child in sorted(ann_root.glob("instances_*.json")):
            name = child.stem
            if name.startswith("instances_"):
                splits.append(name[len("instances_") :])
    except OSError:
        return []
    return splits


def _resolve_coco_root_layout(
    dataset_root: Path,
    split: str | None,
) -> tuple[Path, Path, str] | None:
    ann_root = dataset_root / "annotations"
    if not ann_root.exists() or not ann_root.is_dir():
        return None

    requested = str(split).strip() if split is not None else ""
    candidates = [requested] if requested else []
    if not candidates:
        candidates = _available_coco_splits(dataset_root)
        preferred = ("val2017", "val", "train2017", "train")
        ordered: list[str] = []
        for item in preferred:
            if item in candidates and item not in ordered:
                ordered.append(item)
        for item in candidates:
            if item not in ordered:
                ordered.append(item)
        candidates = ordered
    for split_name in candidates:
        if not split_name:
            continue
        instances_path = ann_root / f"instances_{split_name}.json"
        if not instances_path.exists():
            continue
        images_dir = dataset_root / "images" / split_name
        if not images_dir.exists():
            fallback = dataset_root / split_name
            if fallback.exists():
                images_dir = fallback
            else:
                continue
        return images_dir, instances_path, split_name
    return None


def inspect_dataset_layout(dataset_root, *, split: str | None = None) -> dict[str, Any] | None:
    path = Path(dataset_root)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if path.is_file():
        if path.suffix.lower() in (".yaml", ".yml"):
            resolved = _resolve_ultralytics_data_yaml(path, split)
            if resolved is not None:
                images_dir, labels_dir, split_effective, root, label_format = resolved
                return DatasetLayoutInfo(
                    format_name="ultralytics_data_yaml",
                    root=str(root),
                    split=split_effective,
                    images_dir=str(images_dir),
                    labels_dir=str(labels_dir),
                    data_yaml=str(path),
                    label_format=label_format,
                    split_candidates=_available_yolo_splits(root),
                ).to_dict()
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict):
                fmt = str(data.get("format") or "").strip().lower()
                if path.name == "dataset.json":
                    if fmt in ("coco_instances", "coco-instances", "coco"):
                        return DatasetLayoutInfo(
                            format_name="yolozu_coco_wrapper",
                            root=str(path.parent),
                            split=str(data.get("split") or split or ""),
                            images_dir=str(data.get("images_dir") or ""),
                            instances_json=str(data.get("instances_json") or ""),
                            split_candidates=([str(data.get("split"))] if data.get("split") else None),
                        ).to_dict()
                    resolved = _resolve_dataset_json_layout(path.parent, split)
                    if resolved is not None:
                        images_dir, labels_dir, split_effective, label_format = resolved
                        return DatasetLayoutInfo(
                            format_name="yolozu_wrapper",
                            root=str(path.parent),
                            split=split_effective,
                            images_dir=str(images_dir),
                            labels_dir=str(labels_dir),
                            label_format=label_format,
                            split_candidates=_available_yolo_splits(path.parent),
                        ).to_dict()
        return None

    desc = path / "dataset.json"
    if desc.exists():
        info = inspect_dataset_layout(desc, split=split)
        if info is not None:
            return info

    data_yaml = path / "data.yaml"
    if data_yaml.exists():
        info = inspect_dataset_layout(data_yaml, split=split)
        if info is not None:
            return info

    coco_layout = _resolve_coco_root_layout(path, split)
    if coco_layout is not None:
        images_dir, instances_path, split_effective = coco_layout
        return DatasetLayoutInfo(
            format_name="coco_root",
            root=str(path),
            split=split_effective,
            images_dir=str(images_dir),
            instances_json=str(instances_path),
            split_candidates=_available_coco_splits(path),
        ).to_dict()

    split_effective = _pick_split(path, split)
    images_dir = path / "images" / split_effective
    labels_dir = path / "labels" / split_effective
    if images_dir.exists() and labels_dir.exists():
        return DatasetLayoutInfo(
            format_name="yolo_layout",
            root=str(path),
            split=split_effective,
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            split_candidates=_available_yolo_splits(path),
        ).to_dict()

    return None


def _resolve_path_from_descriptor(value: Any, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value.strip())
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def load_coco_instances_dataset(
    instances_json: dict[str, Any],
    *,
    images_dir: Path,
    include_crowd: bool = False,
) -> list[dict[str, Any]]:
    images = instances_json.get("images") or []
    annotations = instances_json.get("annotations") or []
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("invalid COCO instances JSON (images/annotations)")

    cat_map = build_category_map_from_coco(instances_json)

    image_id_to_meta: dict[int, dict[str, Any]] = {}
    for img in images:
        if not isinstance(img, dict) or "id" not in img:
            continue
        try:
            image_id_to_meta[int(img["id"])] = img
        except (TypeError, ValueError):
            continue

    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if not include_crowd and int(ann.get("iscrowd", 0) or 0) == 1:
            continue
        try:
            img_id = int(ann["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        ann_by_image.setdefault(img_id, []).append(ann)

    records: list[dict[str, Any]] = []
    for img_id, meta in sorted(image_id_to_meta.items(), key=lambda kv: str(kv[1].get("file_name") or "")):
        file_name = str(meta.get("file_name") or "").strip()
        if not file_name:
            continue
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        image_path = images_dir / file_name

        labels: list[dict[str, Any]] = []
        for ann in ann_by_image.get(img_id, []):
            try:
                cat_id = int(ann["category_id"])
            except Exception:
                continue
            class_id = cat_map.category_id_to_class_id.get(cat_id)
            if class_id is None:
                continue
            bbox = ann.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if w <= 0.0 or h <= 0.0:
                continue
            cx = (x + w / 2.0) / float(width)
            cy = (y + h / 2.0) / float(height)
            wn = w / float(width)
            hn = h / float(height)
            labels.append({"class_id": int(class_id), "cx": float(cx), "cy": float(cy), "w": float(wn), "h": float(hn)})

        records.append(
            {
                "image": str(image_path),
                "labels": labels,
                "image_hw": [int(height), int(width)],
            }
        )

    return records


def _build_manifest_from_dataset_descriptor(
    descriptor_path: Path,
    *,
    split: str | None,
    label_format: str | None,
) -> dict[str, Any] | None:
    if not descriptor_path.exists() or not descriptor_path.is_file() or descriptor_path.suffix.lower() != ".json":
        return None
    try:
        data = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    base = descriptor_path.parent
    fmt = str(data.get("format") or "").strip().lower()

    if fmt in ("coco_instances", "coco-instances", "coco"):
        images_dir = _resolve_path_from_descriptor(data.get("images_dir"), base=base)
        instances_path = _resolve_path_from_descriptor(data.get("instances_json"), base=base)
        if images_dir is None or instances_path is None:
            return None
        if not instances_path.exists():
            raise FileNotFoundError(f"COCO instances JSON not found: {instances_path}")
        instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
        if not isinstance(instances_doc, dict):
            raise ValueError("invalid COCO instances JSON (expected object)")
        records = load_coco_instances_dataset(
            instances_doc,
            images_dir=images_dir,
            include_crowd=bool(data.get("include_crowd", False)),
        )
        split_effective = str(split) if split else str(data.get("split") or "")
        return {"images": records, "split": split_effective or None}

    # Default: YOLO wrapper descriptor.
    resolved = _resolve_dataset_json_layout(base, split)
    if resolved is not None and descriptor_path.name == "dataset.json":
        images_dir, labels_dir, split_effective, json_label_format = resolved
        records = load_yolo_dataset(
            images_dir,
            labels_dir,
            dataset_root=base,
            label_format=label_format or json_label_format,
        )
        return {"images": records, "split": split_effective}

    images_dir = _resolve_path_from_descriptor(data.get("images_dir"), base=base)
    labels_dir = _resolve_path_from_descriptor(data.get("labels_dir"), base=base)
    if images_dir is None or labels_dir is None:
        return None

    split_from_desc = data.get("split")
    effective_split: str | None = str(split_from_desc) if isinstance(split_from_desc, str) and split_from_desc else None
    if split:
        effective_split = str(split)
        img_candidate = images_dir.parent / effective_split
        lbl_candidate = labels_dir.parent / effective_split
        if img_candidate.exists():
            images_dir = img_candidate
        if lbl_candidate.exists():
            labels_dir = lbl_candidate
    if effective_split is None:
        effective_split = labels_dir.name

    json_label_format = data.get("label_format") or data.get("task")
    lf_out = str(json_label_format).strip() if isinstance(json_label_format, str) and json_label_format.strip() else None
    records = load_yolo_dataset(
        images_dir,
        labels_dir,
        dataset_root=base,
        label_format=label_format or lf_out,
    )
    return {"images": records, "split": effective_split}


def build_manifest(dataset_root, *, split: str | None = None, label_format: str | None = None):
    dataset_root = Path(dataset_root)
    if dataset_root.exists() and dataset_root.is_file():
        if dataset_root.suffix.lower() == ".json":
            resolved = _build_manifest_from_dataset_descriptor(dataset_root, split=split, label_format=label_format)
            if resolved is not None:
                return resolved
        resolved = _resolve_ultralytics_data_yaml(dataset_root, split)
        if resolved is not None:
            images_dir, labels_dir, split_effective, inferred_root, yaml_label_format = resolved
            records = load_yolo_dataset(
                images_dir,
                labels_dir,
                dataset_root=inferred_root,
                label_format=label_format or yaml_label_format,
            )
            return {"images": records, "split": split_effective}
        raise FileNotFoundError(f"unsupported dataset descriptor file: {dataset_root}")

    # dataset.json descriptor wrapper (YOLO wrapper or COCO wrapper).
    desc = dataset_root / "dataset.json"
    resolved = _build_manifest_from_dataset_descriptor(desc, split=split, label_format=label_format)
    if resolved is not None:
        return resolved

    coco_layout = _resolve_coco_root_layout(dataset_root, split)
    if coco_layout is not None:
        images_dir, instances_path, split_effective = coco_layout
        instances_doc = json.loads(instances_path.read_text(encoding="utf-8"))
        if not isinstance(instances_doc, dict):
            raise ValueError("invalid COCO instances JSON (expected object)")
        records = load_coco_instances_dataset(instances_doc, images_dir=images_dir, include_crowd=False)
        return {"images": records, "split": split_effective}

    split_effective = _pick_split(dataset_root, split)
    images_dir = dataset_root / "images" / split_effective
    labels_dir = dataset_root / "labels" / split_effective
    records = load_yolo_dataset(images_dir, labels_dir, dataset_root=dataset_root, label_format=label_format)
    return {"images": records, "split": split_effective}
