import json
from pathlib import Path

from yolozu.datasets.dataset_contract import normalize_label_bbox, normalize_record_bboxes

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _load_yolo_labels(label_path, *, label_format=None):
    labels = []
    if not label_path.exists():
        return labels
    label_format = str(label_format or "detect").strip().lower()
    for line in label_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"invalid label line: {line}")
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
        if label_format in ("obb", "rotated_bbox", "rotated-bbox"):
            if len(coords) == 5:
                labels.append(
                    normalize_label_bbox(
                        {
                            "class_id": class_id,
                            "bbox": {"format": "cxcywh_norm", "cx": coords[0], "cy": coords[1], "w": coords[2], "h": coords[3]},
                            "angle": coords[4],
                        },
                        bbox_field="cxcywh_norm",
                    )
                )
                continue
            if len(coords) >= 8 and len(coords) % 2 == 0:
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                labels.append(
                    normalize_label_bbox(
                        {
                            "class_id": class_id,
                            "bbox": {
                                "format": "cxcywh_norm",
                                "cx": (x_min + x_max) / 2.0,
                                "cy": (y_min + y_max) / 2.0,
                                "w": x_max - x_min,
                                "h": y_max - y_min,
                            },
                            "obb": coords,
                        },
                        bbox_field="cxcywh_norm",
                    )
                )
                continue
            raise ValueError(f"invalid label line: {line}")
        if len(parts) != 5 and (len(parts) - 5) % 3 != 0:
            raise ValueError(f"invalid label line: {line}")
        entry = {
            "class_id": class_id,
            "bbox": {
                "format": "cxcywh_norm",
                "cx": coords[0],
                "cy": coords[1],
                "w": coords[2],
                "h": coords[3],
            },
        }
        if len(parts) > 5:
            kps = []
            extras = coords[4:]
            for i in range(0, len(extras), 3):
                x = float(extras[i + 0])
                y = float(extras[i + 1])
                v = float(extras[i + 2])
                kps.append({"x": x, "y": y, "v": v})
            entry["keypoints"] = kps
        labels.append(normalize_label_bbox(entry, bbox_field="cxcywh_norm"))
    return labels


def _resolve_optional_path(value, root):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return list(value)
        # If this is a list of paths (strings), resolve each element.
        if isinstance(value[0], (str, Path)):
            resolved = []
            for item in value:
                if item is None:
                    resolved.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = Path(root) / path
                resolved.append(str(path))
            return resolved
        # Otherwise keep as-is (likely an inline array).
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _resolve_optional_value(value, root):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return list(value)
        # List of relative/absolute paths.
        if isinstance(value[0], (str, Path)):
            resolved = []
            for item in value:
                if item is None:
                    resolved.append(None)
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = Path(root) / path
                resolved.append(str(path))
            return resolved
        return value
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)


def _first_key(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_metadata(meta_path, root):
    if not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text())
    metadata = {}
    mask_value = _first_key(data, ("mask_path", "M_path", "M", "mask"))
    depth_value = _first_key(data, ("depth_path", "D_obj_path", "D_obj", "depth"))
    if "mask_format" in data:
        metadata["mask_format"] = data["mask_format"]
    if "mask_class_map" in data:
        metadata["mask_class_map"] = data["mask_class_map"]
    if "mask_class_id" in data:
        metadata["mask_class_id"] = data["mask_class_id"]
    if mask_value is not None:
        resolved = _resolve_optional_value(mask_value, root)
        # Keep using the legacy key name `mask_path` because the validator treats it as
        # "path OR 2D array". This preserves strict-mode behavior for inline masks.
        metadata["mask_path"] = resolved
        metadata["mask"] = resolved
    if depth_value is not None:
        resolved = _resolve_optional_value(depth_value, root)
        metadata["depth_path"] = resolved
        metadata["depth"] = resolved

    image_hw = data.get("image_hw") or data.get("hw")
    if image_hw is not None:
        metadata["image_hw"] = image_hw
    if data.get("image_size") is not None:
        metadata["image_size"] = data.get("image_size")

    if "pose" in data:
        metadata["pose"] = data["pose"]
    r_gt = _first_key(data, ("R_gt", "R"))
    t_gt = _first_key(data, ("t_gt", "t"))
    if r_gt is not None:
        metadata["R_gt"] = r_gt
    if t_gt is not None:
        metadata["t_gt"] = t_gt
    if "pose" not in metadata and (r_gt is not None or t_gt is not None):
        metadata["pose"] = {"R": r_gt, "t": t_gt}

    if "mask_instances" in data:
        metadata["mask_instances"] = bool(data.get("mask_instances"))
    if "mask_classes" in data:
        metadata["mask_classes"] = data.get("mask_classes")
    if "mask_class_map" in data:
        metadata["mask_class_map"] = data.get("mask_class_map")

    if "intrinsics" in data:
        metadata["intrinsics"] = data["intrinsics"]
    k_gt = _first_key(data, ("K_gt", "K"))
    if k_gt is not None:
        metadata["K_gt"] = k_gt
        metadata["intrinsics"] = k_gt

    k_gt_prime = _first_key(data, ("K_gt_prime", "K_gt'"))
    if k_gt_prime is not None:
        metadata["K_gt_prime"] = k_gt_prime
        metadata["intrinsics_prime"] = k_gt_prime
    cad_value = data.get("cad_points") or data.get("cad_path") or data.get("cad_points_path")
    if cad_value is not None:
        metadata["cad_points"] = _resolve_optional_value(cad_value, root)
    return metadata


def _load_mask_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        if suffix == ".png":
            try:
                from PIL import Image
            except Exception as exc:
                raise SystemExit(
                    "Pillow is required for PNG masks. Install it (e.g. pip install Pillow) or use .npy/.json."
                ) from exc
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            else:
                img = img.copy()
            if np is None:
                width, height = img.size
                data = list(img.getdata())
                if img.mode in ("RGB", "RGBA"):
                    step = 3 if img.mode == "RGB" else 4
                    rows = [data[i * width : (i + 1) * width] for i in range(height)]
                    return [
                        [list(pixel[:step]) for pixel in row]
                        for row in rows
                    ]
                return [data[i * width : (i + 1) * width] for i in range(height)]
            return np.asarray(img)
        if suffix in (".npy", ".npz"):
            if np is None:
                return None
            loaded = np.load(path)
            if hasattr(loaded, "files"):
                if not loaded.files:
                    return None
                return loaded[loaded.files[0]]
            return loaded
    return None


def _as_numpy_mask(mask):
    if np is None:
        raise SystemExit("NumPy is required for mask-only labels. Install it (e.g. pip install numpy).")
    if isinstance(mask, np.ndarray):
        return mask
    return np.asarray(mask)


def _parse_color_key(key):
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


def _bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    return x_min, y_min, x_max, y_max


def _connected_components(mask_bool):
    h, w = mask_bool.shape
    visited = np.zeros((h, w), dtype=bool)
    bboxes = []
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


def _mask_to_labels(mask, *, class_map=None, instances=False):
    mask_np = _as_numpy_mask(mask)
    if mask_np.ndim == 3 and mask_np.shape[-1] in (3, 4):
        rgb = mask_np[..., :3]
        flat = rgb.reshape(-1, 3)
        unique = np.unique(flat, axis=0)
        colors = [tuple(c.tolist()) for c in unique]
        colors = [c for c in colors if c != (0, 0, 0)]
        mapping = {}
        if class_map:
            mapping = {tuple(_parse_color_key(k)): int(v) for k, v in class_map.items()}
        else:
            for idx, color in enumerate(sorted(colors)):
                mapping[color] = idx
        labels = []
        h, w = mask_np.shape[:2]
        for color, class_id in mapping.items():
            if not isinstance(color, tuple):
                continue
            mask_bool = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
            if instances:
                for bbox in _connected_components(mask_bool):
                    x_min, y_min, x_max, y_max = bbox
                    cx = (x_min + x_max + 1) / 2.0 / w
                    cy = (y_min + y_max + 1) / 2.0 / h
                    bw = (x_max - x_min + 1) / w
                    bh = (y_max - y_min + 1) / h
                    labels.append({"class_id": int(class_id), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
            else:
                bbox = _bbox_from_mask(mask_bool)
                if bbox is None:
                    continue
                x_min, y_min, x_max, y_max = bbox
                cx = (x_min + x_max + 1) / 2.0 / w
                cy = (y_min + y_max + 1) / 2.0 / h
                bw = (x_max - x_min + 1) / w
                bh = (y_max - y_min + 1) / h
                labels.append({"class_id": int(class_id), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
        return labels

    mask_np = mask_np.astype(int)
    unique = np.unique(mask_np)
    classes = [int(v) for v in unique if int(v) != 0]
    labels = []
    h, w = mask_np.shape[:2]
    class_map = class_map or {}
    for class_val in classes:
        class_id = int(class_map.get(str(class_val), class_map.get(class_val, class_val)))
        mask_bool = mask_np == class_val
        if instances:
            for bbox in _connected_components(mask_bool):
                x_min, y_min, x_max, y_max = bbox
                cx = (x_min + x_max + 1) / 2.0 / w
                cy = (y_min + y_max + 1) / 2.0 / h
                bw = (x_max - x_min + 1) / w
                bh = (y_max - y_min + 1) / h
                labels.append({"class_id": int(class_id), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
        else:
            bbox = _bbox_from_mask(mask_bool)
            if bbox is None:
                continue
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max + 1) / 2.0 / w
            cy = (y_min + y_max + 1) / 2.0 / h
            bw = (x_max - x_min + 1) / w
            bh = (y_max - y_min + 1) / h
            labels.append({"class_id": int(class_id), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
    return labels


def _load_yolo_dataset_dirs(images_dir, labels_dir, *, metadata_root, label_format=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    metadata_root = Path(metadata_root)
    records = []
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"):
        images.extend(images_dir.glob(ext))
    for image_path in sorted(images):
        label_path = labels_dir / f"{image_path.stem}.txt"
        meta_path = labels_dir / f"{image_path.stem}.json"
        labels = _load_yolo_labels(label_path, label_format=label_format)
        metadata = _load_metadata(meta_path, metadata_root)

        if not labels and metadata.get("mask_path") is not None:
            mask_value = metadata.get("mask") or metadata.get("mask_path")
            mask_instances = bool(metadata.get("mask_instances", False))
            mask_class_map = metadata.get("mask_class_map") or {}
            mask_classes = metadata.get("mask_classes")

            if isinstance(mask_value, (list, tuple)) and mask_value and isinstance(mask_value[0], (str, Path)):
                labels = []
                class_ids = mask_classes or list(range(len(mask_value)))
                for idx, mask_path in enumerate(mask_value):
                    mask_data = _load_mask_value(mask_path)
                    if mask_data is None:
                        continue
                    class_id = class_ids[idx] if idx < len(class_ids) else idx
                    labels.extend(
                        _mask_to_labels(
                            mask_data,
                            class_map={1: int(class_id), "1": int(class_id)},
                            instances=mask_instances,
                        )
                    )
            else:
                mask_data = _load_mask_value(mask_value)
                if mask_data is not None:
                    labels = _mask_to_labels(
                        mask_data,
                        class_map=mask_class_map,
                        instances=mask_instances,
                    )

        k_raw = metadata.get("K_gt")
        if k_raw is None:
            k_raw = metadata.get("intrinsics")
        k_3x3 = _k_to_3x3(k_raw)

        k_prime_raw = metadata.get("K_gt_prime")
        if k_prime_raw is None:
            k_prime_raw = metadata.get("intrinsics_prime")
        k_prime_3x3 = _k_to_3x3(k_prime_raw)

        records.append(
            normalize_record_bboxes(
                {
                "image_path": str(image_path),
                "labels": labels,
                "image_hw": metadata.get("image_hw"),
                "image_size": metadata.get("image_size"),
                "mask_path": metadata.get("mask_path"),
                "mask": metadata.get("mask"),
                "depth_path": metadata.get("depth_path"),
                "depth": metadata.get("depth"),
                "pose": metadata.get("pose"),
                "R_gt": metadata.get("R_gt"),
                "t_gt": metadata.get("t_gt"),
                "intrinsics": metadata.get("intrinsics"),
                "K_gt": k_3x3 if k_3x3 is not None else metadata.get("K_gt"),
                "intrinsics_prime": metadata.get("intrinsics_prime"),
                "K_gt_prime": k_prime_3x3 if k_prime_3x3 is not None else metadata.get("K_gt_prime"),
                "cad_points": metadata.get("cad_points"),
                "M": metadata.get("mask"),
                "D_obj": metadata.get("depth"),
                },
                bbox_field="cxcywh_norm",
            )
        )
    return records


def _resolve_relative_path(value, *, base):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path


def _resolve_dataset_json_layout(root, split):
    root = Path(root)
    descriptor = root if root.is_file() and root.suffix.lower() == ".json" else root / "dataset.json"
    if not descriptor.exists():
        return None
    base = descriptor.parent
    try:
        data = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    images_dir = _resolve_relative_path(data.get("images_dir"), base=base)
    labels_dir = _resolve_relative_path(data.get("labels_dir"), base=base)
    if images_dir is None or labels_dir is None:
        return None

    effective_split = str(split or data.get("split") or labels_dir.name)
    if split:
        images_candidate = images_dir.parent / str(split)
        labels_candidate = labels_dir.parent / str(split)
        if images_candidate.exists():
            images_dir = images_candidate
        if labels_candidate.exists():
            labels_dir = labels_candidate
    label_format = data.get("label_format") or data.get("task")
    return images_dir, labels_dir, effective_split, base, (str(label_format) if label_format else None)


def _load_yaml_mapping(path):
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise SystemExit("PyYAML is required to read data.yaml dataset descriptors.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _resolve_data_yaml_layout(root, split):
    root = Path(root)
    yaml_path = root if root.is_file() and root.suffix.lower() in (".yaml", ".yml") else root / "data.yaml"
    if not yaml_path.exists():
        return None
    data = _load_yaml_mapping(yaml_path)
    base = _resolve_relative_path(data.get("path"), base=yaml_path.parent) or yaml_path.parent

    split_key = "train"
    split_text = str(split or "").strip().lower()
    if split_text in ("val", "valid", "validation") or split_text.startswith("val"):
        split_key = "val"
    elif split_text in ("test", "testing") or split_text.startswith("test"):
        split_key = "test"
    elif split_text in data:
        split_key = split_text

    images_dir = _resolve_relative_path(data.get(split_key), base=base)
    if images_dir is None:
        return None
    labels_dir = images_dir
    if images_dir.name:
        if images_dir.parent.name == "images":
            labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        elif (images_dir / "images").exists():
            labels_dir = images_dir / "labels"
            images_dir = images_dir / "images"
        else:
            labels_dir = images_dir.parent / "labels" / images_dir.name
    label_format = data.get("label_format") or data.get("task")
    return images_dir, labels_dir, str(split or split_key), base, (str(label_format) if label_format else None)


def load_yolo_dataset(root, split="train2017"):
    root = Path(root)
    resolved = _resolve_data_yaml_layout(root, split)
    if resolved is None:
        resolved = _resolve_dataset_json_layout(root, split)
    if resolved is not None:
        images_dir, labels_dir, _effective_split, metadata_root, *rest = resolved
        label_format = rest[0] if rest else None
        return _load_yolo_dataset_dirs(images_dir, labels_dir, metadata_root=metadata_root, label_format=label_format)

    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    return _load_yolo_dataset_dirs(images_dir, labels_dir, metadata_root=root)


def _availability_stats(records):
    stats = {
        "total": len(records),
        "mask": 0,
        "depth": 0,
        "pose": 0,
        "intrinsics": 0,
        "cad_points": 0,
    }
    for rec in records:
        if rec.get("mask_path") is not None:
            stats["mask"] += 1
        if rec.get("depth_path") is not None:
            stats["depth"] += 1
        if rec.get("pose") is not None or rec.get("R_gt") is not None or rec.get("t_gt") is not None:
            stats["pose"] += 1
        if rec.get("intrinsics") is not None or rec.get("K_gt") is not None:
            stats["intrinsics"] += 1
        if rec.get("cad_points") is not None:
            stats["cad_points"] += 1
    return stats


def _normalize_keypoint_names(value):
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        text = str(item).strip()
        if text:
            names.append(text)
    return names


def _normalize_skeleton(value, *, num_keypoints=0):
    if not isinstance(value, list):
        return []
    edges = []
    seen = set()
    for edge in value:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            a = int(edge[0])
            b = int(edge[1])
        except Exception:
            continue
        if a <= 0 or b <= 0 or a == b:
            continue
        if int(num_keypoints) > 0 and (a > int(num_keypoints) or b > int(num_keypoints)):
            continue
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append([a, b])
    return edges


def _extract_keypoints_meta_from_categories(categories):
    if not isinstance(categories, list):
        return {}
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        keypoint_names = _normalize_keypoint_names(cat.get("keypoints") or [])
        if not keypoint_names:
            continue
        skeleton = _normalize_skeleton(cat.get("skeleton") or [], num_keypoints=len(keypoint_names))
        out = {
            "keypoint_names": keypoint_names,
            "num_keypoints": int(len(keypoint_names)),
        }
        if skeleton:
            out["skeleton"] = skeleton
        try:
            category_id = int(cat.get("id"))
        except (TypeError, ValueError):
            category_id = None
        if category_id is not None:
            out["keypoint_category_id"] = category_id
        return out
    return {}


def _load_keypoints_meta(root, split):
    root = Path(root)
    split = str(split)

    keypoint_names = []
    skeleton = []
    source = None

    classes_path = root / "labels" / split / "classes.json"
    if classes_path.exists():
        try:
            classes_doc = json.loads(classes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            classes_doc = {}
        if isinstance(classes_doc, dict):
            keypoint_names = _normalize_keypoint_names(classes_doc.get("keypoint_names") or [])
            if keypoint_names:
                skeleton = _normalize_skeleton(classes_doc.get("skeleton") or [], num_keypoints=len(keypoint_names))
                source = "classes_json"

    dataset_json_path = root / "dataset.json"
    if dataset_json_path.exists() and (not keypoint_names):
        try:
            dataset_doc = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset_doc = {}
        if isinstance(dataset_doc, dict):
            direct_names = _normalize_keypoint_names(dataset_doc.get("keypoint_names") or [])
            if direct_names:
                keypoint_names = direct_names
                skeleton = _normalize_skeleton(dataset_doc.get("skeleton") or [], num_keypoints=len(keypoint_names))
                source = "dataset_json"
            elif str(dataset_doc.get("format") or "").strip().lower() in ("coco_instances", "coco-instances", "coco"):
                instances_path = dataset_doc.get("instances_json")
                if isinstance(instances_path, str) and instances_path.strip():
                    instances_file = Path(instances_path)
                    if not instances_file.is_absolute():
                        instances_file = (dataset_json_path.parent / instances_file).resolve()
                    if instances_file.exists():
                        try:
                            coco_doc = json.loads(instances_file.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            coco_doc = {}
                        if isinstance(coco_doc, dict):
                            extracted = _extract_keypoints_meta_from_categories(coco_doc.get("categories") or [])
                            if extracted:
                                keypoint_names = list(extracted.get("keypoint_names") or [])
                                skeleton = list(extracted.get("skeleton") or [])
                                source = "coco_categories"

    if not keypoint_names:
        return {}

    meta = {
        "keypoint_names": keypoint_names,
        "num_keypoints": int(len(keypoint_names)),
    }
    if skeleton:
        meta["skeleton"] = skeleton
    if source:
        meta["source"] = source
    return meta


def build_manifest(root, split="train2017"):
    root = Path(root)
    resolved_split = split
    resolved = _resolve_data_yaml_layout(root, split)
    if resolved is None:
        resolved = _resolve_dataset_json_layout(root, split)
    if resolved is not None:
        _images_dir, _labels_dir, resolved_split, _metadata_root, *_rest = resolved

    images = load_yolo_dataset(root, split=split)
    manifest = {"images": images, "stats": _availability_stats(images)}
    keypoints_meta = _load_keypoints_meta(root, resolved_split)
    if keypoints_meta:
        manifest["keypoints_meta"] = keypoints_meta
    return manifest


def _k_to_3x3(k):
    if k is None:
        return None
    if hasattr(k, "tolist"):
        try:
            k = k.tolist()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    # K may be a 3x3 nested list, or (fx, fy, cx, cy)
    if isinstance(k, (list, tuple)) and len(k) == 4 and not isinstance(k[0], (list, tuple)):
        fx, fy, cx, cy = [float(v) for v in k]
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    # Flat row-major 3x3.
    if isinstance(k, (list, tuple)) and len(k) == 9 and not isinstance(k[0], (list, tuple)):
        flat = [float(v) for v in k]
        return [
            [flat[0], flat[1], flat[2]],
            [flat[3], flat[4], flat[5]],
            [flat[6], flat[7], flat[8]],
        ]
    if isinstance(k, (list, tuple)) and len(k) == 3 and isinstance(k[0], (list, tuple)):
        return [[float(v) for v in row] for row in k]
    if isinstance(k, dict) and {"fx", "fy", "cx", "cy"}.issubset(k.keys()):
        fx = float(k["fx"])
        fy = float(k["fy"])
        cx = float(k["cx"])
        cy = float(k["cy"])
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    # OpenCV FileStorage / YAML style: {rows:3, cols:3, dt:..., data:[...]}
    if isinstance(k, dict):
        try:
            rows = int(k.get("rows"))
            cols = int(k.get("cols"))
        except Exception:
            rows = None
            cols = None
        data = k.get("data")
        if rows == 3 and cols == 3 and isinstance(data, (list, tuple)) and len(data) >= 9:
            flat = [float(data[i]) for i in range(9)]
            return [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ]
        for key in ("camera_matrix", "cameraMatrix", "K", "K_gt", "intrinsics"):
            if key in k:
                nested = _k_to_3x3(k.get(key))
                if nested is not None:
                    return nested
    return None


def _expand_to_instances(value, num_instances, *, default=None):
    if value is None:
        return [default for _ in range(int(num_instances))]
    if isinstance(value, (list, tuple)) and len(value) == int(num_instances):
        return list(value)
    return [value for _ in range(int(num_instances))]


def extract_pose_intrinsics_targets(record, num_instances: int):
    """Extract and normalize optional pose/intrinsics targets.

    Returns a dict of python lists suitable for conversion to torch tensors:
    - K_gt: 3x3 nested list or None
    - t_gt: list length num_instances, each item is [x,y,z] or None
    - R_gt: list length num_instances, each item is 3x3 nested list or None
    - offsets_gt: list length num_instances, each item is [du,dv] or None
    """

    pose = record.get("pose") or {}
    r_gt = record.get("R_gt")
    t_gt = record.get("t_gt")
    if r_gt is None and isinstance(pose, dict):
        r_gt = pose.get("R")
    if t_gt is None and isinstance(pose, dict):
        t_gt = pose.get("t")

    k_raw = record.get("K_gt")
    if k_raw is None:
        k_raw = record.get("intrinsics")
    k_gt = _k_to_3x3(k_raw)

    # R and t may be per-image or per-instance.
    r_list = _expand_to_instances(r_gt, num_instances, default=None)
    t_list = _expand_to_instances(t_gt, num_instances, default=None)

    def _norm_R(r):
        if r is None:
            return None
        if isinstance(r, (list, tuple)) and len(r) == 3 and isinstance(r[0], (list, tuple)):
            return [[float(v) for v in row] for row in r]
        return None

    def _norm_t(t):
        if t is None:
            return None
        if isinstance(t, (list, tuple)) and len(t) == 3 and not isinstance(t[0], (list, tuple)):
            return [float(v) for v in t]
        return None

    r_list = [_norm_R(r) for r in r_list]
    t_list = [_norm_t(t) for t in t_list]

    offsets_raw = record.get("offsets_gt")
    if offsets_raw is None:
        offsets_raw = record.get("offsets")
    off_list = _expand_to_instances(offsets_raw, num_instances, default=None)

    def _norm_off(off):
        if off is None:
            return None
        if isinstance(off, (list, tuple)) and len(off) == 2:
            return [float(off[0]), float(off[1])]
        return None

    off_list = [_norm_off(o) for o in off_list]

    return {
        "K_gt": k_gt,
        "t_gt": t_list,
        "R_gt": r_list,
        "offsets_gt": off_list,
    }


def extract_full_gt_targets(record, num_instances: int):
    """Extract and normalize full GT fields per spec.

    Returns canonical python lists suitable for conversion to torch tensors
    (where applicable) without eager decoding:

    - M: list length num_instances, each item is a 2D array OR a path string OR None
    - D_obj: list length num_instances, each item is a 2D array OR a path string OR None
    - R_gt/t_gt/K_gt/offsets_gt: see extract_pose_intrinsics_targets
    - *_mask: list[bool] indicating availability per instance
    """

    base = extract_pose_intrinsics_targets(record, num_instances=num_instances)

    m_raw = record.get("M")
    if m_raw is None:
        m_raw = record.get("mask")
    if m_raw is None:
        m_raw = record.get("mask_path")

    d_raw = record.get("D_obj")
    if d_raw is None:
        d_raw = record.get("depth")
    if d_raw is None:
        d_raw = record.get("depth_path")

    m_list = _expand_to_instances(m_raw, num_instances, default=None)
    d_list = _expand_to_instances(d_raw, num_instances, default=None)

    def _norm_m(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return value
        return None

    def _norm_d(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return value
        return None

    m_list = [_norm_m(v) for v in m_list]
    d_list = [_norm_d(v) for v in d_list]
    m_mask = [v is not None for v in m_list]
    d_mask = [v is not None for v in d_list]

    return {
        **base,
        "M": m_list,
        "D_obj": d_list,
        "M_mask": m_mask,
        "D_obj_mask": d_mask,
    }


def depth_at_bbox_center(d_obj, bbox, mask=None):
    """Sample depth (z) at bbox center from an object-only depth map.

    d_obj: 2D array-like (nested lists/tuples or ndarray)
    bbox: dict with cx/cy or sequence [cx,cy,w,h] in normalized coords
    mask: optional 2D mask (same spatial size); if provided and center is masked out,
          returns None.
    """

    if d_obj is None:
        return None

    # Extract bbox center in normalized coords.
    if isinstance(bbox, dict):
        cx = float(bbox.get("cx", 0.0))
        cy = float(bbox.get("cy", 0.0))
    else:
        cx = float(bbox[0])
        cy = float(bbox[1])

    # Determine H,W.
    if hasattr(d_obj, "shape"):
        h = int(d_obj.shape[0])
        w = int(d_obj.shape[1])
    else:
        if not isinstance(d_obj, (list, tuple)) or not d_obj:
            return None
        h = len(d_obj)
        w = len(d_obj[0]) if isinstance(d_obj[0], (list, tuple)) else 0
    if h <= 0 or w <= 0:
        return None

    u = int(round(cx * (w - 1)))
    v = int(round(cy * (h - 1)))
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))

    if mask is not None:
        try:
            m_val = mask[v][u] if not hasattr(mask, "shape") else mask[v, u]
        except Exception:
            m_val = None
        if m_val is not None and float(m_val) <= 0.0:
            return None

    try:
        z = d_obj[v][u] if not hasattr(d_obj, "shape") else d_obj[v, u]
    except Exception:
        return None

    try:
        zf = float(z)
    except Exception:
        return None
    if not (zf >= 0.0):
        return None
    return zf
