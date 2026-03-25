import json
import math
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _shape(value):
    if hasattr(value, "shape"):
        try:
            dims = tuple(int(dim) for dim in value.shape)
        except TypeError:
            return None
        if len(dims) >= 2:
            return (dims[0], dims[1])
        if len(dims) == 1:
            return (dims[0], 0)
        return None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value), 0)
    return None


def _is_matrix(value, size):
    shape = _shape(value)
    if shape is None:
        return False
    return shape[0] == size and shape[1] == size


def _is_vector(value, size):
    shape = _shape(value)
    if shape is None:
        return False
    return shape[0] == size and shape[1] == 0


def _validate_optional_path_or_array(value, name):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise ValueError(f"{name} missing: {path}")
        return ("path", None)
    shape = _shape(value)
    if shape is None or shape[1] == 0:
        raise ValueError(f"{name} must be a path or 2D array")
    return ("array", shape)


def _is_2d_list_array(value):
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    if not isinstance(first, (list, tuple)) or not first:
        return False
    # 2D numeric array-like: value[0][0] is scalar.
    inner = first[0]
    return not isinstance(inner, (list, tuple, dict))


def _split_per_instance(value, num_instances, *, name):
    """Return a list length num_instances for broadcast/per-instance values.

    Supports:
    - None -> [None]*N
    - path/array -> broadcast
    - list length N -> per-instance
    - 2D inline array (list-of-list numeric) -> broadcast
    """

    n = int(num_instances)
    if value is None:
        return [None for _ in range(n)]

    # Inline 2D array is treated as broadcast.
    if _is_2d_list_array(value) or hasattr(value, "shape"):
        return [value for _ in range(n)]

    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have length {n} for multi-object samples")
        return list(value)

    # Scalar/broadcast.
    return [value for _ in range(n)]


def _validate_pose(value):
    if value is None:
        return
    if isinstance(value, dict):
        r = value.get("R") or value.get("R_gt")
        t = value.get("t") or value.get("t_gt")
        if r is None or t is None:
            raise ValueError("pose must include R and t")
        if not _is_matrix(r, 3):
            raise ValueError("pose R must be 3x3")
        if not _is_vector(t, 3):
            raise ValueError("pose t must be length 3")
        return
    if _is_matrix(value, 4):
        return
    raise ValueError("pose must be a dict with R/t or a 4x4 matrix")


def _validate_pose_multi(value, num_instances):
    if value is None:
        return
    # Per-instance pose list.
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (dict, list, tuple)):
        # Avoid treating 4x4 matrix (list of 4 lists) as per-instance.
        if not _is_matrix(value, 4):
            if len(value) != int(num_instances):
                raise ValueError(f"pose must have length {int(num_instances)} for multi-object samples")
            for item in value:
                _validate_pose(item)
            return
    _validate_pose(value)


def _validate_intrinsics(value):
    if value is None:
        return
    if isinstance(value, dict):
        if all(k in value for k in ("fx", "fy", "cx", "cy")):
            return
        if "K" in value and _is_matrix(value["K"], 3):
            return
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            return
    if _is_matrix(value, 3):
        return
    raise ValueError("intrinsics must be fx/fy/cx/cy or 3x3 matrix")


def _validate_intrinsics_multi(value, num_instances):
    if value is None:
        return
    if isinstance(value, (list, tuple)) and value and not _is_matrix(value, 3):
        # If this is per-instance list, validate each.
        if len(value) == int(num_instances) and not (len(value) == 4 and all(isinstance(v, (int, float)) for v in value)):
            for item in value:
                _validate_intrinsics(item)
            return
    _validate_intrinsics(value)


def _load_array_from_path(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    if suffix == ".png":
        try:
            from PIL import Image  # type: ignore
        except Exception:  # pragma: no cover
            return None
        try:
            img = Image.open(path)
        except Exception:
            return None
        img = img.convert("L")
        if np is not None:
            return np.asarray(img)
        width, height = img.size
        data = list(img.getdata())
        return [data[i * width : (i + 1) * width] for i in range(height)]  # pragma: no cover
    if suffix in (".npy", ".npz") and np is not None:
        try:
            loaded = np.load(path)
        except Exception:
            return None
        if hasattr(loaded, "files"):
            if not loaded.files:
                return None
            return loaded[loaded.files[0]]
        return loaded
    return None


def _load_array(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return _load_array_from_path(Path(value))
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (str, Path)):
        # Per-instance list of paths.
        return [_load_array_from_path(Path(v)) if v is not None else None for v in value]
    return value


def _infer_image_shape(mask, depth):
    for value in (mask, depth):
        shape = _shape(value)
        if shape and shape[1] != 0:
            return (shape[0], shape[1])
    return None


def _mask_bbox(mask):
    if mask is None:
        return None
    if np is not None and hasattr(mask, "shape"):
        arr = np.asarray(mask)
        if arr.ndim > 2:
            arr = arr[..., 0]
        ys, xs = np.where(arr > 0)
        if ys.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    x0 = y0 = None
    x1 = y1 = None
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value > 0:
                if x0 is None:
                    x0 = x1 = x
                    y0 = y1 = y
                else:
                    x0 = min(x0, x)
                    y0 = min(y0, y)
                    x1 = max(x1, x)
                    y1 = max(y1, y)
    if x0 is None:
        return None
    return (x0, y0, x1, y1)


def _bbox_to_pixels(bbox, image_shape):
    height, width = image_shape
    cx = bbox["cx"] * width
    cy = bbox["cy"] * height
    bw = bbox["w"] * width
    bh = bbox["h"] * height
    x0 = max(0.0, cx - bw * 0.5)
    x1 = min(float(width), cx + bw * 0.5)
    y0 = max(0.0, cy - bh * 0.5)
    y1 = min(float(height), cy + bh * 0.5)
    return (x0, y0, x1, y1)


def _bboxes_intersect(a, b):
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return ix1 >= ix0 and iy1 >= iy0


def _extract_intrinsics(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("fx", "fy", "cx", "cy")):
            return (value["fx"], value["fy"], value["cx"], value["cy"])
        if "K" in value and _is_matrix(value["K"], 3):
            k = value["K"]
            return (k[0][0], k[1][1], k[0][2], k[1][2])
    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            return (value[0], value[1], value[2], value[3])
    if _is_matrix(value, 3):
        return (value[0][0], value[1][1], value[0][2], value[1][2])
    return None


def _extract_pose(value):
    if value is None:
        return None
    if isinstance(value, dict):
        r = value.get("R") or value.get("R_gt")
        t = value.get("t") or value.get("t_gt")
        if r is None or t is None:
            return None
        return r, t
    if _is_matrix(value, 4):
        r = [row[:3] for row in value[:3]]
        t = [value[0][3], value[1][3], value[2][3]]
        return r, t
    return None


def _project_points(points, pose, intrinsics):
    if points is None or pose is None or intrinsics is None:
        return []
    r, t = pose
    fx, fy, cx, cy = intrinsics
    projected = []
    for point in points:
        if len(point) < 3:
            continue
        x, y, z = point[0], point[1], point[2]
        xc = r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0]
        yc = r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1]
        zc = r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2]
        if zc <= 0:
            continue
        u = fx * xc / zc + cx
        v = fy * yc / zc + cy
        projected.append((u, v))
    return projected


def _validate_content(mask, depth, labels, pose, intrinsics, cad_points):
    mask_bbox = _mask_bbox(mask)
    if mask is not None and mask_bbox is None:
        raise ValueError("mask has no foreground pixels")
    if mask is not None and depth is not None:
        shape_mask = _shape(mask)
        shape_depth = _shape(depth)
        if shape_mask and shape_depth and shape_mask[:2] != shape_depth[:2]:
            raise ValueError("mask/depth shapes must match")
        if np is not None and hasattr(mask, "shape") and hasattr(depth, "shape"):
            mask_arr = np.asarray(mask)
            depth_arr = np.asarray(depth)
            if mask_arr.ndim > 2:
                mask_arr = mask_arr[..., 0]
            if depth_arr.ndim > 2:
                depth_arr = depth_arr[..., 0]
            if (depth_arr > 0).any() and (mask_arr > 0).sum() == 0:
                raise ValueError("depth has values but mask is empty")
            if ((depth_arr > 0) & (mask_arr <= 0)).any():
                raise ValueError("depth must be zero outside mask")
        else:
            for row_mask, row_depth in zip(mask, depth):
                for mask_val, depth_val in zip(row_mask, row_depth):
                    if depth_val is None:
                        continue
                    if depth_val > 0 and mask_val <= 0:
                        raise ValueError("depth must be zero outside mask")

    image_shape = _infer_image_shape(mask, depth)
    if image_shape and mask_bbox and labels:
        overlaps = False
        for label in labels:
            label_bbox = _bbox_to_pixels(label["bbox"], image_shape)
            if _bboxes_intersect(label_bbox, mask_bbox):
                overlaps = True
                break
        if not overlaps:
            raise ValueError("mask bbox does not overlap any label bbox")

    if cad_points and image_shape and labels:
        if len(labels) != 1:
            return
        pose_rt = _extract_pose(pose)
        k = _extract_intrinsics(intrinsics)
        if pose_rt is None or k is None:
            return
        projected = _project_points(cad_points, pose_rt, k)
        if not projected:
            raise ValueError("cad projection produced no visible points")
        label_bbox = _bbox_to_pixels(labels[0]["bbox"], image_shape)
        inside = False
        for u, v in projected:
            if label_bbox[0] <= u <= label_bbox[2] and label_bbox[1] <= v <= label_bbox[3]:
                inside = True
                break
        if not inside:
            raise ValueError("cad projection does not intersect label bbox")


def _validate_mask_binary(mask):
    if mask is None:
        return

    # Numpy path (fast).
    if np is not None and hasattr(mask, "shape"):
        arr = np.asarray(mask)
        if arr.ndim > 2:
            arr = arr[..., 0]

        if arr.dtype == np.bool_:
            return

        if np.issubdtype(arr.dtype, np.integer):
            ok01 = np.isin(arr, (0, 1)).all()
            ok0255 = np.isin(arr, (0, 255)).all()
            if not (ok01 or ok0255):
                raise ValueError("mask must be binary (0/1 or 0/255)")
            return

        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                raise ValueError("mask must be finite")
            if (arr < 0).any() or (arr > 1).any():
                raise ValueError("mask float values must be in [0,1]")
            if (np.abs(arr - np.round(arr)) > 1e-3).any():
                raise ValueError("mask must be binary (near 0/1)")
            return

    # Pure-Python list/tuple path.
    shape = _shape(mask)
    if shape is None or shape[1] == 0:
        return

    seen_nonfinite = False
    seen_float = False
    allowed_int_values = {0, 1, 255}

    for row in mask:
        for value in row:
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                if value not in allowed_int_values:
                    raise ValueError("mask must be binary (0/1 or 0/255)")
                continue
            if isinstance(value, float):
                seen_float = True
                if not math.isfinite(value):
                    seen_nonfinite = True
                    continue
                if value < 0.0 or value > 1.0:
                    raise ValueError("mask float values must be in [0,1]")
                if abs(value - round(value)) > 1e-3:
                    raise ValueError("mask must be binary (near 0/1)")
                continue

    if seen_nonfinite:
        raise ValueError("mask must be finite")
    if seen_float:
        return


def _validate_depth_range(depth):
    if depth is None:
        return

    # Numpy path.
    if np is not None and hasattr(depth, "shape"):
        arr = np.asarray(depth)
        if arr.ndim > 2:
            arr = arr[..., 0]
        if not np.isfinite(arr).all():
            raise ValueError("depth must be finite")
        if (arr < 0).any():
            raise ValueError("depth must be non-negative")
        return

    # Pure-Python list/tuple path.
    shape = _shape(depth)
    if shape is None or shape[1] == 0:
        return
    seen_nonfinite = False
    for row in depth:
        for value in row:
            if value is None:
                continue
            if isinstance(value, (int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    seen_nonfinite = True
                    continue
                if value < 0:
                    raise ValueError("depth must be non-negative")
    if seen_nonfinite:
        raise ValueError("depth must be finite")


REQUIRED_KEYS = {
    "image_path",
    "labels",
    "mask_path",
    "depth_path",
    "pose",
    "intrinsics",
}


def validate_sample(sample, strict=False, check_content=False, check_ranges=False):
    missing = REQUIRED_KEYS - set(sample.keys())
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise ValueError(f"image missing: {image_path}")
    if not isinstance(sample["labels"], list):
        raise ValueError("labels must be list")
    num_instances = len(sample["labels"])
    for label in sample["labels"]:
        if "class_id" not in label or "bbox" not in label:
            raise ValueError("label missing class_id/bbox")
        bbox = label["bbox"]
        for key in ("cx", "cy", "w", "h"):
            value = bbox.get(key)
            if value is None or not (0.0 <= value <= 1.0):
                raise ValueError(f"bbox {key} out of range: {value}")
    # Multi-object handling: allow per-instance mask/depth/pose lists, otherwise broadcast.
    mask_items = _split_per_instance(sample.get("mask_path"), num_instances, name="mask_path")
    depth_items = _split_per_instance(sample.get("depth_path"), num_instances, name="depth_path")
    pose_items = _split_per_instance(sample.get("pose"), num_instances, name="pose")

    for item in mask_items:
        _validate_optional_path_or_array(item, "mask_path")
    for item in depth_items:
        _validate_optional_path_or_array(item, "depth_path")

    # Per-instance pose validation (allows broadcast).
    for item in pose_items:
        _validate_pose(item)

    # Intrinsics is typically per-image, but we allow per-instance lists.
    _validate_intrinsics_multi(sample.get("intrinsics"), num_instances)
    _validate_intrinsics_multi(sample.get("intrinsics_prime"), num_instances)

    if check_content or check_ranges:
        mask_loaded = _load_array(mask_items)
        depth_loaded = _load_array(depth_items)
        cad_points = _load_array(sample.get("cad_points"))

        # Validate per-instance when possible.
        for i in range(num_instances):
            m_i = mask_loaded[i] if isinstance(mask_loaded, list) else None
            d_i = depth_loaded[i] if isinstance(depth_loaded, list) else None
            if check_content:
                _validate_content(
                    m_i,
                    d_i,
                    [sample["labels"][i]] if sample.get("labels") else [],
                    pose_items[i],
                    sample.get("intrinsics"),
                    cad_points,
                )
            if check_ranges:
                _validate_mask_binary(m_i)
                _validate_depth_range(d_i)
    if strict:
        if any(item is None for item in mask_items):
            raise ValueError("mask_path required in strict mode")
        if any(item is None for item in depth_items):
            raise ValueError("depth_path required in strict mode")
        if any(item is None for item in pose_items):
            raise ValueError("pose required in strict mode")
        if sample["intrinsics"] is None:
            raise ValueError("intrinsics required in strict mode")


def validate_manifest(manifest, strict=False, check_content=False, check_ranges=False):
    images = manifest.get("images", [])
    if not images:
        raise ValueError("manifest has no images")
    for sample in images:
        validate_sample(
            sample,
            strict=strict,
            check_content=check_content,
            check_ranges=check_ranges,
        )
