"""Dataset export helpers for YOLOZU wrappers.

Exports canonical YOLOZU dataset records into external training layouts.
Current targets focus on bbox-detection datasets:

- YOLO-style layout with ``data.yaml`` + ``images/<split>`` + ``labels/<split>``
- KITTI-style layout with ``image_2`` + ``label_2`` + ``ImageSets/Main/<split>.txt``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from yolozu.core.image_size import get_image_size
from yolozu.datasets.dataset_contract import normalize_label_bbox

__all__ = [
    "export_coco_dataset",
    "export_yolo_dataset",
    "export_kitti_dataset",
    "export_segmentation_dataset",
]


def _resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root


def _resolve_source_split(dataset_root: Path, split: str | None) -> tuple[Path, str]:
    from .dataset import _pick_split

    if dataset_root.is_file():
        return dataset_root.parent, str(split or "")
    split_effective = str(split or _pick_split(dataset_root, split))
    return dataset_root, split_effective


def _load_classes_payload(*, dataset_root: Path, split: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_payload: dict[str, Any] = {}
    descriptor = dataset_root / "dataset.json"
    if descriptor.exists():
        try:
            raw_descriptor = json.loads(descriptor.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_descriptor = None
        if isinstance(raw_descriptor, dict):
            dataset_payload = raw_descriptor

    candidates = [
        dataset_root / "labels" / split / "classes.json",
        dataset_root / "labels" / split / "classes.txt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".json":
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                for key in ("keypoint_names", "num_keypoints", "skeleton", "task"):
                    if key in dataset_payload and key not in payload:
                        payload[key] = dataset_payload.get(key)
                return payload
        else:
            try:
                names = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                continue
            if names:
                payload: dict[str, Any] = {"class_names": names}
                for key in ("keypoint_names", "num_keypoints", "skeleton", "task"):
                    if key in dataset_payload:
                        payload[key] = dataset_payload.get(key)
                return payload

    max_class_id = -1
    for record in records:
        for label in record.get("labels") or []:
            try:
                max_class_id = max(max_class_id, int(label.get("class_id")))
            except (TypeError, ValueError):
                continue

    class_names = [f"class_{idx}" for idx in range(max_class_id + 1)] if max_class_id >= 0 else []
    payload = {
        "class_names": class_names,
        "class_id_to_category_id": {str(idx): idx for idx in range(len(class_names))},
        "category_id_to_class_id": {str(idx): idx for idx in range(len(class_names))},
    }
    for key in ("keypoint_names", "num_keypoints", "skeleton", "task"):
        if key in dataset_payload:
            payload[key] = dataset_payload.get(key)
    return payload


def _class_names_from_payload(payload: dict[str, Any]) -> list[str]:
    for key in ("class_names", "names"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if isinstance(raw, dict):
            parsed: list[tuple[int, str]] = []
            for item_key, item_value in raw.items():
                try:
                    parsed.append((int(item_key), str(item_value)))
                except (TypeError, ValueError):
                    continue
            parsed.sort(key=lambda item: item[0])
            if parsed:
                upper = parsed[-1][0]
                names = [f"class_{idx}" for idx in range(upper + 1)]
                for idx, value in parsed:
                    names[idx] = value
                return names
    return []


def _write_classes_files(labels_dir: Path, payload: dict[str, Any]) -> list[str]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    names = _class_names_from_payload(payload)
    if names and "class_names" not in payload:
        payload = dict(payload)
        payload["class_names"] = names
    (labels_dir / "classes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (labels_dir / "classes.txt").write_text(
        ("\n".join(names) + "\n") if names else "",
        encoding="utf-8",
    )
    return names


def _prepare_output_root(out_dir: str | Path, *, force: bool) -> Path:
    out_root = Path(out_dir)
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()
    if out_root.exists():
        if not force:
            raise FileExistsError(f"output already exists: {out_root} (use --force to overwrite)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _materialize_file(src: Path, dst: Path, *, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return
    raise ValueError("mode must be copy|symlink")


def _format_yolo_label(label: dict[str, Any], *, image_wh: tuple[int, int] | None = None) -> str:
    if image_wh is not None:
        label = normalize_label_bbox(label, image_wh=(float(image_wh[0]), float(image_wh[1])), bbox_field="cxcywh_norm")
    class_id = int(label.get("class_id", 0))
    polygon = label.get("polygon")
    keypoints = label.get("keypoints")
    if isinstance(polygon, list) and len(polygon) >= 6:
        poly_values = " ".join(f"{float(value):.6g}" for value in polygon)
        return f"{class_id} {poly_values}"

    coords = [
        float(label.get("cx", 0.0)),
        float(label.get("cy", 0.0)),
        float(label.get("w", 0.0)),
        float(label.get("h", 0.0)),
    ]
    out = f"{class_id} " + " ".join(f"{value:.6g}" for value in coords)
    if isinstance(keypoints, list) and keypoints:
        flat_keypoints: list[float] = []
        if isinstance(keypoints[0], dict):
            for item in keypoints:
                if not isinstance(item, dict):
                    continue
                flat_keypoints.extend(
                    [
                        float(item.get("x", 0.0)),
                        float(item.get("y", 0.0)),
                        float(item.get("v", 2.0)),
                    ]
                )
        else:
            for value in keypoints:
                flat_keypoints.append(float(value))
        if flat_keypoints:
            out = out + " " + " ".join(f"{float(value):.6g}" for value in flat_keypoints)
    return out


def _record_image_size(record: dict[str, Any]) -> tuple[int, int]:
    image_hw = record.get("image_hw") or record.get("image_size") or record.get("hw")
    if isinstance(image_hw, (list, tuple)) and len(image_hw) >= 2:
        try:
            height = int(image_hw[0])
            width = int(image_hw[1])
        except (TypeError, ValueError):
            height = width = 0
        if width > 0 and height > 0:
            return width, height

    image_path = Path(str(record.get("image")))
    width, height = get_image_size(image_path)
    return int(width), int(height)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_kitti_label_line(label: dict[str, Any], *, class_names: list[str], image_w: int, image_h: int) -> str:
    class_id = int(label.get("class_id", 0))
    if 0 <= class_id < len(class_names):
        class_name = class_names[class_id]
    else:
        class_name = f"class_{class_id}"
    class_name = "_".join(str(class_name).split()) or f"class_{class_id}"

    cx = float(label.get("cx", 0.0))
    cy = float(label.get("cy", 0.0))
    bw = float(label.get("w", 0.0))
    bh = float(label.get("h", 0.0))

    x1 = _clip((cx - bw / 2.0) * float(image_w), 0.0, float(image_w))
    y1 = _clip((cy - bh / 2.0) * float(image_h), 0.0, float(image_h))
    x2 = _clip((cx + bw / 2.0) * float(image_w), 0.0, float(image_w))
    y2 = _clip((cy + bh / 2.0) * float(image_h), 0.0, float(image_h))
    return (
        f"{class_name} 0.00 0 0.00 "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        "-1.00 -1.00 -1.00 -1000.00 -1000.00 -1000.00 -10.00"
    )


def _category_mappings_from_payload(payload: dict[str, Any], *, class_names: list[str]) -> tuple[dict[int, int], dict[int, int]]:
    class_to_category_raw = payload.get("class_id_to_category_id") or payload.get("class_to_category_id") or {}
    category_to_class_raw = payload.get("category_id_to_class_id") or {}

    class_to_category: dict[int, int] = {}
    if isinstance(class_to_category_raw, dict):
        for key, value in class_to_category_raw.items():
            try:
                class_to_category[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    if not class_to_category:
        class_to_category = {idx: idx + 1 for idx in range(len(class_names))}

    category_to_class: dict[int, int] = {}
    if isinstance(category_to_class_raw, dict):
        for key, value in category_to_class_raw.items():
            try:
                category_to_class[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    if not category_to_class:
        category_to_class = {category_id: class_id for class_id, category_id in class_to_category.items()}
    return class_to_category, category_to_class


def _keypoint_schema_from_payload(payload: dict[str, Any]) -> tuple[list[str], list[list[int]]]:
    raw_names = payload.get("keypoint_names") or []
    keypoint_names: list[str] = []
    if isinstance(raw_names, list):
        for item in raw_names:
            text = str(item).strip()
            if text:
                keypoint_names.append(text)
    raw_skeleton = payload.get("skeleton") or []
    skeleton: list[list[int]] = []
    if isinstance(raw_skeleton, list):
        for edge in raw_skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            try:
                skeleton.append([int(edge[0]), int(edge[1])])
            except Exception:
                continue
    return keypoint_names, skeleton


def _keypoints_to_coco_list(
    keypoints: Any,
    *,
    image_w: int,
    image_h: int,
    expected_count: int,
) -> tuple[list[float], int]:
    out: list[float] = []
    num_labeled = 0
    normalized = keypoints if isinstance(keypoints, list) else []
    for idx in range(expected_count):
        base = idx * 3
        if base + 2 < len(normalized):
            try:
                x = float(normalized[base]) * float(image_w)
                y = float(normalized[base + 1]) * float(image_h)
                v = float(normalized[base + 2])
            except Exception:
                x = y = 0.0
                v = 0.0
        else:
            x = y = 0.0
            v = 0.0
        if v > 0.0:
            num_labeled += 1
        out.extend([float(x), float(y), int(round(v))])
    return out, int(num_labeled)


def _polygon_to_coco_segmentation(polygon: Any, *, image_w: int, image_h: int) -> list[list[float]]:
    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
        return []
    coords: list[float] = []
    for idx, value in enumerate(polygon):
        try:
            numeric = float(value)
        except Exception:
            return []
        coords.append(numeric * float(image_w if idx % 2 == 0 else image_h))
    return [coords]


def export_coco_dataset(
    *,
    dataset_root: str | Path,
    split: str | None,
    out_dir: str | Path,
    image_mode: str = "copy",
    force: bool = False,
) -> Path:
    from .dataset import build_manifest

    source_root = _resolve_dataset_root(dataset_root)
    manifest = build_manifest(source_root, split=split)
    records = manifest.get("images") or []
    split_effective = str(manifest.get("split") or split or "val2017")
    if not isinstance(records, list):
        raise ValueError("invalid dataset manifest (expected list under 'images')")

    out_root = _prepare_output_root(out_dir, force=force)
    images_dir = out_root / "images" / split_effective
    ann_dir = out_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    base_root, _ = _resolve_source_split(source_root, split_effective)
    classes_payload = _load_classes_payload(dataset_root=base_root, split=split_effective, records=records)
    class_names = _class_names_from_payload(classes_payload)
    class_to_category, _ = _category_mappings_from_payload(classes_payload, class_names=class_names)
    keypoint_names, skeleton = _keypoint_schema_from_payload(classes_payload)

    categories: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        category: dict[str, Any] = {"id": int(class_to_category.get(class_id, class_id + 1)), "name": str(name)}
        if keypoint_names and class_id == 0:
            category["keypoints"] = list(keypoint_names)
            if skeleton:
                category["skeleton"] = list(skeleton)
        categories.append(category)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, record in enumerate(records, start=1):
        image_path = Path(str(record.get("image")))
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        image_w, image_h = _record_image_size(record)
        _materialize_file(image_path, images_dir / image_path.name, mode=image_mode)

        images.append(
            {
                "id": int(image_id),
                "file_name": image_path.name,
                "width": int(image_w),
                "height": int(image_h),
            }
        )

        for label in record.get("labels") or []:
            class_id = int(label.get("class_id", 0))
            category_id = int(class_to_category.get(class_id, class_id + 1))
            bw = float(label.get("w", 0.0)) * float(image_w)
            bh = float(label.get("h", 0.0)) * float(image_h)
            cx = float(label.get("cx", 0.0)) * float(image_w)
            cy = float(label.get("cy", 0.0)) * float(image_h)
            x = cx - bw / 2.0
            y = cy - bh / 2.0
            if bw <= 0.0 or bh <= 0.0:
                continue
            annotation: dict[str, Any] = {
                "id": int(ann_id),
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            }
            segmentation = _polygon_to_coco_segmentation(label.get("polygon"), image_w=image_w, image_h=image_h)
            if segmentation:
                annotation["segmentation"] = segmentation
            if keypoint_names and label.get("keypoints") is not None and class_id == 0:
                flat_keypoints = label.get("keypoints")
                if isinstance(flat_keypoints, list) and flat_keypoints and isinstance(flat_keypoints[0], dict):
                    normalized: list[float] = []
                    for item in flat_keypoints:
                        if not isinstance(item, dict):
                            continue
                        normalized.extend(
                            [
                                float(item.get("x", 0.0)),
                                float(item.get("y", 0.0)),
                                float(item.get("v", 2.0)),
                            ]
                        )
                    flat_keypoints = normalized
                coco_keypoints, num_keypoints = _keypoints_to_coco_list(
                    flat_keypoints,
                    image_w=image_w,
                    image_h=image_h,
                    expected_count=len(keypoint_names),
                )
                annotation["keypoints"] = coco_keypoints
                annotation["num_keypoints"] = int(num_keypoints)
            annotations.append(annotation)
            ann_id += 1

    instances_path = ann_dir / f"instances_{split_effective}.json"
    coco_payload = {"images": images, "annotations": annotations, "categories": categories}
    instances_path.write_text(json.dumps(coco_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if keypoint_names:
        (ann_dir / f"person_keypoints_{split_effective}.json").write_text(
            json.dumps(coco_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return out_root


def export_yolo_dataset(
    *,
    dataset_root: str | Path,
    split: str | None,
    out_dir: str | Path,
    image_mode: str = "copy",
    force: bool = False,
) -> Path:
    from .dataset import build_manifest

    source_root = _resolve_dataset_root(dataset_root)
    manifest = build_manifest(source_root, split=split)
    records = manifest.get("images") or []
    split_effective = str(manifest.get("split") or split or "val")
    if not isinstance(records, list):
        raise ValueError("invalid dataset manifest (expected list under 'images')")

    out_root = _prepare_output_root(out_dir, force=force)
    images_dir = out_root / "images" / split_effective
    labels_dir = out_root / "labels" / split_effective
    base_root, _ = _resolve_source_split(source_root, split_effective)
    classes_payload = _load_classes_payload(dataset_root=base_root, split=split_effective, records=records)
    class_names = _write_classes_files(labels_dir, classes_payload)

    for record in records:
        image_path = Path(str(record.get("image")))
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        dst_image = images_dir / image_path.name
        _materialize_file(image_path, dst_image, mode=image_mode)

        image_wh = _record_image_size(record)
        label_lines = [_format_yolo_label(label, image_wh=image_wh) for label in (record.get("labels") or [])]
        (labels_dir / f"{image_path.stem}.txt").write_text(
            ("\n".join(label_lines) + "\n") if label_lines else "",
            encoding="utf-8",
        )

    split_alias = "train" if split_effective.lower().startswith("train") else "val"
    yaml_lines = [
        f"path: {out_root}",
        f"{split_alias}: images/{split_effective}",
        "names:",
    ]
    if class_names:
        yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    else:
        yaml_lines.append("  0: class_0")
    keypoint_names, _skeleton = _keypoint_schema_from_payload(classes_payload)
    if keypoint_names:
        yaml_lines.append(f"kpt_shape: [{len(keypoint_names)}, 3]")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return out_root


def export_kitti_dataset(
    *,
    dataset_root: str | Path,
    split: str | None,
    out_dir: str | Path,
    image_mode: str = "copy",
    force: bool = False,
) -> Path:
    from .dataset import build_manifest

    source_root = _resolve_dataset_root(dataset_root)
    manifest = build_manifest(source_root, split=split)
    records = manifest.get("images") or []
    split_effective = str(manifest.get("split") or split or "val")
    if not isinstance(records, list):
        raise ValueError("invalid dataset manifest (expected list under 'images')")

    out_root = _prepare_output_root(out_dir, force=force)
    images_dir = out_root / "image_2"
    labels_dir = out_root / "label_2"
    sets_dir = out_root / "ImageSets" / "Main"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    base_root, _ = _resolve_source_split(source_root, split_effective)
    classes_payload = _load_classes_payload(dataset_root=base_root, split=split_effective, records=records)
    class_names = _class_names_from_payload(classes_payload)
    if not class_names:
        class_names = [f"class_{idx}" for idx in range(len(classes_payload.get("class_id_to_category_id") or {}))]
    (out_root / "classes.txt").write_text(
        ("\n".join(class_names) + "\n") if class_names else "",
        encoding="utf-8",
    )

    stems: list[str] = []
    for record in records:
        image_path = Path(str(record.get("image")))
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        stems.append(image_path.stem)
        _materialize_file(image_path, images_dir / image_path.name, mode=image_mode)

        image_w, image_h = _record_image_size(record)
        label_lines = [
            _to_kitti_label_line(label, class_names=class_names, image_w=image_w, image_h=image_h)
            for label in (record.get("labels") or [])
        ]
        (labels_dir / f"{image_path.stem}.txt").write_text(
            ("\n".join(label_lines) + "\n") if label_lines else "",
            encoding="utf-8",
        )

    (sets_dir / f"{split_effective}.txt").write_text(
        ("\n".join(stems) + "\n") if stems else "",
        encoding="utf-8",
    )
    return out_root


def export_segmentation_dataset(
    *,
    dataset_root: str | Path,
    out_dir: str | Path,
    image_mode: str = "copy",
    force: bool = False,
) -> Path:
    from .segmentation_dataset import load_seg_dataset_descriptor, resolve_dataset_path

    source_root = _resolve_dataset_root(dataset_root)
    descriptor_path = source_root if source_root.is_file() else source_root / "dataset.json"
    desc = load_seg_dataset_descriptor(descriptor_path)

    out_root = _prepare_output_root(out_dir, force=force)
    images_dir = out_root / "images" / str(desc.split)
    masks_dir = out_root / "masks" / str(desc.split)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    out_samples: list[dict[str, Any]] = []
    for sample in desc.samples:
        image_src = resolve_dataset_path(sample.image, dataset_root=descriptor_path.parent, path_type=desc.path_type)
        image_dst = images_dir / image_src.name
        _materialize_file(image_src, image_dst, mode=image_mode)

        mask_out: str | None = None
        if sample.mask is not None:
            mask_src = resolve_dataset_path(sample.mask, dataset_root=descriptor_path.parent, path_type=desc.path_type)
            mask_dst = masks_dir / mask_src.name
            _materialize_file(mask_src, mask_dst, mode=image_mode)
            mask_out = str(Path("masks") / str(desc.split) / mask_dst.name)

        out_samples.append(
            {
                "id": sample.sample_id,
                "image": str(Path("images") / str(desc.split) / image_dst.name),
                "mask": mask_out,
            }
        )

    payload: dict[str, Any] = {
        "dataset": desc.dataset,
        "task": "semantic_segmentation",
        "split": desc.split,
        "mode": ("copy" if image_mode == "copy" else "symlink"),
        "path_type": "relative",
        "ignore_index": int(desc.ignore_index),
        "samples": out_samples,
    }
    if desc.classes is not None:
        payload["classes"] = list(desc.classes)
        (out_root / "classes.txt").write_text("\n".join(desc.classes) + "\n", encoding="utf-8")
    (out_root / "dataset.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_root
