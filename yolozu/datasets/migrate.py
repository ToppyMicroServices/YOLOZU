"""Dataset format migration utilities.

Converts external dataset layouts (Ultralytics YAML, COCO JSON,
segmentation descriptors) into YOLOZU’s native wrapper format.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "write_dataset_wrapper",
    "migrate_ultralytics_dataset_wrapper",
    "migrate_coco_dataset_wrapper",
    "migrate_coco_results_predictions",
    "migrate_predictions_entries_schema",
    "write_seg_dataset_descriptor",
    "migrate_seg_dataset_descriptor",
]

from yolozu.core.config import simple_yaml_load


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            return data or {}
        except Exception:
            return simple_yaml_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except Exception:
        return simple_yaml_load(text)


def _normalize_task(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    task = value.strip().lower()
    if not task:
        return None
    if task in ("detect", "det"):
        return "detect"
    if task in ("segment", "seg"):
        return "segment"
    if task in ("pose", "keypoints"):
        return "pose"
    return None


def _as_path(value: Any, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value.strip())
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def write_dataset_wrapper(
    output: str | Path,
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    split: str,
    label_format: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    out_path = Path(output)
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "dataset.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    payload: dict[str, Any] = {
        "images_dir": str(Path(images_dir)),
        "labels_dir": str(Path(labels_dir)),
        "split": str(split),
    }
    if label_format is not None:
        payload["label_format"] = str(label_format)
    if metadata is not None:
        for key, value in dict(metadata).items():
            if value is None:
                continue
            payload[str(key)] = value
    if source is not None:
        payload["source"] = dict(source)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def migrate_ultralytics_dataset_wrapper(
    *,
    data_yaml: str | Path | None,
    args_yaml: str | Path | None = None,
    split: str | None = None,
    task: str | None = None,
    output: str | Path,
    force: bool = False,
) -> Path:
    from .dataset import _resolve_ultralytics_data_yaml

    args_path = Path(args_yaml) if args_yaml is not None else None
    data_path = Path(data_yaml) if data_yaml is not None else None

    args_cfg: dict[str, Any] | None = None
    if args_path is not None:
        if not args_path.exists():
            raise FileNotFoundError(f"--args not found: {args_path}")
        args_cfg = _load_config(args_path)

    if data_path is None and args_cfg is not None:
        inferred = _as_path(args_cfg.get("data"), base=args_path.parent)
        if inferred is not None:
            data_path = inferred

    if data_path is None:
        raise ValueError("--data is required (or provide --args with a usable data: path)")
    if not data_path.exists():
        raise FileNotFoundError(f"--data not found: {data_path}")

    resolved = _resolve_ultralytics_data_yaml(Path(data_path), split)
    if resolved is None:
        raise ValueError(f"failed to resolve Ultralytics data.yaml: {data_path}")

    images_dir, labels_dir, split_effective, inferred_root, yaml_label_format = resolved

    task_norm = _normalize_task(task)
    args_task = _normalize_task(args_cfg.get("task")) if args_cfg is not None else None
    effective_task = task_norm or args_task

    label_format: str | None = None
    if effective_task == "segment":
        label_format = "segment"
    elif yaml_label_format is not None:
        label_format = str(yaml_label_format)

    source: dict[str, Any] = {"from": "ultralytics", "data_yaml": str(data_path)}
    if args_path is not None:
        source["args_yaml"] = str(args_path)
    if effective_task is not None:
        source["task"] = effective_task
    if inferred_root is not None:
        source["dataset_root"] = str(inferred_root)

    return write_dataset_wrapper(
        output,
        images_dir=images_dir,
        labels_dir=labels_dir,
        split=split_effective,
        label_format=label_format,
        source=source,
        force=force,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_coco_images_dir(coco_root: Path, split: str) -> Path:
    images_src = coco_root / "images" / split
    if images_src.exists():
        return images_src

    fallback = coco_root / split
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"COCO images not found for split={split} under {coco_root}")


def migrate_coco_dataset_wrapper(
    *,
    coco_root: str | Path,
    split: str,
    output: str | Path,
    instances_json: str | Path | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = False,
) -> Path:
    """Convert COCO instances JSON into YOLO labels + dataset.json wrapper.

    - mode=manifest: do not copy images; wrapper points to COCO images dir.
    - mode=copy: copy referenced images under output/images/<split>.
    - mode=symlink: symlink referenced images under output/images/<split>.
    """

    mode = str(mode).strip().lower()
    if mode not in ("manifest", "copy", "symlink"):
        raise ValueError("--mode must be manifest|copy|symlink")

    coco_root_path = Path(coco_root)
    if not coco_root_path.is_absolute():
        coco_root_path = (Path.cwd() / coco_root_path).resolve()
    split = str(split)
    images_src = _resolve_coco_images_dir(coco_root_path, split)

    if instances_json is None:
        instances_path = coco_root_path / "annotations" / f"instances_{split}.json"
    else:
        instances_path = Path(instances_json)
        if not instances_path.is_absolute():
            instances_path = (Path.cwd() / instances_path).resolve()
    if not instances_path.exists():
        raise FileNotFoundError(f"COCO instances JSON not found: {instances_path}")

    out_root = Path(output)
    if out_root.suffix.lower() == ".json":
        raise ValueError("--output must be a directory for --from coco (got .json path)")
    out_root.mkdir(parents=True, exist_ok=True)

    labels_dir = out_root / "labels" / split
    images_out = out_root / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    from .coco_convert import convert_coco_instances_to_yolo_labels

    instances_doc = _load_json(instances_path)
    convert_coco_instances_to_yolo_labels(
        instances_json=instances_doc,
        images_dir=images_src,
        labels_dir=labels_dir,
        include_crowd=bool(include_crowd),
    )

    if mode in ("copy", "symlink"):
        images = instances_doc.get("images") or []
        if not isinstance(images, list):
            raise ValueError("invalid COCO instances JSON: images must be a list")
        for img in images:
            file_name = str(img.get("file_name") or "").strip()
            if not file_name:
                continue
            src = images_src / file_name
            dst = images_out / Path(file_name).name
            if dst.exists():
                continue
            if mode == "copy":
                if src.exists():
                    shutil.copy2(src, dst)
            else:
                # symlink
                if src.exists():
                    dst.symlink_to(src)

    images_dir = images_src if mode == "manifest" else images_out
    source: dict[str, Any] = {
        "from": "coco",
        "coco_root": str(coco_root_path),
        "split": split,
        "instances_json": str(instances_path),
        "mode": mode,
        "include_crowd": bool(include_crowd),
    }

    return write_dataset_wrapper(
        out_root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        split=split,
        label_format="detect",
        source=source,
        force=force,
    )


def _normalize_coco_results_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            return [p for p in (payload.get("annotations") or []) if isinstance(p, dict)]
        if isinstance(payload.get("results"), list):
            return [p for p in (payload.get("results") or []) if isinstance(p, dict)]
    raise ValueError("unsupported COCO results JSON (expected a list[dict])")


def migrate_coco_results_predictions(
    *,
    results_json: str | Path,
    instances_json: str | Path,
    output: str | Path,
    score_threshold: float = 0.0,
    force: bool = False,
) -> Path:
    """Convert COCO detection results into YOLOZU predictions.json entries."""

    results_path = Path(results_json)
    instances_path = Path(instances_json)
    out_path = Path(output)

    if not results_path.exists():
        raise FileNotFoundError(f"--results not found: {results_path}")
    if not instances_path.exists():
        raise FileNotFoundError(f"--instances not found: {instances_path}")

    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instances_doc = _load_json(instances_path)
    images = instances_doc.get("images") or []
    if not isinstance(images, list):
        raise ValueError("invalid COCO instances JSON: images must be a list")

    image_id_to_meta: dict[int, dict[str, Any]] = {}
    for img in images:
        if not isinstance(img, dict) or "id" not in img:
            continue
        try:
            image_id_to_meta[int(img["id"])] = img
        except Exception:
            continue

    from .coco_convert import build_category_map_from_coco

    cat_map = build_category_map_from_coco(instances_doc)

    raw_results = _normalize_coco_results_payload(_load_json(results_path))
    grouped: dict[str, list[dict[str, Any]]] = {}

    for det in raw_results:
        try:
            image_id = int(det["image_id"])
            category_id = int(det["category_id"])
        except Exception:
            continue

        score = det.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if float(score) < float(score_threshold):
            continue

        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        meta = image_id_to_meta.get(image_id)
        if meta is None:
            continue
        file_name = str(meta.get("file_name") or "").strip()
        if not file_name:
            continue
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if w <= 0.0 or h <= 0.0:
            continue

        cx = (x + w / 2.0) / float(width)
        cy = (y + h / 2.0) / float(height)
        wn = w / float(width)
        hn = h / float(height)

        class_id = cat_map.category_id_to_class_id.get(category_id)
        if class_id is None:
            continue

        grouped.setdefault(file_name, []).append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "bbox": {"cx": float(cx), "cy": float(cy), "w": float(wn), "h": float(hn)},
            }
        )

    entries: list[dict[str, Any]] = []
    for image in sorted(grouped.keys()):
        dets = grouped[image]
        dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        entries.append({"image": image, "detections": dets})

    out_path.write_text(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def migrate_coco_keypoints_results_predictions(
    *,
    results_json: str | Path,
    instances_json: str | Path,
    output: str | Path,
    score_threshold: float = 0.0,
    force: bool = False,
) -> Path:
    """Convert COCO-style keypoints results into YOLOZU predictions entries."""

    results_path = Path(results_json)
    instances_path = Path(instances_json)
    out_path = Path(output)

    if not results_path.exists():
        raise FileNotFoundError(f"--results-json not found: {results_path}")
    if not instances_path.exists():
        raise FileNotFoundError(f"--instances-json not found: {instances_path}")
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instances_doc = _load_json(instances_path)
    images = instances_doc.get("images") or []
    if not isinstance(images, list):
        raise ValueError("invalid COCO instances JSON: images must be a list")

    image_id_to_meta: dict[int, dict[str, Any]] = {}
    for img in images:
        if not isinstance(img, dict) or "id" not in img:
            continue
        try:
            image_id_to_meta[int(img["id"])] = img
        except Exception:
            continue

    from .coco_convert import build_category_map_from_coco

    cat_map = build_category_map_from_coco(instances_doc)
    raw_results = _normalize_coco_results_payload(_load_json(results_path))
    grouped: dict[str, list[dict[str, Any]]] = {}

    for det in raw_results:
        try:
            image_id = int(det["image_id"])
            category_id = int(det["category_id"])
        except Exception:
            continue

        score = det.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if float(score) < float(score_threshold):
            continue

        bbox = det.get("bbox")
        keypoints = det.get("keypoints")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not isinstance(keypoints, list) or len(keypoints) < 3 or len(keypoints) % 3 != 0:
            continue

        meta = image_id_to_meta.get(image_id)
        if meta is None:
            continue
        file_name = str(meta.get("file_name") or "").strip()
        if not file_name:
            continue
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        class_id = cat_map.category_id_to_class_id.get(category_id)
        if class_id is None:
            continue

        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if w <= 0.0 or h <= 0.0:
            continue
        cx = (x + w / 2.0) / float(width)
        cy = (y + h / 2.0) / float(height)
        wn = w / float(width)
        hn = h / float(height)

        normalized_keypoints: list[dict[str, Any]] = []
        for idx in range(0, len(keypoints), 3):
            xk = float(keypoints[idx]) / float(width)
            yk = float(keypoints[idx + 1]) / float(height)
            vis = float(keypoints[idx + 2])
            payload: dict[str, Any] = {"x": float(xk), "y": float(yk)}
            if 0.0 <= vis <= 2.0:
                payload["v"] = float(vis)
            else:
                payload["score"] = float(vis)
            normalized_keypoints.append(payload)

        grouped.setdefault(file_name, []).append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "bbox": {"cx": float(cx), "cy": float(cy), "w": float(wn), "h": float(hn)},
                "keypoints": normalized_keypoints,
            }
        )

    entries: list[dict[str, Any]] = []
    for image in sorted(grouped.keys()):
        dets = grouped[image]
        dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        entries.append({"image": image, "detections": dets})

    out_path.write_text(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _schema_tag_to_int(value: str) -> int:
    text = str(value).strip().lower()
    if text.startswith("v"):
        text = text[1:]
    return int(text)


def migrate_predictions_entries_schema(
    *,
    input_path: str | Path,
    output: str | Path,
    from_version: str = "v1",
    to_version: str = "v2",
    strict_source: bool = False,
    force: bool = False,
) -> Path:
    """Migrate predictions entry schema versions (currently v1 -> v2)."""

    source_v = _schema_tag_to_int(from_version)
    target_v = _schema_tag_to_int(to_version)
    if source_v != 1 or target_v != 2:
        raise ValueError("only --from v1 --to v2 is currently supported")

    in_path = Path(input_path)
    out_path = Path(output)
    if not in_path.exists():
        raise FileNotFoundError(f"--input not found: {in_path}")
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_json(in_path)
    is_wrapper = isinstance(payload, dict) and isinstance(payload.get("predictions"), list)

    from yolozu.predictions import canonicalize_predictions, normalize_predictions_payload

    entries, wrapped_meta = normalize_predictions_payload(payload)

    incompatible_indices: list[int] = []
    migrated_candidates = 0
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        raw = entry.get("schema_version")
        if raw is None:
            migrated_candidates += 1
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            if raw == 1:
                migrated_candidates += 1
            elif raw != 2:
                incompatible_indices.append(idx)
            elif bool(strict_source):
                incompatible_indices.append(idx)
        else:
            incompatible_indices.append(idx)

    if incompatible_indices:
        raise ValueError(
            "input contains entries that do not satisfy --from version policy: "
            f"indices={incompatible_indices[:10]}"
        )

    canonical = canonicalize_predictions(
        entries,
        strict=False,
        policy="clamp",
        unknown_keys=("error" if strict_source else "warn"),
    )

    migration_meta = {
        "tool": "yolozu predictions migrate",
        "from_entry_schema_version": int(source_v),
        "to_entry_schema_version": int(target_v),
        "entry_count": int(len(canonical.entries)),
        "migrated_candidates": int(migrated_candidates),
        "warnings": list(canonical.warnings),
    }

    if is_wrapper:
        out_payload = dict(payload)
        out_payload["predictions"] = canonical.entries
        if wrapped_meta is None:
            meta: dict[str, Any] = {}
        else:
            meta = dict(wrapped_meta)
        meta["entry_schema_migration"] = migration_meta
        out_payload["meta"] = meta
    else:
        out_payload = canonical.entries

    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _maybe_relative(path: Path | None, *, root: Path, path_type: str) -> str | None:
    if path is None:
        return None
    if path_type == "absolute":
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def write_seg_dataset_descriptor(
    output: str | Path,
    *,
    dataset: str,
    split: str,
    ignore_index: int,
    samples: Iterable[dict[str, Any]],
    classes: list[str] | None,
    path_type: str = "absolute",
    mode: str = "manifest",
    source: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"output already exists: {out_path} (use --force to overwrite)")

    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "task": "semantic_segmentation",
        "split": str(split),
        "mode": str(mode),
        "path_type": str(path_type),
        "ignore_index": int(ignore_index),
        "samples": list(samples),
    }
    if classes is not None:
        payload["classes"] = list(classes)
    if source is not None:
        payload["source"] = dict(source)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def migrate_seg_dataset_descriptor(
    *,
    from_format: str,
    root: str | Path,
    split: str,
    output: str | Path,
    path_type: str = "absolute",
    mode: str = "manifest",
    force: bool = False,
    voc_year: str | None = None,
    voc_masks_dirname: str = "SegmentationClass",
    cityscapes_label_type: str = "labelTrainIds",
) -> Path:
    from_format = str(from_format).strip().lower()
    path_type = str(path_type).strip().lower()
    mode = str(mode).strip().lower()
    if path_type not in ("absolute", "relative"):
        raise ValueError("--path-type must be absolute|relative")
    if mode not in ("manifest", "symlink", "copy"):
        raise ValueError("--mode must be manifest|symlink|copy")

    root_path = Path(root)
    source: dict[str, Any] = {"from": from_format, "root": str(root_path), "split": str(split)}

    if from_format in ("voc", "pascal_voc", "pascal-voc"):
        from .pascal_voc import (
            PASCAL_VOC_IGNORE_INDEX,
            PASCAL_VOC_SEG_CLASSES_21,
            resolve_pascal_voc_root,
            iter_pascal_voc_seg_samples,
        )

        paths = resolve_pascal_voc_root(root_path, year=voc_year)
        samples_out: list[dict[str, Any]] = []
        for sample in iter_pascal_voc_seg_samples(
            root_path,
            split=split,
            year=voc_year,
            masks_dirname=voc_masks_dirname,
        ):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.root, path_type=path_type),
                }
            )

        dataset_name = "pascal_voc" if paths.year is None else f"pascal_voc{paths.year}"
        source.update({"year": paths.year, "dataset_root_hint": str(paths.root)})
        return write_seg_dataset_descriptor(
            output,
            dataset=dataset_name,
            split=split,
            ignore_index=PASCAL_VOC_IGNORE_INDEX,
            samples=samples_out,
            classes=PASCAL_VOC_SEG_CLASSES_21,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    if from_format in ("cityscapes",):
        from .cityscapes import (
            CITYSCAPES_IGNORE_INDEX,
            CITYSCAPES_TRAIN_CLASSES_19,
            resolve_cityscapes_paths,
            iter_cityscapes_samples,
        )

        paths = resolve_cityscapes_paths(root_path)
        samples_out = []
        for sample in iter_cityscapes_samples(root_path, split=split, label_type=cityscapes_label_type):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.images_root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.labels_root or root_path, path_type=path_type),
                }
            )

        source.update({"label_type": cityscapes_label_type, "dataset_root_hint": str(root_path)})
        return write_seg_dataset_descriptor(
            output,
            dataset="cityscapes",
            split=split,
            ignore_index=CITYSCAPES_IGNORE_INDEX,
            samples=samples_out,
            classes=CITYSCAPES_TRAIN_CLASSES_19,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    if from_format in ("ade20k", "ade"):
        from .ade20k import ADE20K_IGNORE_INDEX, resolve_ade20k_paths, iter_ade20k_samples, load_ade20k_classes

        paths = resolve_ade20k_paths(root_path)
        samples_out = []
        for sample in iter_ade20k_samples(root_path, split=split):
            samples_out.append(
                {
                    "id": str(sample.sample_id),
                    "image": _maybe_relative(sample.image_path, root=paths.root, path_type=path_type),
                    "mask": _maybe_relative(sample.mask_path, root=paths.root, path_type=path_type),
                }
            )

        classes = load_ade20k_classes(root_path)
        source.update({"dataset_root_hint": str(paths.root)})
        return write_seg_dataset_descriptor(
            output,
            dataset="ade20k",
            split=split,
            ignore_index=ADE20K_IGNORE_INDEX,
            samples=samples_out,
            classes=classes,
            path_type=path_type,
            mode=mode,
            source=source,
            force=force,
        )

    raise ValueError(f"unsupported --from: {from_format}")
