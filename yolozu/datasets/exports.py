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

__all__ = [
    "export_yolo_dataset",
    "export_kitti_dataset",
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
                return payload
        else:
            try:
                names = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                continue
            if names:
                return {"class_names": names}

    max_class_id = -1
    for record in records:
        for label in record.get("labels") or []:
            try:
                max_class_id = max(max_class_id, int(label.get("class_id")))
            except (TypeError, ValueError):
                continue

    class_names = [f"class_{idx}" for idx in range(max_class_id + 1)] if max_class_id >= 0 else []
    return {
        "class_names": class_names,
        "class_id_to_category_id": {str(idx): idx for idx in range(len(class_names))},
        "category_id_to_class_id": {str(idx): idx for idx in range(len(class_names))},
    }


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


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _format_yolo_label(label: dict[str, Any]) -> str:
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
        out = out + " " + " ".join(f"{float(value):.6g}" for value in keypoints)
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


def export_yolo_dataset(
    *,
    dataset_root: str | Path,
    split: str | None,
    out_dir: str | Path,
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
        _copy_image(image_path, dst_image)

        label_lines = [_format_yolo_label(label) for label in (record.get("labels") or [])]
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
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return out_root


def export_kitti_dataset(
    *,
    dataset_root: str | Path,
    split: str | None,
    out_dir: str | Path,
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
        _copy_image(image_path, images_dir / image_path.name)

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
