"""Validation helpers for YOLO-format dataset records.

``validate_dataset_records`` checks image existence, label schema, normalised
bbox ranges, and optional metadata sanity — used by ``yolozu validate dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .image_size import get_image_size

__all__ = ["DatasetValidationResult", "validate_dataset_records"]


@dataclass(frozen=True)
class DatasetValidationResult:
    warnings: list[str]
    errors: list[str]

    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("\n".join(self.errors))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return int(value)


def validate_dataset_records(
    records: Iterable[dict[str, Any]],
    *,
    strict: bool = True,
    mode: str = "fail",
    check_images: bool = True,
) -> DatasetValidationResult:
    """Validate dataset records from build_manifest/load_yolo_dataset.

    mode:
      - fail: return errors and keep errors list populated
      - warn: downgrade errors to warnings (errors list will be empty)
    """

    mode = str(mode)
    if mode not in ("fail", "warn"):
        raise ValueError("mode must be one of: fail, warn")

    warnings: list[str] = []
    errors: list[str] = []

    def add_error(msg: str) -> None:
        if mode == "warn":
            warnings.append(msg)
        else:
            errors.append(msg)

    for idx, record in enumerate(records):
        where = f"records[{idx}]"
        if not isinstance(record, dict):
            add_error(f"{where}: record must be an object")
            continue

        image = record.get("image")
        if not isinstance(image, str) or not image:
            add_error(f"{where}: missing or invalid image path")
            continue

        image_path = Path(image)
        if check_images:
            if not image_path.exists():
                add_error(f"{where}: image file does not exist: {image}")
            else:
                try:
                    w, h = get_image_size(image_path)
                    if w <= 0 or h <= 0:
                        add_error(f"{where}: invalid image size: {w}x{h}")
                except Exception as exc:
                    add_error(f"{where}: failed to read image size: {image} ({exc})")

        labels = record.get("labels") or []
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            add_error(f"{where}: labels must be a list")
            labels = []

        for j, label in enumerate(labels):
            lwhere = f"{where}.labels[{j}]"
            if not isinstance(label, dict):
                add_error(f"{lwhere}: label must be an object")
                continue

            class_id = _as_int(label.get("class_id"))
            if class_id is None or class_id < 0:
                add_error(f"{lwhere}: class_id must be a non-negative int")

            for key in ("cx", "cy", "w", "h"):
                if key not in label:
                    add_error(f"{lwhere}: missing '{key}'")
                    continue
                val = _as_float(label.get(key))
                if val is None:
                    add_error(f"{lwhere}.{key}: must be a number")
                    continue
                if strict:
                    if key in ("w", "h") and val <= 0.0:
                        add_error(f"{lwhere}.{key}: must be > 0")
                    if val < 0.0 or val > 1.0:
                        add_error(f"{lwhere}.{key}: out of range [0,1] (got {val})")

            cx = _as_float(label.get("cx"))
            cy = _as_float(label.get("cy"))
            bw = _as_float(label.get("w"))
            bh = _as_float(label.get("h"))
            if strict and None not in (cx, cy, bw, bh):
                x1 = float(cx) - float(bw) / 2.0
                y1 = float(cy) - float(bh) / 2.0
                x2 = float(cx) + float(bw) / 2.0
                y2 = float(cy) + float(bh) / 2.0
                if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
                    add_error(f"{lwhere}: bbox extends outside image in normalized coords")

        # Metadata sanity checks (optional).
        image_hw = record.get("image_hw") or record.get("hw")
        if image_hw is not None and isinstance(image_hw, (list, tuple)) and len(image_hw) == 2:
            h0 = _as_float(image_hw[0])
            w0 = _as_float(image_hw[1])
            if h0 is None or w0 is None or h0 <= 0 or w0 <= 0:
                add_error(f"{where}: image_hw must be [h,w] positive numbers")

    return DatasetValidationResult(warnings=warnings, errors=errors)
