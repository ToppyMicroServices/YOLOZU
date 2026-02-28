"""Semantic-segmentation dataset descriptor loader.

Loads and validates a ``dataset.json`` descriptor that specifies
samples, class names, paths, and split information for segmentation
tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SegDatasetSample",
    "SegDatasetDescriptor",
    "load_seg_dataset_descriptor",
    "resolve_dataset_path",
]


@dataclass(frozen=True)
class SegDatasetSample:
    sample_id: str
    image: str
    mask: str | None


@dataclass(frozen=True)
class SegDatasetDescriptor:
    dataset: str
    task: str
    split: str
    mode: str
    path_type: str
    ignore_index: int
    classes: list[str] | None
    samples: list[SegDatasetSample]
    raw: dict[str, Any]


def _require_type(value: Any, expected: type | tuple[type, ...], *, where: str) -> None:
    if not isinstance(value, expected):
        name = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
        raise ValueError(f"{where} must be {name}")


def _require_str(value: Any, *, where: str) -> str:
    _require_type(value, str, where=where)
    if not value:
        raise ValueError(f"{where} must be non-empty string")
    return str(value)


def _require_int(value: Any, *, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} must be int")
    return int(value)


def load_seg_dataset_descriptor(path: str | Path) -> SegDatasetDescriptor:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_type(raw, dict, where="dataset")

    dataset = _require_str(raw.get("dataset"), where="dataset.dataset")
    task = _require_str(raw.get("task"), where="dataset.task")
    if task != "semantic_segmentation":
        raise ValueError(f"dataset.task must be 'semantic_segmentation' (got {task!r})")

    split = _require_str(raw.get("split"), where="dataset.split")
    mode = _require_str(raw.get("mode"), where="dataset.mode")
    if mode not in ("manifest", "symlink", "copy"):
        raise ValueError("dataset.mode must be one of: manifest, symlink, copy")
    path_type = _require_str(raw.get("path_type"), where="dataset.path_type")
    if path_type not in ("absolute", "relative"):
        raise ValueError("dataset.path_type must be one of: absolute, relative")

    ignore_index = _require_int(raw.get("ignore_index"), where="dataset.ignore_index")

    classes_raw = raw.get("classes")
    classes: list[str] | None
    if classes_raw is None:
        classes = None
    else:
        _require_type(classes_raw, list, where="dataset.classes")
        classes = []
        for i, name in enumerate(classes_raw):
            classes.append(_require_str(name, where=f"dataset.classes[{i}]"))

    samples_raw = raw.get("samples")
    _require_type(samples_raw, list, where="dataset.samples")

    samples: list[SegDatasetSample] = []
    for i, s in enumerate(samples_raw):
        _require_type(s, dict, where=f"dataset.samples[{i}]")
        sample_id = _require_str(s.get("id"), where=f"dataset.samples[{i}].id")
        image = _require_str(s.get("image"), where=f"dataset.samples[{i}].image")
        mask_raw = s.get("mask")
        mask: str | None
        if mask_raw is None:
            mask = None
        else:
            mask = _require_str(mask_raw, where=f"dataset.samples[{i}].mask")
        samples.append(SegDatasetSample(sample_id=sample_id, image=image, mask=mask))

    return SegDatasetDescriptor(
        dataset=dataset,
        task=task,
        split=split,
        mode=mode,
        path_type=path_type,
        ignore_index=ignore_index,
        classes=classes,
        samples=samples,
        raw=raw,
    )


def resolve_dataset_path(value: str, *, dataset_root: Path, path_type: str) -> Path:
    p = Path(value)
    if str(path_type) == "absolute":
        return p
    return dataset_root / p

