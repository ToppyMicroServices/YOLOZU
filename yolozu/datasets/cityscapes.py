"""Cityscapes dataset adapter (semantic segmentation).

Supports the standard Cityscapes directory layout::

    <root>/
        leftImg8bit/
            train/ val/ test/
                <city>/
                    <city>_000000_000000_leftImg8bit.png
        gtFine/
            train/ val/ test/
                <city>/
                    <city>_000000_000000_gtFine_labelTrainIds.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .registry import DatasetInfo, DatasetSample, register_adapter

__all__ = [
    "CITYSCAPES_TRAIN_CLASSES_19",
    "CITYSCAPES_IGNORE_INDEX",
    "CityscapesPaths",
    "CityscapesSample",
    "resolve_cityscapes_paths",
    "iter_cityscapes_samples",
    "CityscapesAdapter",
]


CITYSCAPES_TRAIN_CLASSES_19: list[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_IGNORE_INDEX: int = 255


@dataclass(frozen=True)
class CityscapesPaths:
    images_root: Path
    labels_root: Path | None


@dataclass(frozen=True)
class CityscapesSample:
    image_path: Path
    mask_path: Path | None
    split: str
    city: str
    sample_id: str


def _pick_existing(root: Path, candidates: Iterable[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_cityscapes_paths(root: str | Path) -> CityscapesPaths:
    """Resolve Cityscapes directory layout (supports both common downloads).

    Accepts either:
      - <root>/leftImg8bit + <root>/gtFine
      - <root>/leftImg8bit_trainvaltest/leftImg8bit + <root>/gtFine_trainvaltest/gtFine
    """

    root = Path(root)
    images_root = _pick_existing(
        root,
        (
            root / "leftImg8bit",
            root / "leftImg8bit_trainvaltest" / "leftImg8bit",
        ),
    )
    if images_root is None:
        raise ValueError(f"Cityscapes images root not found under: {root}")

    labels_root = _pick_existing(
        root,
        (
            root / "gtFine",
            root / "gtFine_trainvaltest" / "gtFine",
        ),
    )
    return CityscapesPaths(images_root=images_root, labels_root=labels_root)


def _strip_suffix(value: str, suffix: str) -> str:
    if not value.endswith(suffix):
        raise ValueError(f"expected suffix '{suffix}': {value}")
    return value[: -len(suffix)]


def iter_cityscapes_samples(
    root: str | Path,
    *,
    split: str = "train",
    label_type: str = "labelTrainIds",
) -> Iterator[CityscapesSample]:
    """Yield Cityscapes (image, mask) pairs for semantic segmentation.

    Args:
        root: Cityscapes root.
        split: train|val|test.
        label_type: labelTrainIds|labelIds. For test split, masks are always None.
    """

    split = str(split)
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train, val, test")

    label_type = str(label_type)
    if label_type not in ("labelTrainIds", "labelIds"):
        raise ValueError("label_type must be one of: labelTrainIds, labelIds")

    paths = resolve_cityscapes_paths(root)
    split_dir = paths.images_root / split
    if not split_dir.exists():
        raise ValueError(f"Cityscapes split dir not found: {split_dir}")

    images = sorted(split_dir.rglob("*_leftImg8bit.png"))
    for img_path in images:
        city = img_path.parent.name
        base = _strip_suffix(img_path.name, "_leftImg8bit.png")
        sample_id = base

        mask_path: Path | None = None
        if split != "test":
            if paths.labels_root is None:
                raise ValueError("Cityscapes labels root not found (gtFine)")
            mask_path = paths.labels_root / split / city / f"{base}_gtFine_{label_type}.png"
            if not mask_path.exists():
                raise ValueError(f"Cityscapes mask not found for image: {img_path} (expected {mask_path})")

        yield CityscapesSample(
            image_path=img_path,
            mask_path=mask_path,
            split=split,
            city=city,
            sample_id=sample_id,
        )


# ---------------------------------------------------------------------------
# Adapter class (registered in the global registry)
# ---------------------------------------------------------------------------

class CityscapesAdapter:
    """Cityscapes dataset adapter for the unified registry."""

    format_name = "cityscapes"

    def probe(self, root: Path) -> DatasetInfo | None:
        """Return ``DatasetInfo`` if *root* has a Cityscapes layout, else ``None``."""
        try:
            paths = resolve_cityscapes_paths(root)
        except ValueError:
            return None

        splits: list[str] = []
        for name in ("train", "val", "test"):
            if (paths.images_root / name).is_dir():
                splits.append(name)

        if not splits:
            return None

        return DatasetInfo(
            format_name=self.format_name,
            root=paths.images_root.parent,
            splits=splits,
            task="segmentation",
            num_classes=19,
            class_names=CITYSCAPES_TRAIN_CLASSES_19,
        )

    def iter_samples(
        self,
        root: Path,
        *,
        split: str = "train",
        **kwargs: Any,
    ) -> Iterator[DatasetSample]:
        """Yield ``DatasetSample`` wrapping Cityscapes segmentation pairs."""
        for cs in iter_cityscapes_samples(root, split=split, **kwargs):
            yield DatasetSample(
                image_path=cs.image_path,
                split=cs.split,
                sample_id=cs.sample_id,
                mask_path=cs.mask_path,
                extra={"city": cs.city},
            )


# Auto-register on import.
register_adapter(CityscapesAdapter())
