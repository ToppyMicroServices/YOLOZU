from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


PASCAL_VOC_SEG_CLASSES_21: list[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PASCAL_VOC_IGNORE_INDEX: int = 255


@dataclass(frozen=True)
class PascalVOCPaths:
    root: Path
    images_dir: Path
    masks_dir: Path
    split_dir: Path
    year: str | None


@dataclass(frozen=True)
class PascalVOCSample:
    image_path: Path
    mask_path: Path | None
    split: str
    sample_id: str


def resolve_pascal_voc_root(root: str | Path, *, year: str | None = None) -> PascalVOCPaths:
    """Resolve a Pascal VOC root to its canonical layout (JPEGImages, SegmentationClass, ImageSets/Segmentation).

    Args:
        root: Either a year dir (e.g. VOCdevkit/VOC2012) or VOCdevkit root.
        year: Optional year (e.g. "2012"). When provided and root is VOCdevkit, selects VOC{year}.
    """

    root_path = Path(root)

    def is_year_dir(path: Path) -> bool:
        return bool((path / "JPEGImages").is_dir() and (path / "ImageSets" / "Segmentation").is_dir())

    year_val: str | None = None
    if is_year_dir(root_path):
        if root_path.name.startswith("VOC") and root_path.name[3:].isdigit():
            year_val = root_path.name[3:]
        return PascalVOCPaths(
            root=root_path,
            images_dir=root_path / "JPEGImages",
            masks_dir=root_path / "SegmentationClass",
            split_dir=root_path / "ImageSets" / "Segmentation",
            year=year_val,
        )

    if year is not None:
        cand = root_path / f"VOC{year}"
        if not is_year_dir(cand):
            raise ValueError(f"VOC year directory not found under {root_path}: {cand}")
        return PascalVOCPaths(
            root=cand,
            images_dir=cand / "JPEGImages",
            masks_dir=cand / "SegmentationClass",
            split_dir=cand / "ImageSets" / "Segmentation",
            year=str(year),
        )

    # Common default.
    for guess in ("2012", "2007"):
        cand = root_path / f"VOC{guess}"
        if is_year_dir(cand):
            return PascalVOCPaths(
                root=cand,
                images_dir=cand / "JPEGImages",
                masks_dir=cand / "SegmentationClass",
                split_dir=cand / "ImageSets" / "Segmentation",
                year=guess,
            )

    # Best-effort: pick any VOC* child with expected structure.
    for cand in sorted(root_path.glob("VOC*")):
        if is_year_dir(cand):
            year_val = None
            if cand.name.startswith("VOC") and cand.name[3:].isdigit():
                year_val = cand.name[3:]
            return PascalVOCPaths(
                root=cand,
                images_dir=cand / "JPEGImages",
                masks_dir=cand / "SegmentationClass",
                split_dir=cand / "ImageSets" / "Segmentation",
                year=year_val,
            )

    raise ValueError(f"Pascal VOC layout not found under: {root_path}")


def _read_split_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        ids.append(line.split()[0])
    return ids


def iter_pascal_voc_seg_samples(
    root: str | Path,
    *,
    split: str = "train",
    year: str | None = None,
    masks_dirname: str = "SegmentationClass",
) -> Iterator[PascalVOCSample]:
    """Yield (image, mask) pairs for Pascal VOC semantic segmentation.

    Args:
        root: VOCdevkit root or VOC year root (e.g. VOC2012).
        split: train|val|trainval|test (must exist in ImageSets/Segmentation).
        year: Optional year selector when passing VOCdevkit root.
        masks_dirname: "SegmentationClass" (semantic) or another mask directory under the VOC year root.
    """

    split = str(split)
    paths = resolve_pascal_voc_root(root, year=year)
    split_file = paths.split_dir / f"{split}.txt"
    if not split_file.exists():
        raise ValueError(f"VOC split file not found: {split_file}")

    images_dir = paths.images_dir
    masks_dir = paths.root / str(masks_dirname)
    if not masks_dir.exists():
        raise ValueError(f"VOC mask directory not found: {masks_dir}")

    for sample_id in _read_split_ids(split_file):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
            cand = images_dir / f"{sample_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            raise ValueError(f"VOC image not found for id={sample_id} under {images_dir}")

        mask_path: Path | None = masks_dir / f"{sample_id}.png"
        if not mask_path.exists():
            if split == "test":
                mask_path = None
            else:
                raise ValueError(f"VOC mask not found for id={sample_id} (expected {masks_dir / (sample_id + '.png')})")

        yield PascalVOCSample(
            image_path=img_path,
            mask_path=mask_path,
            split=split,
            sample_id=sample_id,
        )

