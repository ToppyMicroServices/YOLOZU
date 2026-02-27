from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


ADE20K_IGNORE_INDEX: int = 255


@dataclass(frozen=True)
class ADE20KPaths:
    root: Path
    images_root: Path
    annotations_root: Path


@dataclass(frozen=True)
class ADE20KSample:
    image_path: Path
    mask_path: Path | None
    split: str
    sample_id: str


def resolve_ade20k_paths(root: str | Path) -> ADE20KPaths:
    """Resolve an ADE20K root to a layout with images/ and annotations/."""

    root_path = Path(root)

    def is_layout(path: Path) -> bool:
        return bool((path / "images").is_dir() and (path / "annotations").is_dir())

    if is_layout(root_path):
        return ADE20KPaths(
            root=root_path,
            images_root=root_path / "images",
            annotations_root=root_path / "annotations",
        )

    for cand_name in ("ADEChallengeData2016", "ADE20K_2021_17_01"):
        cand = root_path / cand_name
        if is_layout(cand):
            return ADE20KPaths(
                root=cand,
                images_root=cand / "images",
                annotations_root=cand / "annotations",
            )

    for cand in sorted(root_path.iterdir()) if root_path.exists() else []:
        if cand.is_dir() and is_layout(cand):
            return ADE20KPaths(
                root=cand,
                images_root=cand / "images",
                annotations_root=cand / "annotations",
            )

    raise ValueError(f"ADE20K layout not found under: {root_path}")


def _split_to_src(split: str) -> str:
    split = str(split)
    if split in ("train", "training"):
        return "training"
    if split in ("val", "validation"):
        return "validation"
    if split == "test":
        return "test"
    raise ValueError("split must be one of: train, val, test")


def iter_ade20k_samples(
    root: str | Path,
    *,
    split: str = "train",
) -> Iterator[ADE20KSample]:
    """Yield (image, mask) pairs for ADE20K semantic segmentation.

    Supported split aliases:
      - train -> training
      - val -> validation
      - test -> test
    """

    paths = resolve_ade20k_paths(root)

    split_out = str(split)
    split_src = _split_to_src(split_out)
    images_dir = paths.images_root / split_src
    masks_dir = paths.annotations_root / split_src
    if not images_dir.exists():
        raise ValueError(f"ADE20K images split dir not found: {images_dir}")

    images: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"):
        images.extend(images_dir.glob(ext))
    images = sorted(images)

    for img_path in images:
        sample_id = img_path.stem
        mask_path: Path | None = masks_dir / f"{sample_id}.png"
        if not mask_path.exists():
            if split_src == "test":
                mask_path = None
            else:
                raise ValueError(f"ADE20K mask not found for image: {img_path} (expected {mask_path})")

        yield ADE20KSample(
            image_path=img_path,
            mask_path=mask_path,
            split=split_out,
            sample_id=sample_id,
        )


def load_ade20k_classes(root: str | Path) -> list[str] | None:
    """Load ADE20K class names (optional) from objectInfo150.txt if present.

    Returns a list like ["background", ...] or None when the file is missing or unparseable.
    """

    paths = resolve_ade20k_paths(root)
    info_path = paths.root / "objectInfo150.txt"
    if not info_path.exists():
        return None

    entries: dict[int, str] = {}
    for raw in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "\t" in line:
            cols = [c.strip() for c in line.split("\t") if c.strip()]
            if not cols:
                continue
            if not cols[0].isdigit():
                continue
            idx = int(cols[0])
            name = str(cols[-1]).strip()
        else:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            idx = int(parts[0])
            name = " ".join(parts[1:]).strip()

        if not name:
            continue
        entries[idx] = name

    if not entries:
        return None

    ordered = [entries[i] for i in sorted(entries)]
    if ordered and ordered[0].lower() == "background":
        return ordered
    return ["background", *ordered]

