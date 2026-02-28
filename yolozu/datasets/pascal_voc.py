"""Pascal VOC dataset adapter (detection + semantic segmentation).

Supports the standard VOCdevkit layout::

    <root>/
        VOC2012/
            JPEGImages/
            Annotations/          (XML per image — detection)
            SegmentationClass/    (PNG masks — semantic segmentation)
            ImageSets/
                Main/             (train.txt, val.txt, … — detection)
                Segmentation/     (train.txt, val.txt, … — segmentation)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .registry import DatasetInfo, DatasetSample, register_adapter

__all__ = [
    "PASCAL_VOC_SEG_CLASSES_21",
    "PASCAL_VOC_DET_CLASSES_20",
    "PASCAL_VOC_IGNORE_INDEX",
    "PascalVOCPaths",
    "PascalVOCSample",
    "resolve_pascal_voc_root",
    "iter_pascal_voc_seg_samples",
    "parse_voc_xml",
    "iter_pascal_voc_det_samples",
    "PascalVOCAdapter",
]

# -- 20 detection classes (no background) -----------------------------------

PASCAL_VOC_DET_CLASSES_20: list[str] = [
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

_DET_CLASS_TO_ID: dict[str, int] = {n: i for i, n in enumerate(PASCAL_VOC_DET_CLASSES_20)}

# -- 21 segmentation classes (with background) ------------------------------

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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def parse_voc_xml(xml_path: str | Path) -> dict[str, Any]:
    """Parse a Pascal VOC annotation XML file and return a normalised dict.

    Returns a dict with keys:

    * ``filename`` (``str``)
    * ``size`` — ``{"width": int, "height": int, "depth": int}``
    * ``objects`` — list of ``{"name": str, "difficult": bool,
      "bndbox": {"xmin": int, "ymin": int, "xmax": int, "ymax": int}}``
    """
    tree = ET.parse(str(xml_path))  # noqa: S314
    root = tree.getroot()

    filename = (root.findtext("filename") or "").strip()

    size_el = root.find("size")
    width = int(size_el.findtext("width") or "0") if size_el is not None else 0
    height = int(size_el.findtext("height") or "0") if size_el is not None else 0
    depth = int(size_el.findtext("depth") or "3") if size_el is not None else 3

    objects: list[dict[str, Any]] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        difficult = int(obj.findtext("difficult") or "0") != 0
        bndbox_el = obj.find("bndbox")
        if bndbox_el is None:
            continue
        xmin = int(float(bndbox_el.findtext("xmin") or "0"))
        ymin = int(float(bndbox_el.findtext("ymin") or "0"))
        xmax = int(float(bndbox_el.findtext("xmax") or "0"))
        ymax = int(float(bndbox_el.findtext("ymax") or "0"))
        objects.append({
            "name": name,
            "difficult": difficult,
            "bndbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        })

    return {
        "filename": filename,
        "size": {"width": width, "height": height, "depth": depth},
        "objects": objects,
    }


def iter_pascal_voc_det_samples(
    root: str | Path,
    *,
    split: str = "train",
    year: str | None = None,
    include_difficult: bool = False,
) -> Iterator[DatasetSample]:
    """Yield detection samples from a Pascal VOC dataset.

    Each sample's ``labels`` list contains YOLO-normalised bboxes::

        {"class_id": int, "cx": float, "cy": float, "w": float, "h": float}

    Args:
        root: VOCdevkit root or a year directory (e.g. VOC2012).
        split: train|val|trainval|test (must exist in ``ImageSets/Main``).
        year: Optional year selector when root is VOCdevkit.
        include_difficult: If ``False`` (default), skip objects marked ``difficult``.
    """
    paths = resolve_pascal_voc_root(root, year=year)
    main_dir = paths.root / "ImageSets" / "Main"
    split_file = main_dir / f"{split}.txt"
    if not split_file.exists():
        raise ValueError(f"VOC detection split file not found: {split_file}")

    annotations_dir = paths.root / "Annotations"
    if not annotations_dir.is_dir():
        raise ValueError(f"VOC Annotations directory not found: {annotations_dir}")

    images_dir = paths.images_dir

    for sample_id in _read_split_ids(split_file):
        xml_path = annotations_dir / f"{sample_id}.xml"
        if not xml_path.exists():
            continue  # skip missing annotations (common for test split)

        ann = parse_voc_xml(xml_path)
        width = ann["size"]["width"]
        height = ann["size"]["height"]
        if width <= 0 or height <= 0:
            continue

        # Resolve image path
        img_path: Path | None = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
            cand = images_dir / f"{sample_id}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        labels: list[dict[str, Any]] = []
        for obj in ann["objects"]:
            if not include_difficult and obj["difficult"]:
                continue
            name = obj["name"].lower()
            class_id = _DET_CLASS_TO_ID.get(name)
            if class_id is None:
                continue
            bb = obj["bndbox"]
            bw = bb["xmax"] - bb["xmin"]
            bh = bb["ymax"] - bb["ymin"]
            if bw <= 0 or bh <= 0:
                continue
            labels.append({
                "class_id": class_id,
                "cx": float((bb["xmin"] + bw / 2.0) / width),
                "cy": float((bb["ymin"] + bh / 2.0) / height),
                "w": float(bw / width),
                "h": float(bh / height),
            })

        yield DatasetSample(
            image_path=img_path,
            split=split,
            sample_id=sample_id,
            labels=labels,
            annotation_path=xml_path,
            extra={"image_hw": [height, width]},
        )


# ---------------------------------------------------------------------------
# Adapter class (registered in the global registry)
# ---------------------------------------------------------------------------

class PascalVOCAdapter:
    """Pascal VOC dataset adapter for the unified registry (detection + segmentation)."""

    format_name = "pascal_voc"

    def probe(self, root: Path) -> DatasetInfo | None:
        """Return ``DatasetInfo`` if *root* has a Pascal VOC layout, else ``None``."""
        try:
            paths = resolve_pascal_voc_root(root)
        except ValueError:
            return None

        # Determine available splits from ImageSets/Main (detection) first.
        splits: list[str] = []
        main_dir = paths.root / "ImageSets" / "Main"
        if main_dir.is_dir():
            for p in sorted(main_dir.glob("*.txt")):
                name = p.stem
                if name in ("train", "val", "trainval", "test"):
                    splits.append(name)

        # Determine task based on available directories.
        has_annotations = (paths.root / "Annotations").is_dir()
        has_seg = paths.masks_dir.is_dir()
        if has_annotations and has_seg:
            task = "multi"
        elif has_annotations:
            task = "detection"
        else:
            task = "segmentation"

        if not splits:
            # Fallback — check Segmentation splits.
            if paths.split_dir.is_dir():
                for p in sorted(paths.split_dir.glob("*.txt")):
                    splits.append(p.stem)
            if not splits:
                return None

        return DatasetInfo(
            format_name=self.format_name,
            root=paths.root,
            splits=splits,
            task=task,
            num_classes=20,
            class_names=PASCAL_VOC_DET_CLASSES_20,
        )

    def iter_samples(
        self,
        root: Path,
        *,
        split: str = "train",
        **kwargs: Any,
    ) -> Iterator[DatasetSample]:
        """Yield detection samples (falls back to segmentation if no Annotations dir)."""
        paths = resolve_pascal_voc_root(root)
        if (paths.root / "Annotations").is_dir():
            yield from iter_pascal_voc_det_samples(root, split=split, **kwargs)
        else:
            # Segmentation-only layout — wrap PascalVOCSample → DatasetSample.
            for seg in iter_pascal_voc_seg_samples(root, split=split, **kwargs):
                yield DatasetSample(
                    image_path=seg.image_path,
                    split=seg.split,
                    sample_id=seg.sample_id,
                    mask_path=seg.mask_path,
                )


# Auto-register on import.
register_adapter(PascalVOCAdapter())
