"""Generate a small, portable labeled dataset without downloads or inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from yolozu import __version__ as yolozu_version

__all__ = ["generate_labeled_sample"]

_RECIPE = "circles_rectangles_v1"
_WIDTH, _HEIGHT = 320, 240
_NAMES = ["circle", "rectangle"]
_COLORS = [(50, 119, 190), (224, 111, 49)]
_BACKGROUND = (245, 247, 250)


def _integer(seed: int, split: str, index: int, field: str, lower: int, upper: int) -> int:
    # Hash-based coordinates avoid depending on a runtime's random implementation.
    key = f"{_RECIPE}:{seed}:{split}:{index}:{field}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return lower + value % (upper - lower + 1)


def _objects(seed: int, split: str, index: int) -> list[dict[str, Any]]:
    if split == "val" and index == 3:
        return []
    objects = []
    for class_id in range(2):
        prefix = _NAMES[class_id]
        x0 = _integer(seed, split, index, f"{prefix}.x", 25 if class_id == 0 else 180, 75 if class_id == 0 else 230)
        y0 = _integer(seed, split, index, f"{prefix}.y", 25, 140)
        width = _integer(seed, split, index, f"{prefix}.width", 42, 70)
        height = width if class_id == 0 else _integer(seed, split, index, f"{prefix}.height", 35, 75)
        objects.append({"class_id": class_id, "bbox_xyxy": [x0, y0, x0 + width, y0 + height]})
    return objects


def _normalized_bbox(obj: dict[str, Any]) -> dict[str, float]:
    x0, y0, x1, y1 = obj["bbox_xyxy"]
    return {
        "cx": (x0 + x1) / (2 * _WIDTH),
        "cy": (y0 + y1) / (2 * _HEIGHT),
        "w": (x1 - x0) / _WIDTH,
        "h": (y1 - y0) / _HEIGHT,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def generate_labeled_sample(*, run_dir: str | Path, seed: int = 0) -> Path:
    """Create eight labeled 320x240 images and return ``sample_manifest.json``.

    The directory must not exist, including as a symlink. Nothing is overwritten.
    Coordinates are deterministic for a seed and recipe version. Validation
    predictions come from the labels, not a model; their scores test the workflow,
    not model quality. All paths stored in the artifacts are relative to run_dir.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=False)

    classes = {
        "schema_version": 1,
        "names": _NAMES,
        "class_to_category_id": {"0": 1, "1": 2},
        "category_id_to_class_id": {"1": 0, "2": 1},
    }
    samples: list[dict[str, Any]] = []
    predictions = []
    preview = Image.new("RGB", (2 * _WIDTH, 40 + 4 * (_HEIGHT + 28)), "white")
    preview_draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=14)
    preview_draw.text((10, 12), "Synthetic labeled sample - workflow fixture, not model quality", fill="black", font=font)

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        _write_json(root / "labels" / split / "classes.json", classes)
        for index in range(4):
            filename = f"{split}_{index + 1:04d}"
            image_rel = f"images/{split}/{filename}.png"
            label_rel = f"labels/{split}/{filename}.txt"
            objects = _objects(seed, split, index)
            image = Image.new("RGB", (_WIDTH, _HEIGHT), _BACKGROUND)
            draw = ImageDraw.Draw(image)
            for obj in objects:
                x0, y0, x1, y1 = obj["bbox_xyxy"]
                box = (x0, y0, x1 - 1, y1 - 1)
                if obj["class_id"] == 0:
                    draw.ellipse(box, fill=_COLORS[0])
                else:
                    draw.rectangle(box, fill=_COLORS[1])
            image.save(root / image_rel)
            labels = []
            detections = []
            for obj in objects:
                bbox = _normalized_bbox(obj)
                labels.append(f"{obj['class_id']} " + " ".join(str(bbox[key]) for key in ("cx", "cy", "w", "h")))
                detections.append({"class_id": obj["class_id"], "score": 1.0, "bbox": bbox})
            (root / label_rel).write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            if split == "val":
                predictions.append({"schema_version": 2, "image": image_rel, "detections": detections})

            sample = {"image": image_rel, "label": label_rel, "split": split, "objects": objects}
            tile_index = len(samples)
            samples.append(sample)
            left = (tile_index % 2) * _WIDTH
            top = 40 + (tile_index // 2) * (_HEIGHT + 28)
            preview.paste(image, (left, top + 28))
            suffix = " (empty)" if not objects else ""
            preview_draw.text((left + 8, top + 6), filename + suffix, fill="black", font=font)
            for obj in objects:
                x0, y0, x1, y1 = obj["bbox_xyxy"]
                preview_draw.rectangle((left + x0, top + 28 + y0, left + x1 - 1, top + 28 + y1 - 1), outline="black", width=2)
                preview_draw.text((left + x0, top + 28 + y0 - 18), _NAMES[obj["class_id"]], fill="black", font=font)
            image.close()

    preview.save(root / "labeled_preview.png")
    preview.close()
    (root / "data.yaml").write_text(
        "# Paths are relative to this file; keep the directory together.\n"
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: circle\n  1: rectangle\n",
        encoding="utf-8",
    )
    _write_json(root / "predictions.json", {"schema_version": 1, "predictions": predictions})
    (root / "README.md").write_text(
        "# Labeled synthetic sample\n\n"
        "Eight 320x240 images: four train and four validation images. Classes are\n"
        "0 = circle and 1 = rectangle. The last validation image is intentionally\n"
        "empty and has an empty label file. All other images contain both classes.\n\n"
        "Images are drawn procedurally. No external images, downloads, training,\n"
        "or model inference are used. YOLO boxes come from the integer drawing\n"
        "geometry; right and bottom pixel bounds are exclusive.\n\n"
        "predictions.json contains validation predictions copied from the labels.\n"
        "A perfect evaluation score checks the data/evaluation workflow only.\n"
        "It is not evidence of model quality or real-world accuracy.\n\n"
        "Use data.yaml or this directory as the dataset; choose train or val.\n"
        "View labeled_preview.png to inspect all images and bounding boxes.\n"
        "Keep this directory together when copying it or reusing it elsewhere.\n"
        "Paths in the artifacts are relative to this directory.\n\n"
        "Other tools may resolve data.yaml paths differently. When required by\n"
        "that tool, set its path field to your dataset's absolute directory.\n"
        "External training-runtime compatibility is not tested by this generator.\n\n"
        "sample_manifest.json records the seed, recipe version, geometry, package\n"
        "versions, and file hashes. Keep it with the data when upgrading YOLOZU.\n"
        "The same recipe and seed preserve coordinates; PNG encoding and preview\n"
        "text may change with Pillow versions. Generation always requires a new\n"
        "directory and never overwrites an existing path.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "format_version": 1,
        "kind": "yolozu_labeled_sample",
        "seed": seed,
        "recipe": {
            "id": _RECIPE,
            "version": 1,
            "image_size": {"width": _WIDTH, "height": _HEIGHT},
            "classes": _NAMES,
            "class_colors_rgb": _COLORS,
            "background_rgb": _BACKGROUND,
            "bbox_format": "xyxy_pixels_right_bottom_exclusive",
            "coordinates": "sha256 recipe:seed:split:index:field, first 8 bytes big-endian modulo inclusive range",
            "empty_validation_image": "images/val/val_0004.png",
        },
        "counts": {"images": 8, "train_images": 4, "val_images": 4, "objects": 14, "train_objects": 8, "val_objects": 6, "classes": 2, "empty_images": 1},
        "paths": {
            "dataset": ".",
            "data_yaml": "data.yaml",
            "predictions": "predictions.json",
            "preview": "labeled_preview.png",
            "readme": "README.md",
            "train_images": "images/train",
            "train_labels": "labels/train",
            "train_classes": "labels/train/classes.json",
            "val_images": "images/val",
            "val_labels": "labels/val",
            "val_classes": "labels/val/classes.json",
        },
        "provenance": {
            "generator": "yolozu.demos.labeled_dataset.generate_labeled_sample",
            "generator_version": 1,
            "yolozu_version": yolozu_version,
            "pillow_version": pillow_version,
            "image_source": "procedural_geometry",
            "predictions_source": "ground_truth_labels",
            "model_inference": False,
        },
        "samples": samples,
        "sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
        },
    }
    manifest_path = root / "sample_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
