#!/usr/bin/env python3
"""Prepare bounded detection, keypoint, and segmentation runtime fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a repository YOLO fixture into bounded COCO detection, "
            "COCO keypoint, and Cityscapes-style segmentation layouts."
        )
    )
    parser.add_argument("--source", required=True, help="YOLO dataset root.")
    parser.add_argument("--split", default="train", help="Source split (default: train).")
    parser.add_argument("--output", required=True, help="Fresh output root.")
    parser.add_argument("--max-images", type=int, default=6, help="Maximum images (default: 6).")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _class_names(source: Path, split: str) -> list[str]:
    path = source / "labels" / split / "classes.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(value) for value in payload]
        if isinstance(payload, dict):
            names = payload.get("names")
            if isinstance(names, list):
                return [str(value) for value in names]
            if isinstance(names, dict):
                return [
                    str(names[key])
                    for key in sorted(names, key=int)
                ]
            numeric = {
                str(key): value
                for key, value in payload.items()
                if str(key).isdigit()
            }
            if numeric:
                return [
                    str(numeric[key])
                    for key in sorted(numeric, key=int)
                ]
    max_class = -1
    for path in sorted((source / "labels" / split).glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                max_class = max(max_class, int(float(line.split()[0])))
    return [f"class_{index}" for index in range(max_class + 1)]


def _keypoints_from_bbox(x: float, y: float, width: float, height: float) -> list[float]:
    anchors = [
        (0.50, 0.10),
        (0.35, 0.18),
        (0.65, 0.18),
        (0.25, 0.30),
        (0.75, 0.30),
        (0.38, 0.35),
        (0.62, 0.35),
        (0.30, 0.50),
        (0.70, 0.50),
        (0.25, 0.68),
        (0.75, 0.68),
        (0.42, 0.60),
        (0.58, 0.60),
        (0.40, 0.78),
        (0.60, 0.78),
        (0.38, 0.95),
        (0.62, 0.95),
    ]
    values: list[float] = []
    for x_fraction, y_fraction in anchors:
        values.extend([x + x_fraction * width, y + y_fraction * height, 2.0])
    return values


def _write_coco(
    *,
    destination: Path,
    split: str,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    keypoints: bool,
) -> None:
    path = destination / "annotations" / (
        f"person_keypoints_{split}.json" if keypoints else f"instances_{split}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
        "info": {
            "description": "YOLOZU external-runtime availability fixture",
            "label_quality": (
                "bbox-derived keypoint anchors; runtime-only, not efficacy GT"
                if keypoints
                else "bbox labels inherited from the repository real-image fixture"
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    split = str(args.split)
    if not source.is_dir():
        raise SystemExit(f"source dataset not found: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace existing output: {output}")
    if args.max_images <= 0:
        raise SystemExit("--max-images must be positive")

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow is required to prepare runtime fixtures") from exc

    source_images = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        source_images.extend((source / "images" / split).glob(extension))
    source_images = sorted(set(source_images))[: int(args.max_images)]
    if not source_images:
        raise SystemExit(f"no source images found for split: {split}")
    names = _class_names(source, split)
    if not names:
        raise SystemExit("no source classes found")

    detection_root = output / "detection"
    keypoint_root = output / "keypoints"
    segmentation_root = output / "segmentation"
    images: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    poses: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, source_image in enumerate(source_images, start=1):
        with Image.open(source_image) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            filename = source_image.name
            for root in (detection_root, keypoint_root):
                for destination_split in ("train2017", "val2017"):
                    destination = root / destination_split / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image, destination)
            for destination_split in ("train2017", "val2017"):
                destination = detection_root / "images" / destination_split / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, destination)
            destination = keypoint_root / "images" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
            city_image_name = f"{source_image.stem}_leftImg8bit.png"
            city_label_name = f"{source_image.stem}_gtFine_labelTrainIds.png"
            city_image = segmentation_root / "images" / "train" / city_image_name
            city_label = segmentation_root / "labels" / "train" / city_label_name
            city_image.parent.mkdir(parents=True, exist_ok=True)
            city_label.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(city_image)
            mask = Image.new("L", (width, height), color=255)
            draw = ImageDraw.Draw(mask)

            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": width,
                    "height": height,
                }
            )
            label_path = source / "labels" / split / f"{source_image.stem}.txt"
            for root in (detection_root, keypoint_root):
                for destination_split in ("train2017", "val2017"):
                    destination_label = (
                        root / "labels" / destination_split / label_path.name
                    )
                    destination_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_path, destination_label)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                class_id = int(float(values[0]))
                center_x, center_y, box_width, box_height = map(float, values[1:5])
                box_width *= width
                box_height *= height
                x = center_x * width - box_width * 0.5
                y = center_y * height - box_height * 0.5
                bbox = [x, y, box_width, box_height]
                base = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
                detections.append(dict(base))
                poses.append(
                    {
                        **base,
                        "category_id": 1,
                        "keypoints": _keypoints_from_bbox(x, y, box_width, box_height),
                        "num_keypoints": 17,
                    }
                )
                draw.rectangle(
                    [
                        max(0, int(round(x))),
                        max(0, int(round(y))),
                        min(width - 1, int(round(x + box_width))),
                        min(height - 1, int(round(y + box_height))),
                    ],
                    fill=min(class_id, 18),
                )
                annotation_id += 1
            mask.save(city_label)

    source_classes = source / "labels" / split / "classes.json"
    for root in (detection_root, keypoint_root):
        for destination_split in ("train2017", "val2017"):
            metadata_root = root / "labels" / destination_split
            metadata_root.mkdir(parents=True, exist_ok=True)
            if source_classes.is_file():
                shutil.copy2(source_classes, metadata_root / "classes.json")
            (metadata_root / "classes.txt").write_text(
                "\n".join(names) + "\n",
                encoding="utf-8",
            )

    categories = [
        {"id": index + 1, "name": name, "supercategory": "object"}
        for index, name in enumerate(names)
    ]
    pose_categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [f"kp_{point + 1}" for point in range(17)],
            "skeleton": [[point, point + 1] for point in range(1, 17)],
        }
    ]
    _write_coco(
        destination=detection_root,
        split="train2017",
        images=images,
        annotations=detections,
        categories=categories,
        keypoints=False,
    )
    _write_coco(
        destination=detection_root,
        split="val2017",
        images=images,
        annotations=detections,
        categories=categories,
        keypoints=False,
    )
    _write_coco(
        destination=keypoint_root,
        split="train2017",
        images=images,
        annotations=poses,
        categories=pose_categories,
        keypoints=True,
    )
    _write_coco(
        destination=keypoint_root,
        split="val2017",
        images=images,
        annotations=poses,
        categories=pose_categories,
        keypoints=True,
    )
    report = {
        "schema_version": 1,
        "kind": "external_runtime_smoke_datasets",
        "source": str(source),
        "source_split": split,
        "images": len(images),
        "instances": len(detections),
        "classes": len(names),
        "ground_truth": {
            "detection": "repository fixture bbox labels",
            "keypoints": "bbox-derived anchors for runtime availability only",
            "segmentation": "bbox-derived class masks for runtime availability only",
        },
        "outputs": {
            "detection": str(detection_root),
            "keypoints": str(keypoint_root),
            "segmentation": str(segmentation_root),
        },
    }
    report_path = output / "preparation_report.json"
    report["tree_sha256"] = hashlib.sha256(
        "\n".join(
            [
                f"detection {_tree_sha256(detection_root)}",
                f"keypoints {_tree_sha256(keypoint_root)}",
                f"segmentation {_tree_sha256(segmentation_root)}",
            ]
        ).encode("utf-8")
    ).hexdigest()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
