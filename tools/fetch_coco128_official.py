#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.coco_convert import convert_coco_instances_to_yolo_labels


DEFAULT_ANN_ZIP_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DEFAULT_IMG_BASE = "https://images.cocodataset.org"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch a tiny COCO subset from official hosting and write YOLO-format labels.")
    p.add_argument("--out", default=str(repo_root / "data" / "coco128"), help="Output dataset root (default: data/coco128).")
    p.add_argument(
        "--n",
        type=int,
        default=128,
        help="Number of images to fetch (deterministic: first N images by id from COCO instances JSON).",
    )
    p.add_argument("--split", choices=("val2017", "train2017"), default="val2017", help="COCO split to sample from.")
    p.add_argument(
        "--out-split",
        default="train2017",
        help="Output split folder name under out/images/ and out/labels (default: train2017 to match tests).",
    )
    p.add_argument(
        "--coco-root",
        default=None,
        help="Optional local COCO root (expects images/<split>/ and annotations/instances_<split>.json). If set, avoids downloads.",
    )
    p.add_argument("--annotations-zip-url", default=DEFAULT_ANN_ZIP_URL, help="URL to annotations_trainval2017.zip")
    p.add_argument("--image-base-url", default=DEFAULT_IMG_BASE, help="Base URL for images.cocodataset.org")
    p.add_argument("--keep-tmp", action="store_true", help="Keep temporary files (debug).")
    return p.parse_args(argv)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def _load_instances_from_local(coco_root: Path, split: str) -> dict:
    instances_path = coco_root / "annotations" / f"instances_{split}.json"
    if not instances_path.exists():
        raise SystemExit(f"instances JSON not found: {instances_path}")
    return json.loads(instances_path.read_text(encoding="utf-8"))


def _ensure_instances_json(tmp_dir: Path, *, split: str, ann_zip_url: str) -> dict:
    ann_zip_path = tmp_dir / "annotations_trainval2017.zip"
    if not ann_zip_path.exists():
        _download(ann_zip_url, ann_zip_path)

    with zipfile.ZipFile(ann_zip_path) as zf:
        member = f"annotations/instances_{split}.json"
        try:
            with zf.open(member) as f:
                return json.loads(f.read().decode("utf-8"))
        except KeyError as e:
            raise SystemExit(f"{member} not found in {ann_zip_path}") from e


def _select_subset(instances: dict, *, n: int) -> tuple[list[dict], set[int]]:
    images = instances.get("images") or []
    if not isinstance(images, list) or not images:
        raise SystemExit("COCO instances JSON missing images[]")
    images_sorted = sorted(images, key=lambda x: int(x["id"]))
    subset = images_sorted[: max(0, int(n))]
    image_ids = {int(img["id"]) for img in subset}
    return subset, image_ids


def _filter_annotations(instances: dict, image_ids: set[int]) -> list[dict]:
    annotations = instances.get("annotations") or []
    if not isinstance(annotations, list):
        raise SystemExit("COCO instances JSON missing annotations[]")
    return [ann for ann in annotations if int(ann.get("image_id", -1)) in image_ids]


def _download_images(*, images: list[dict], split: str, img_base_url: str, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        file_name = str(img.get("file_name") or "")
        if not file_name:
            continue
        dst = dst_dir / Path(file_name).name
        if dst.exists():
            continue
        url = f"{img_base_url}/{split}/{file_name}"
        _download(url, dst)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    out_root = Path(args.out)
    out_split = str(args.out_split)
    images_out = out_root / "images" / out_split
    labels_out = out_root / "labels" / out_split

    tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    tmp_dir: Path | None = None
    try:
        if args.coco_root:
            coco_root = Path(args.coco_root)
            instances = _load_instances_from_local(coco_root, args.split)
            images_src = coco_root / "images" / args.split
            if not images_src.exists():
                fallback = coco_root / args.split
                if fallback.exists():
                    images_src = fallback
                else:
                    raise SystemExit(f"images not found for split={args.split} under {coco_root}")
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

            images_subset, image_ids = _select_subset(instances, n=args.n)
            subset_instances = {
                "images": images_subset,
                "annotations": _filter_annotations(instances, image_ids),
                "categories": instances.get("categories") or [],
            }
            convert_coco_instances_to_yolo_labels(
                instances_json=subset_instances,
                images_dir=images_out,
                labels_dir=labels_out,
                include_crowd=False,
            )

            # Copy referenced images into the output dataset.
            for img in images_subset:
                file_name = str(img.get("file_name") or "")
                if not file_name:
                    continue
                src = images_src / file_name
                dst = images_out / Path(file_name).name
                if dst.exists():
                    continue
                if src.exists():
                    shutil.copy2(src, dst)
        else:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="yolozu_coco128_")
            tmp_dir = Path(tmp_dir_obj.name)

            instances = _ensure_instances_json(tmp_dir, split=args.split, ann_zip_url=str(args.annotations_zip_url))
            images_subset, image_ids = _select_subset(instances, n=args.n)

            subset_instances = {
                "images": images_subset,
                "annotations": _filter_annotations(instances, image_ids),
                "categories": instances.get("categories") or [],
            }

            # Download images and write YOLO labels.
            _download_images(
                images=images_subset,
                split=args.split,
                img_base_url=str(args.image_base_url).rstrip("/"),
                dst_dir=images_out,
            )
            convert_coco_instances_to_yolo_labels(
                instances_json=subset_instances,
                images_dir=images_out,
                labels_dir=labels_out,
                include_crowd=False,
            )
    finally:
        if tmp_dir_obj and args.keep_tmp:
            print(f"Kept temp dir: {tmp_dir}", file=sys.stderr)
            tmp_dir_obj = None

    # Minimal sanity checks (align with tests).
    if not images_out.is_dir() or not labels_out.is_dir():
        raise SystemExit("failed to create expected coco128 layout")

    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
