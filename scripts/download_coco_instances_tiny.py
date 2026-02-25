#!/usr/bin/env python3
"""Download a tiny COCO instances (polygon) subset for `yolozu demo`.

This script:
- Downloads COCO annotations zip (trainval2017) if needed
- Extracts `instances_val2017.json`
- Selects N images that have polygon segmentations (non-crowd)
- Downloads just those image files from the COCO image server
- Writes a tiny subset JSON to `<out_root>/annotations/instances_val2017.json`

Default output layout matches what `yolozu demo` auto-detects:
- data/coco/annotations/instances_val2017.json
- data/coco/images/val2017/<file>.jpg

Notes:
- COCO images have per-image licenses; this script avoids bundling images in git.
- Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


COCO_ANN_ZIP_HTTPS = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ANN_ZIP_HTTP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGE_HTTP_PREFIX = "http://images.cocodataset.org/val2017/"


def _urlretrieve(url: str, dst: Path, *, timeout: float = 60.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "yolozu-coco-tiny/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dst.write_bytes(resp.read())


def _download_annotations_zip(zip_path: Path, *, timeout: float) -> None:
    # Prefer HTTPS, but some environments have broken TLS/certs.
    try:
        _urlretrieve(COCO_ANN_ZIP_HTTPS, zip_path, timeout=timeout)
        return
    except Exception:
        pass

    try:
        _urlretrieve(COCO_ANN_ZIP_HTTP, zip_path, timeout=timeout)
        return
    except Exception as exc:
        raise RuntimeError(f"failed to download COCO annotations zip from {COCO_ANN_ZIP_HTTPS} or {COCO_ANN_ZIP_HTTP}: {exc}")


def _has_polygon_segmentation(ann: dict[str, Any]) -> bool:
    if int(ann.get("iscrowd", 0)) != 0:
        return False
    seg = ann.get("segmentation")
    # COCO polygon segmentations are list[list[float]]
    if not isinstance(seg, list) or not seg:
        return False
    # RLE is a dict; reject.
    if isinstance(seg, dict):
        return False
    # Validate first polygon looks like a flat coordinate list.
    poly0 = seg[0]
    return isinstance(poly0, list) and len(poly0) >= 6


def _select_image_ids(
    *,
    instances_full: dict[str, Any],
    num_images: int,
    seed: int,
) -> list[int]:
    anns = instances_full.get("annotations")
    if not isinstance(anns, list):
        raise ValueError("instances JSON missing 'annotations' list")

    by_image: dict[int, int] = {}
    for a in anns:
        if not isinstance(a, dict):
            continue
        if not _has_polygon_segmentation(a):
            continue
        image_id = a.get("image_id")
        if not isinstance(image_id, int):
            continue
        by_image[image_id] = by_image.get(image_id, 0) + 1

    candidates = list(by_image.keys())
    if not candidates:
        raise RuntimeError("no polygon segmentations found in instances JSON")

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    return candidates[: max(1, int(num_images))]


def _subset_instances(
    *,
    instances_full: dict[str, Any],
    selected_image_ids: set[int],
) -> dict[str, Any]:
    images = instances_full.get("images")
    anns = instances_full.get("annotations")
    cats = instances_full.get("categories")

    if not isinstance(images, list) or not isinstance(anns, list) or not isinstance(cats, list):
        raise ValueError("instances JSON missing 'images'/'annotations'/'categories'")

    images_out: list[dict[str, Any]] = []
    images_by_id: dict[int, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        iid = im.get("id")
        if isinstance(iid, int):
            images_by_id[iid] = im

    anns_out: list[dict[str, Any]] = []
    used_cat_ids: set[int] = set()

    for a in anns:
        if not isinstance(a, dict):
            continue
        image_id = a.get("image_id")
        if not isinstance(image_id, int) or image_id not in selected_image_ids:
            continue
        if not _has_polygon_segmentation(a):
            continue
        cid = a.get("category_id")
        if isinstance(cid, int):
            used_cat_ids.add(cid)
        anns_out.append(a)

    for iid in selected_image_ids:
        im = images_by_id.get(iid)
        if im is None:
            continue
        images_out.append(im)

    cats_out = [c for c in cats if isinstance(c, dict) and isinstance(c.get("id"), int) and c["id"] in used_cat_ids]

    return {
        "images": images_out,
        "annotations": anns_out,
        "categories": cats_out,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="download_coco_instances_tiny.py",
        description="Download a tiny COCO instances polygon subset for yolozu demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-root", default="data/coco", help="Output COCO root (annotations/ + images/).")
    p.add_argument("--split", default="val2017", help="COCO split name.")
    p.add_argument("--num-images", type=int, default=2, help="How many images to download.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for selecting images.")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    p.add_argument("--force", action="store_true", help="Overwrite existing subset JSON and images.")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded annotations zip.")
    args = p.parse_args(argv)

    out_root = Path(str(args.out_root))
    split = str(args.split)

    ann_dir = out_root / "annotations"
    img_dir = out_root / "images" / split
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    subset_json_path = ann_dir / f"instances_{split}.json"
    # yolozu demo auto-detects instances_val2017.json specifically.
    if split == "val2017":
        subset_json_path = ann_dir / "instances_val2017.json"

    full_json_path = ann_dir / f"instances_{split}_full.json"
    if split == "val2017":
        full_json_path = ann_dir / "instances_val2017_full.json"

    if subset_json_path.exists() and not args.force:
        print(f"subset JSON already exists (use --force to overwrite): {subset_json_path}")
        return 0

    zip_path = ann_dir / "annotations_trainval2017.zip"

    if not full_json_path.exists() or args.force:
        if not zip_path.exists() or args.force:
            print(f"downloading annotations zip -> {zip_path}")
            _download_annotations_zip(zip_path, timeout=float(args.timeout))
            print("downloaded annotations zip")

        print("extracting instances JSON...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            member = f"annotations/instances_{split}.json"
            try:
                raw = zf.read(member)
            except KeyError as exc:
                raise SystemExit(f"missing member in zip: {member}") from exc
            full_json_path.write_bytes(raw)
        print(f"wrote full instances JSON: {full_json_path}")

        if not args.keep_zip:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass

    instances_full = json.loads(full_json_path.read_text(encoding="utf-8"))
    selected = _select_image_ids(instances_full=instances_full, num_images=int(args.num_images), seed=int(args.seed))
    selected_set = set(selected)
    subset = _subset_instances(instances_full=instances_full, selected_image_ids=selected_set)

    # Download images
    images = subset.get("images")
    if not isinstance(images, list):
        raise SystemExit("subset missing images")

    downloaded = 0
    for im in images:
        if not isinstance(im, dict):
            continue
        file_name = im.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue
        dst = img_dir / file_name
        if dst.exists() and not args.force:
            continue
        url = COCO_VAL_IMAGE_HTTP_PREFIX + file_name
        print(f"downloading image: {url} -> {dst}")
        try:
            _urlretrieve(url, dst, timeout=float(args.timeout))
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"failed to download image: {url} ({exc})") from exc
        downloaded += 1

    subset_json_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("---")
    print(f"subset instances JSON: {subset_json_path}")
    print(f"images dir:           {img_dir}")
    print(f"images kept:          {len(images)}")
    print(f"images downloaded:    {downloaded}")
    print("next:")
    print("  python -m yolozu demo")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
