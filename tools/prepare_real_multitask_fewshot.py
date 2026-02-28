#!/usr/bin/env python3
"""Prepare a real-image multitask few-shot dataset from COCO instances.

This tool builds a compact YOLO-style dataset root with:
- bbox labels (YOLO txt)
- optional synthetic keypoints derived from GT boxes
- mask/depth/pose sidecars for multitask finetune smoke checks

Input images and base annotations are real (COCO val images).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare real-image multitask few-shot dataset from COCO instances.")
    p.add_argument(
        "--instances-json",
        default="data/coco/annotations/instances_val2017.json",
        help="Path to COCO instances_*.json (default: data/coco/annotations/instances_val2017.json).",
    )
    p.add_argument(
        "--images-dir",
        default="data/coco/images/val2017",
        help="COCO images directory for the annotation split (default: data/coco/images/val2017).",
    )
    p.add_argument(
        "--out",
        default="data/real_multitask_fewshot",
        help="Output dataset root (default: data/real_multitask_fewshot).",
    )
    p.add_argument("--train-images", type=int, default=6, help="Number of images for train split (default: 6).")
    p.add_argument("--val-images", type=int, default=2, help="Number of images for val split (default: 2).")
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=4,
        help="Number of synthetic bbox-derived keypoints per instance (default: 4).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    return p


def _ensure_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise SystemExit("NumPy is required: pip install numpy") from exc
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise SystemExit("Pillow is required: pip install Pillow") from exc
    return np, (Image, ImageDraw)


def _rot_yaw(yaw_rad: float) -> list[list[float]]:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _bbox_to_keypoints(cx: float, cy: float, w: float, h: float, *, count: int) -> list[tuple[float, float, int]]:
    # Deterministic "manual-style" anchors around each bbox.
    points = [
        (cx - w * 0.5, cy - h * 0.5, 2),
        (cx + w * 0.5, cy - h * 0.5, 2),
        (cx + w * 0.5, cy + h * 0.5, 2),
        (cx - w * 0.5, cy + h * 0.5, 2),
        (cx, cy - h * 0.5, 2),
        (cx + w * 0.5, cy, 2),
        (cx, cy + h * 0.5, 2),
        (cx - w * 0.5, cy, 2),
    ]
    out = []
    for x, y, v in points[: max(1, int(count))]:
        out.append((max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y))), int(v)))
    return out


def _draw_polygons_mask(image_w: int, image_h: int, anns: list[dict[str, Any]], *, image_draw: Any) -> Any:
    from PIL import Image

    mask_img = Image.new("I", (int(image_w), int(image_h)), color=0)
    draw = image_draw.Draw(mask_img)

    inst_id = 1
    for ann in anns:
        seg = ann.get("segmentation")
        if not isinstance(seg, list):
            continue
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
                continue
            pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=int(inst_id), outline=int(inst_id))
        inst_id += 1
    return mask_img


def _prepare_dataset(
    *,
    instances_json: Path,
    images_dir: Path,
    out_root: Path,
    train_images: int,
    val_images: int,
    num_keypoints: int,
    force: bool,
) -> dict[str, Any]:
    np, pil = _ensure_deps()
    _Image, image_draw = pil

    if not instances_json.exists():
        raise SystemExit(f"instances json not found: {instances_json}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    if out_root.exists():
        if not force:
            raise SystemExit(f"output already exists: {out_root} (use --force)")
        shutil.rmtree(out_root)

    doc = json.loads(instances_json.read_text(encoding="utf-8"))
    images = list(doc.get("images") or [])
    annotations = list(doc.get("annotations") or [])
    categories = list(doc.get("categories") or [])
    if not images:
        raise SystemExit("instances json has no images")

    want = int(train_images) + int(val_images)
    if want <= 0:
        raise SystemExit("train-images + val-images must be > 0")
    if len(images) < want:
        raise SystemExit(f"not enough images in instances json: need {want}, found {len(images)}")

    # Stable class mapping to contiguous ids.
    sorted_cats = sorted((c for c in categories if isinstance(c, dict) and "id" in c), key=lambda c: int(c["id"]))
    cat_to_idx = {int(cat["id"]): idx for idx, cat in enumerate(sorted_cats)}
    class_names = [str(cat.get("name", f"class_{idx}")) for idx, cat in enumerate(sorted_cats)]

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if int(ann.get("iscrowd", 0) or 0) == 1:
            continue
        try:
            image_id = int(ann.get("image_id"))
            cat_id = int(ann.get("category_id"))
        except Exception:
            continue
        if cat_id not in cat_to_idx:
            continue
        anns_by_image[image_id].append(ann)

    selected = sorted(images, key=lambda x: int(x.get("id", 0)))[:want]
    split_lookup: dict[int, str] = {}
    for idx, image in enumerate(selected):
        image_id = int(image.get("id"))
        split_lookup[image_id] = "train" if idx < int(train_images) else "val"

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "depth" / split).mkdir(parents=True, exist_ok=True)

    manifest_counts = {"train": 0, "val": 0}
    instance_counts = {"train": 0, "val": 0}

    for image in selected:
        try:
            image_id = int(image.get("id"))
            file_name = str(image.get("file_name"))
            width = int(image.get("width"))
            height = int(image.get("height"))
        except Exception:
            continue

        split = split_lookup.get(image_id)
        if split not in ("train", "val"):
            continue

        src_image = images_dir / file_name
        if not src_image.exists():
            raise SystemExit(f"missing source image: {src_image}")

        dst_image = out_root / "images" / split / Path(file_name).name
        shutil.copy2(src_image, dst_image)

        anns = anns_by_image.get(image_id, [])
        anns = sorted(anns, key=lambda x: int(x.get("id", 0)))

        # Build per-image instance mask from polygons (if available).
        mask_img = _draw_polygons_mask(width, height, anns, image_draw=image_draw)
        mask_rel = Path("masks") / split / f"{dst_image.stem}_inst.png"
        (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(out_root / mask_rel)

        depth_map = np.zeros((height, width), dtype=np.float32)

        lines: list[str] = []
        r_list: list[list[list[float]]] = []
        t_list: list[list[float]] = []

        fx = max(1.0, 0.9 * float(width))
        fy = max(1.0, 0.9 * float(height))
        cx0 = float(width) * 0.5
        cy0 = float(height) * 0.5
        k_gt = [[fx, 0.0, cx0], [0.0, fy, cy0], [0.0, 0.0, 1.0]]

        for ann in anns:
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 1.0 or h <= 1.0:
                continue

            cat_id = int(ann.get("category_id"))
            class_id = int(cat_to_idx.get(cat_id, -1))
            if class_id < 0:
                continue

            cx = (x + 0.5 * w) / float(width)
            cy = (y + 0.5 * h) / float(height)
            wn = w / float(width)
            hn = h / float(height)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            wn = max(1e-6, min(1.0, wn))
            hn = max(1e-6, min(1.0, hn))

            keypoints = _bbox_to_keypoints(cx, cy, wn, hn, count=int(num_keypoints))
            parts = [f"{class_id}", f"{cx:.6f}", f"{cy:.6f}", f"{wn:.6f}", f"{hn:.6f}"]
            for kx, ky, kv in keypoints:
                parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{int(kv)}"])
            lines.append(" ".join(parts))

            # Pseudo depth target from bbox scale (annotation-driven; no model inference).
            scale = max(1e-6, math.sqrt(wn * hn))
            z = float(max(0.4, min(6.0, 0.8 / scale)))

            x0 = int(max(0, min(width - 1, math.floor(x))))
            y0 = int(max(0, min(height - 1, math.floor(y))))
            x1 = int(max(x0 + 1, min(width, math.ceil(x + w))))
            y1 = int(max(y0 + 1, min(height, math.ceil(y + h))))
            depth_map[y0:y1, x0:x1] = z

            u = cx * float(width)
            v = cy * float(height)
            tx = (u - cx0) / fx * z
            ty = (v - cy0) / fy * z
            t_list.append([float(tx), float(ty), float(z)])

            yaw = math.atan2((wn - hn), max(1e-6, wn + hn))
            r_list.append(_rot_yaw(yaw))

        label_path = out_root / "labels" / split / f"{dst_image.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        depth_rel = Path("depth") / split / f"{dst_image.stem}_depth.npy"
        np.save(out_root / depth_rel, depth_map)

        meta = {
            "mask_path": str(mask_rel),
            "mask_format": "instance",
            "mask_instances": True,
            "depth_path": str(depth_rel),
            "depth_unit": "relative",
            "K_gt": k_gt,
            "R_gt": r_list,
            "t_gt": t_list,
            "offsets_gt": [[0.0, 0.0] for _ in t_list],
        }
        meta_path = out_root / "labels" / split / f"{dst_image.stem}.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        manifest_counts[split] += 1
        instance_counts[split] += int(len(t_list))

    keypoint_names = [f"kp_{i+1}" for i in range(max(1, int(num_keypoints)))]
    skeleton = [[i, i + 1] for i in range(1, max(1, int(num_keypoints)))]

    classes_doc = {
        "names": class_names,
        "num_classes": int(len(class_names)),
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
    }
    for split in ("train", "val"):
        classes_path = out_root / "labels" / split / "classes.json"
        classes_path.write_text(json.dumps(classes_doc, indent=2, sort_keys=True), encoding="utf-8")

    dataset_doc = {
        "format": "yolo",
        "task": "multi",
        "name": "real_multitask_fewshot",
        "root": str(out_root),
        "splits": {
            "train": {"images": "images/train", "labels": "labels/train"},
            "val": {"images": "images/val", "labels": "labels/val"},
        },
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_doc, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_root": str(out_root),
        "instances_json": str(instances_json),
        "images_dir": str(images_dir),
        "counts": {
            "train_images": int(manifest_counts["train"]),
            "val_images": int(manifest_counts["val"]),
            "train_instances": int(instance_counts["train"]),
            "val_instances": int(instance_counts["val"]),
            "classes": int(len(class_names)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    summary = _prepare_dataset(
        instances_json=Path(str(args.instances_json)).expanduser(),
        images_dir=Path(str(args.images_dir)).expanduser(),
        out_root=Path(str(args.out)).expanduser(),
        train_images=int(args.train_images),
        val_images=int(args.val_images),
        num_keypoints=int(args.num_keypoints),
        force=bool(args.force),
    )

    out_summary = Path(str(args.out)).expanduser() / "prepare_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
