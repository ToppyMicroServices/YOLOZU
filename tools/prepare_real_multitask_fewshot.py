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
import logging
import math
import subprocess
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATASET_LICENSE_WARNING = (
    "WARNING: datasets/weights are separate artifacts with their own licenses. "
    "YOLOZU repository Apache-2.0 does not automatically apply to dataset terms."
)


def _manual_download_message(*, instances_json: Path, images_dir: Path, total_images: int, seed: int, timeout: float) -> str:
    coco_root = instances_json.parent.parent
    split = str(images_dir.name)
    cmd = (
        "python3 scripts/download_coco_instances_tiny.py "
        f"--out-root {coco_root} --split {split} --num-images {int(total_images)} "
        f"--seed {int(seed)} --timeout {float(timeout):.1f} --force"
    )
    return (
        f"{_DATASET_LICENSE_WARNING}\n"
        "COCO input files are missing and automatic download is disabled by default.\n"
        "Please download data manually, review license terms, then rerun this tool.\n"
        f"Suggested command:\n  {cmd}"
    )


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
    p.add_argument(
        "--download-if-missing",
        action="store_true",
        help=(
            "If COCO inputs are missing, request tiny subset download. "
            "Requires --allow-auto-download and --accept-dataset-license."
        ),
    )
    p.add_argument(
        "--allow-auto-download",
        action="store_true",
        help="Allow this tool to invoke tiny COCO downloader automatically.",
    )
    p.add_argument(
        "--accept-dataset-license",
        action="store_true",
        help="Acknowledge that dataset/weights licenses are separate from repo license.",
    )
    p.add_argument(
        "--download-num-images",
        type=int,
        default=None,
        help="Override tiny COCO download image count (default: train-images + val-images).",
    )
    p.add_argument(
        "--download-seed",
        type=int,
        default=0,
        help="Seed used by tiny COCO downloader (default: 0).",
    )
    p.add_argument(
        "--download-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds for tiny COCO downloader (default: 60).",
    )
    p.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail if model-inference-generated labels are detected in provenance.",
    )
    p.add_argument(
        "--strict-realism",
        action="store_true",
        help="Fail if heuristic/scaffold labels are present (bbox-derived keypoints/depth/pose).",
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


def _download_coco_tiny_if_needed(
    *,
    instances_json: Path,
    images_dir: Path,
    total_images: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    downloader = repo_root / "scripts" / "download_coco_instances_tiny.py"
    if not downloader.exists():
        raise SystemExit(f"downloader not found: {downloader}")

    split = str(images_dir.name)
    if split != "val2017":
        raise SystemExit(
            "automatic download currently supports val2017 only. "
            "Provide existing --instances-json/--images-dir or switch to val2017 layout."
        )
    if instances_json.parent.name != "annotations":
        raise SystemExit(
            "automatic download expects instances path under <coco_root>/annotations/. "
            f"got: {instances_json}"
        )
    coco_root = instances_json.parent.parent
    want = max(1, int(total_images))

    cmd = [
        sys.executable,
        str(downloader),
        "--out-root",
        str(coco_root),
        "--split",
        split,
        "--num-images",
        str(want),
        "--seed",
        str(int(seed)),
        "--timeout",
        str(float(timeout)),
        "--force",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "tiny COCO download failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return {
        "enabled": True,
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "").splitlines()[-5:],
    }


def _prepare_dataset(
    *,
    instances_json: Path,
    images_dir: Path,
    out_root: Path,
    train_images: int,
    val_images: int,
    num_keypoints: int,
    download_if_missing: bool,
    allow_auto_download: bool,
    accept_dataset_license: bool,
    download_num_images: int | None,
    download_seed: int,
    download_timeout: float,
    strict_provenance: bool,
    strict_realism: bool,
    force: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    download_info = None
    inputs_missing = bool(not instances_json.exists() or not images_dir.exists())
    total = (
        int(download_num_images)
        if download_num_images is not None
        else (int(train_images) + int(val_images))
    )
    if inputs_missing:
        if bool(download_if_missing) and bool(allow_auto_download):
            if not bool(accept_dataset_license):
                raise SystemExit(
                    "automatic download requires --accept-dataset-license to confirm dataset terms."
                )
            warnings.append(_DATASET_LICENSE_WARNING)
            download_info = _download_coco_tiny_if_needed(
                instances_json=instances_json,
                images_dir=images_dir,
                total_images=int(total),
                seed=int(download_seed),
                timeout=float(download_timeout),
            )
        else:
            raise SystemExit(
                _manual_download_message(
                    instances_json=instances_json,
                    images_dir=images_dir,
                    total_images=int(total),
                    seed=int(download_seed),
                    timeout=float(download_timeout),
                )
            )
    if not instances_json.exists():
        raise SystemExit(f"instances json not found: {instances_json}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    np, pil = _ensure_deps()
    _Image, image_draw = pil

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
    segmentation_checks = {
        "mask_files": 0,
        "mask_non_empty": 0,
    }
    label_provenance = {
        "bbox": "coco_instances_gt",
        "segmentation": "coco_polygon_gt",
        "keypoints": "bbox_derived_anchors",
        "depth": "bbox_scale_heuristic",
        "pose6d": "bbox_depth_intrinsics_heuristic",
        "model_inference_used": False,
    }

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
        mask_path = out_root / mask_rel
        mask_img.save(mask_path)
        segmentation_checks["mask_files"] += 1

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
        try:
            if bool((np.asarray(mask_img, dtype=np.int64) > 0).any()):
                segmentation_checks["mask_non_empty"] += 1
        except (TypeError, ValueError) as exc:
            logger.debug("failed to inspect generated mask occupancy: %s", exc)

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
            "label_provenance": label_provenance,
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
        "label_provenance": label_provenance,
        "splits": {
            "train": {"images": "images/train", "labels": "labels/train"},
            "val": {"images": "images/val", "labels": "labels/val"},
        },
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
        "warnings": list(warnings),
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_doc, indent=2, sort_keys=True), encoding="utf-8")

    heuristic_fields = [
        key
        for key, value in label_provenance.items()
        if isinstance(value, str) and ("heuristic" in value or "derived" in value)
    ]
    if bool(heuristic_fields):
        warnings.append(
            "heuristic/scaffold labels detected for: " + ", ".join(sorted(heuristic_fields))
        )
    if bool(strict_provenance) and bool(label_provenance.get("model_inference_used")):
        raise SystemExit("strict provenance violation: model_inference_used must be false")
    if bool(strict_realism) and bool(heuristic_fields):
        raise SystemExit(
            "strict realism violation: heuristic/scaffold labels are present. "
            "Provide manually annotated keypoints/depth/pose to continue."
        )

    return {
        "dataset_root": str(out_root),
        "instances_json": str(instances_json),
        "images_dir": str(images_dir),
        "download": download_info,
        "label_provenance": label_provenance,
        "warnings": warnings,
        "checks": {
            "segmentation_masks_non_empty": bool(
                int(segmentation_checks["mask_non_empty"]) == int(segmentation_checks["mask_files"])
            ),
            "strict_provenance": bool(strict_provenance),
            "strict_realism": bool(strict_realism),
            "heuristic_fields": heuristic_fields,
        },
        "segmentation_checks": {
            "mask_files": int(segmentation_checks["mask_files"]),
            "mask_non_empty": int(segmentation_checks["mask_non_empty"]),
        },
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
    print(_DATASET_LICENSE_WARNING, file=sys.stderr)

    summary = _prepare_dataset(
        instances_json=Path(str(args.instances_json)).expanduser(),
        images_dir=Path(str(args.images_dir)).expanduser(),
        out_root=Path(str(args.out)).expanduser(),
        train_images=int(args.train_images),
        val_images=int(args.val_images),
        num_keypoints=int(args.num_keypoints),
        download_if_missing=bool(args.download_if_missing),
        allow_auto_download=bool(args.allow_auto_download),
        accept_dataset_license=bool(args.accept_dataset_license),
        download_num_images=(int(args.download_num_images) if args.download_num_images is not None else None),
        download_seed=int(args.download_seed),
        download_timeout=float(args.download_timeout),
        strict_provenance=bool(args.strict_provenance),
        strict_realism=bool(args.strict_realism),
        force=bool(args.force),
    )

    out_summary = Path(str(args.out)).expanduser() / "prepare_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
