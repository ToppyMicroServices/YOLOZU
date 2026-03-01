#!/usr/bin/env python3
"""Prepare a deterministic domain-shift target dataset for TTT evidence.

This script generates a corrupted copy of a YOLO-style dataset split and writes
`domain_shift_recipe.json` containing a deterministic `export_settings` payload.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest

SUPPORTED_CORRUPTIONS = (
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "jpeg",
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare deterministic domain-shift target dataset for TTT.")
    p.add_argument("--dataset-root", required=True, help="Source YOLO-format dataset root.")
    p.add_argument("--split", default="val", help="Source split name (default: val).")
    p.add_argument("--out", required=True, help="Output dataset root.")
    p.add_argument(
        "--corruption",
        choices=SUPPORTED_CORRUPTIONS,
        default="gaussian_blur",
        help="Corruption type (default: gaussian_blur).",
    )
    p.add_argument("--severity", type=int, default=2, help="Corruption severity 1..5 (default: 2).")
    p.add_argument("--seed", type=int, default=0, help="Global deterministic seed (default: 0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on processed images.")
    p.add_argument(
        "--copy-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy labels/<split> and classes.json to output (default: true).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output root if it exists.")
    p.add_argument(
        "--recipe-out",
        default=None,
        help="Optional recipe JSON output path (default: <out>/domain_shift_recipe.json).",
    )
    return p


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}::{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _dataset_images_digest(paths: list[tuple[str, Path]]) -> str:
    hasher = hashlib.sha256()
    for rel_key, path in sorted(paths, key=lambda item: item[0]):
        rel = str(rel_key).encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(_sha256_file(path).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _apply_corruption(image: Image.Image, *, corruption: str, severity: int, seed: int) -> Image.Image:
    sev = max(1, min(5, int(severity)))
    rng = random.Random(int(seed))
    rgb = image.convert("RGB")

    if corruption == "gaussian_blur":
        radius = float(0.5 * sev + rng.uniform(0.0, 0.15))
        return rgb.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        factor = float(max(0.2, 1.0 - 0.11 * sev + rng.uniform(-0.02, 0.02)))
        return ImageEnhance.Brightness(rgb).enhance(factor)

    if corruption == "contrast":
        factor = float(max(0.2, 1.0 + 0.18 * sev + rng.uniform(-0.03, 0.03)))
        return ImageEnhance.Contrast(rgb).enhance(factor)

    if corruption == "gaussian_noise":
        sigma = float(6.0 * sev)
        width, height = rgb.size
        noise_bytes = bytearray(width * height)
        for idx in range(width * height):
            value = int(128.0 + rng.gauss(0.0, sigma))
            noise_bytes[idx] = max(0, min(255, value))
        noise_l = Image.frombytes("L", rgb.size, bytes(noise_bytes))
        noise_rgb = Image.merge("RGB", (noise_l, noise_l, noise_l))
        alpha = float(min(0.75, 0.12 * sev + rng.uniform(0.0, 0.02)))
        return Image.blend(rgb, noise_rgb, alpha=alpha)

    if corruption == "jpeg":
        quality = max(10, 95 - 15 * sev)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(quality), optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as tmp:
            return tmp.convert("RGB")

    raise ValueError(f"unsupported corruption: {corruption}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    severity = int(args.severity)
    if severity < 1 or severity > 5:
        raise SystemExit("--severity must be in [1, 5]")

    dataset_root = _resolve_path(str(args.dataset_root))
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    out_root = _resolve_path(str(args.out))
    if out_root.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {out_root} (use --force)")
        shutil.rmtree(out_root)

    manifest = build_manifest(str(dataset_root), split=str(args.split))
    split = str(manifest.get("split") or args.split)
    records = list(manifest.get("images") or [])
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]
    if not records:
        raise SystemExit(f"no images found for split '{split}' in {dataset_root}")

    out_images_dir = out_root / "images" / split
    out_labels_dir = out_root / "labels" / split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    written_images: list[tuple[str, Path]] = []
    for rec in records:
        image_raw = str(rec.get("image") or "")
        if not image_raw:
            continue
        src_image = Path(image_raw)
        if not src_image.is_absolute():
            src_image = dataset_root / src_image
        src_image = src_image.resolve()
        if not src_image.exists():
            raise SystemExit(f"missing source image: {src_image}")
        try:
            image_key = str(src_image.relative_to(dataset_root))
        except Exception:
            image_key = str(Path("images") / split / src_image.name)
        out_image = (out_root / image_key).resolve()
        if not out_image.is_relative_to(out_root.resolve()):
            raise SystemExit(f"refusing to write outside --out root: {out_image}")
        out_image.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_image) as image:
            img_seed = _stable_seed(int(args.seed), image_key)
            shifted = _apply_corruption(
                image,
                corruption=str(args.corruption),
                severity=int(severity),
                seed=int(img_seed),
            )
            shifted.save(out_image)
        written_images.append((image_key, out_image))

    source_labels_split = dataset_root / "labels" / split
    if bool(args.copy_labels):
        if not source_labels_split.exists():
            raise SystemExit(f"labels split not found: {source_labels_split}")
        shutil.copytree(source_labels_split, out_labels_dir, dirs_exist_ok=True)
    classes_json = source_labels_split / "classes.json"
    if classes_json.exists():
        shutil.copy2(classes_json, out_labels_dir / "classes.json")

    recipe_path = _resolve_path(str(args.recipe_out)) if args.recipe_out else (out_root / "domain_shift_recipe.json")
    recipe_path.parent.mkdir(parents=True, exist_ok=True)

    recipe_id = f"{args.corruption}_s{int(severity)}_seed{int(args.seed)}"
    images_digest = _dataset_images_digest(written_images)
    domain_shift_target = {
        "id": str(recipe_id),
        "split": str(split),
        "corruption": str(args.corruption),
        "severity": int(severity),
        "seed": int(args.seed),
        "source_dataset_root": str(dataset_root),
        "target_dataset_root": str(out_root),
        "image_count": int(len(written_images)),
        "images_sha256": str(images_digest),
        "deterministic": True,
    }
    recipe = {
        "kind": "yolozu_domain_shift_recipe",
        "version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "domain_shift_target": domain_shift_target,
        "export_settings": {"domain_shift_target": dict(domain_shift_target)},
    }
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    print(str(out_root))
    print(str(recipe_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
