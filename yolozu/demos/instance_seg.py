from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _require_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo instance-seg requires numpy and Pillow") from exc
    return np, (Image, ImageDraw)


def _iter_coco128_pairs(*, coco128_root: Path) -> list[tuple[Path, Path]]:
    images_base = coco128_root / "images"
    labels_base = coco128_root / "labels"
    if not images_base.exists() or not labels_base.exists():
        raise FileNotFoundError(f"coco128 not found under: {coco128_root}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_base.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        rel = img.relative_to(images_base)
        lab = labels_base / rel.with_suffix(".txt")
        if lab.exists():
            pairs.append((img, lab))
    if not pairs:
        raise FileNotFoundError(f"no image/label pairs found under: {coco128_root}")
    return pairs


def _yolo_bbox_to_xyxy(*, w: int, h: int, xc: float, yc: float, bw: float, bh: float) -> tuple[int, int, int, int]:
    x0 = int((float(xc) - float(bw) / 2.0) * float(w))
    y0 = int((float(yc) - float(bh) / 2.0) * float(h))
    x1 = int((float(xc) + float(bw) / 2.0) * float(w))
    y1 = int((float(yc) + float(bh) / 2.0) * float(h))
    x0 = max(0, min(int(w) - 1, x0))
    y0 = max(0, min(int(h) - 1, y0))
    x1 = max(0, min(int(w), x1))
    y1 = max(0, min(int(h), y1))
    if x1 <= x0:
        x1 = min(int(w), x0 + 1)
    if y1 <= y0:
        y1 = min(int(h), y0 + 1)
    return x0, y0, x1, y1


def _draw_mask_circle(*, size: int, cx: int, cy: int, r: int) -> Any:
    np, (Image, ImageDraw) = _require_deps()
    img = Image.new("L", (int(size), int(size)), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.array(img) != 0


def _draw_mask_rect(*, size: int, x0: int, y0: int, x1: int, y1: int) -> Any:
    np, (Image, ImageDraw) = _require_deps()
    img = Image.new("L", (int(size), int(size)), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    return np.array(img) != 0


def _overlay_masks(
    *,
    base_rgb_path: Path,
    output_path: Path,
    gt_masks: list[Any],
    pred_masks: list[Any],
    alpha: float = 0.45,
) -> None:
    np, (Image, ImageDraw) = _require_deps()

    base = Image.open(base_rgb_path).convert("RGBA")
    w, h = base.size

    def _mask_to_rgba(mask_bool: Any, *, rgb: tuple[int, int, int]) -> Any:
        m = np.asarray(mask_bool, dtype=bool)
        if m.shape != (h, w):
            # Demo should always be consistent; keep this safe.
            m = np.resize(m, (h, w)).astype(bool)
        arr = np.zeros((h, w, 4), dtype="uint8")
        arr[..., 0] = int(rgb[0])
        arr[..., 1] = int(rgb[1])
        arr[..., 2] = int(rgb[2])
        arr[..., 3] = (m.astype("uint8") * int(max(0, min(1.0, float(alpha))) * 255))
        return Image.fromarray(arr, mode="RGBA")

    # Green = GT, Red = Pred (overlap becomes yellow-ish).
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for m in gt_masks:
        overlay = Image.alpha_composite(overlay, _mask_to_rgba(m, rgb=(0, 255, 0)))
    for m in pred_masks:
        overlay = Image.alpha_composite(overlay, _mask_to_rgba(m, rgb=(255, 0, 0)))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def run_instance_seg_demo(
    *,
    run_dir: str | Path | None = None,
    seed: int = 0,
    num_images: int = 8,
    image_size: int = 96,
    max_instances: int = 2,
    background: str = "synthetic",
    output_name: str = "instance_seg_demo_report.json",
) -> Path:
    """Create a tiny synthetic instance-seg dataset + predictions, then evaluate mask mAP.

    This demo is designed to run on CPU with only numpy + Pillow.
    """

    np, (Image, ImageDraw) = _require_deps()

    rng = random.Random(int(seed))
    np.random.seed(int(seed))

    if run_dir is None:
        run_dir = Path("demo_output") / "instance_seg" / _utc_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    images_dir = run_dir / "images"
    gt_dir = run_dir / "gt_masks"
    pred_dir = run_dir / "pred_masks"
    overlays_dir = run_dir / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    bg_mode = str(background).strip().lower()
    if bg_mode not in ("synthetic", "coco128"):
        raise ValueError(f"unknown background: {background} (expected: synthetic|coco128)")

    coco_pairs: list[tuple[Path, Path]] | None = None
    if bg_mode == "coco128":
        coco_pairs = _iter_coco128_pairs(coco128_root=Path("data") / "coco128")
        if int(num_images) > len(coco_pairs):
            num_images = len(coco_pairs)
        coco_pairs = rng.sample(coco_pairs, k=int(num_images))

    for i in range(int(num_images)):
        image_name = f"img_{i:04d}.png"
        image_path = images_dir / image_name

        gt_paths: list[str] = []
        gt_classes: list[int] = []
        gt_masks_for_overlay: list[Any] = []
        pred_masks_for_overlay: list[Any] = []
        pred_instances: list[dict[str, Any]] = []

        if bg_mode == "coco128":
            assert coco_pairs is not None
            src_img, src_lab = coco_pairs[i]
            img_rgb = Image.open(src_img).convert("RGB")
            draw_rgb = ImageDraw.Draw(img_rgb)
            w, h = img_rgb.size

            lines = [ln.strip() for ln in src_lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Keep runtime bounded; coco128 images can have many objects.
            lines = lines[: max(1, int(max_instances))]

            for j, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                x0, y0, x1, y1 = _yolo_bbox_to_xyxy(w=w, h=h, xc=xc, yc=yc, bw=bw, bh=bh)

                # COCO128 in this repo is YOLO *bbox* labels, not segmentation polygons.
                # To make overlays look less "blocky" and easier to eyeball, we create a
                # simple ellipse pseudo-mask inside the bbox.
                mask_img = Image.new("L", (int(w), int(h)), 0)
                mask_draw = ImageDraw.Draw(mask_img)
                mask_draw.ellipse([x0, y0, x1, y1], fill=255)
                gt_mask = np.array(mask_img) != 0

                # Lightly annotate the image so masks are visually traceable (match the ellipse mask).
                fill = (40, 140, 220) if (class_id % 2) == 0 else (220, 120, 40)
                draw_rgb.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), fill=None)
                draw_rgb.ellipse([x0 + 1, y0 + 1, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)], outline=fill, fill=None)

                gt_path = gt_dir / f"gt_{i:04d}_{j:02d}.png"
                Image.fromarray((gt_mask.astype("uint8") * 255), mode="L").save(gt_path)
                gt_paths.append(str(gt_path))
                gt_classes.append(int(class_id))
                gt_masks_for_overlay.append(gt_mask)

                if rng.random() < 0.9:
                    pred_mask = gt_mask.copy()
                    if rng.random() < 0.25:
                        shift_y = rng.choice([-2, -1, 1, 2])
                        shift_x = rng.choice([-2, -1, 1, 2])
                        pred_mask = np.roll(pred_mask, shift=shift_y, axis=0)
                        pred_mask = np.roll(pred_mask, shift=shift_x, axis=1)

                    pred_path_rel = Path("pred_masks") / f"pred_{i:04d}_{j:02d}.png"
                    Image.fromarray((pred_mask.astype("uint8") * 255), mode="L").save(run_dir / pred_path_rel)
                    pred_masks_for_overlay.append(pred_mask)
                    pred_instances.append(
                        {
                            "class_id": int(class_id),
                            "score": float(0.9 - 0.1 * rng.random()),
                            "mask": str(pred_path_rel),
                        }
                    )

            # False positive patch.
            if rng.random() < 0.25:
                fp_mask = np.zeros((int(h), int(w)), dtype=bool)
                fp_mask[5:15, 5:15] = True
                fp_path_rel = Path("pred_masks") / f"fp_{i:04d}.png"
                Image.fromarray((fp_mask.astype("uint8") * 255), mode="L").save(run_dir / fp_path_rel)
                pred_masks_for_overlay.append(fp_mask)
                pred_instances.append({"class_id": 0, "score": 0.2, "mask": str(fp_path_rel)})

            img_rgb.save(image_path)
        else:
            # Simple textured background.
            # (Previously this was very close to white; we now also paint the GT shapes on top
            # so users can visually confirm masks align with the image content.)
            bg = (np.random.rand(int(image_size), int(image_size), 3) * 25).astype("uint8") + 200
            img_rgb = Image.fromarray(bg, mode="RGB")
            draw_rgb = ImageDraw.Draw(img_rgb)

            n_inst = rng.randint(1, max(1, int(max_instances)))
            for j in range(int(n_inst)):
                class_id = int(rng.randint(0, 1))
                is_circle = bool(rng.random() < 0.6)

                fill = (40, 140, 220) if int(class_id) == 0 else (220, 120, 40)
                outline = (0, 0, 0)

                if is_circle:
                    r = rng.randint(int(image_size * 0.08), int(image_size * 0.18))
                    cx = rng.randint(r + 2, int(image_size) - r - 3)
                    cy = rng.randint(r + 2, int(image_size) - r - 3)
                    gt_mask = _draw_mask_circle(size=int(image_size), cx=cx, cy=cy, r=r)
                    draw_rgb.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)
                else:
                    w = rng.randint(int(image_size * 0.12), int(image_size * 0.25))
                    h = rng.randint(int(image_size * 0.12), int(image_size * 0.25))
                    x0 = rng.randint(2, int(image_size) - w - 3)
                    y0 = rng.randint(2, int(image_size) - h - 3)
                    gt_mask = _draw_mask_rect(size=int(image_size), x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)
                    draw_rgb.rectangle([x0, y0, x0 + w, y0 + h], fill=fill, outline=outline)

                gt_path = gt_dir / f"gt_{i:04d}_{j:02d}.png"
                Image.fromarray((gt_mask.astype("uint8") * 255), mode="L").save(gt_path)
                gt_paths.append(str(gt_path))
                gt_classes.append(int(class_id))
                gt_masks_for_overlay.append(gt_mask)

                # Predictions: mostly correct, with a bit of noise (shift / dropout / FP).
                if rng.random() < 0.85:
                    pred_mask = gt_mask.copy()
                    if rng.random() < 0.25:
                        shift = rng.choice([-2, -1, 1, 2])
                        pred_mask = np.roll(pred_mask, shift=shift, axis=0)
                    pred_path_rel = Path("pred_masks") / f"pred_{i:04d}_{j:02d}.png"
                    pred_path = run_dir / pred_path_rel
                    Image.fromarray((pred_mask.astype("uint8") * 255), mode="L").save(pred_path)
                    pred_masks_for_overlay.append(pred_mask)
                    pred_instances.append(
                        {
                            "class_id": int(class_id),
                            "score": float(0.9 - 0.1 * rng.random()),
                            "mask": str(pred_path_rel),
                        }
                    )

            # Occasional false positive.
            if rng.random() < 0.25:
                fp_mask = np.zeros((int(image_size), int(image_size)), dtype=bool)
                fp_mask[5:15, 5:15] = True
                fp_path_rel = Path("pred_masks") / f"fp_{i:04d}.png"
                Image.fromarray((fp_mask.astype("uint8") * 255), mode="L").save(run_dir / fp_path_rel)
                pred_masks_for_overlay.append(fp_mask)
                pred_instances.append({"class_id": 0, "score": 0.2, "mask": str(fp_path_rel)})

            # Save the RGB image after painting all GT instances.
            img_rgb.save(image_path)

        # Convenience artifact: visualize masks overlaid on the RGB image.
        overlay_path = overlays_dir / f"overlay_{image_name}"
        _overlay_masks(
            base_rgb_path=image_path,
            output_path=overlay_path,
            gt_masks=gt_masks_for_overlay,
            pred_masks=pred_masks_for_overlay,
        )

        records.append(
            {
                "image": str(image_path),
                "mask_path": list(gt_paths),
                "mask_classes": list(gt_classes),
            }
        )
        predictions.append({"image": image_name, "instances": pred_instances})

    from yolozu.instance_segmentation_eval import evaluate_instance_map

    result = evaluate_instance_map(records=records, predictions_entries=predictions, pred_root=run_dir, return_per_image=True)

    report = {
        "kind": "instance_seg_demo",
        "schema_version": 1,
        "meta": {"seed": int(seed), "run_dir": str(run_dir)},
        "result": {
            "map50_95": float(result.map50_95),
            "map50": float(result.map50),
            "per_class": dict(result.per_class),
            "counts": dict(result.counts),
            "warnings": list(result.warnings),
            "per_image": list(result.per_image or []),
        },
        "artifacts": {
            "images_dir": str(images_dir),
            "gt_masks_dir": str(gt_dir),
            "pred_masks_dir": str(pred_dir),
            "overlays_dir": str(overlays_dir),
        },
    }

    out_path = run_dir / str(output_name)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
