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


def _iter_yolo_pairs(*, images_base: Path, labels_base: Path) -> list[tuple[Path, Path]]:
    if not images_base.exists() or not labels_base.exists():
        raise FileNotFoundError(f"dataset not found under: images={images_base} labels={labels_base}")

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_base.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"):
            continue
        rel = img.relative_to(images_base)
        lab = labels_base / rel.with_suffix(".txt")
        if lab.exists():
            pairs.append((img, lab))
    if not pairs:
        raise FileNotFoundError(f"no image/label pairs found under: images={images_base} labels={labels_base}")
    return pairs


def _iter_coco128_pairs(*, coco128_root: Path) -> list[tuple[Path, Path]]:
    images_base = coco128_root / "images"
    labels_base = coco128_root / "labels"
    if not images_base.exists() or not labels_base.exists():
        raise FileNotFoundError(f"coco128 not found under: {coco128_root}")

    return _iter_yolo_pairs(images_base=images_base, labels_base=labels_base)


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


def _load_coco_instances(
    *,
    instances_json: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(instances_json.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    anns = payload.get("annotations") or []
    if not isinstance(images, list) or not isinstance(anns, list):
        raise ValueError("invalid COCO instances JSON: expected 'images' and 'annotations' lists")

    images_by_id: dict[int, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        try:
            image_id = int(im.get("id"))
        except Exception:
            continue
        images_by_id[image_id] = dict(im)

    anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for a in anns:
        if not isinstance(a, dict):
            continue
        try:
            image_id = int(a.get("image_id"))
        except Exception:
            continue
        anns_by_image_id.setdefault(image_id, []).append(dict(a))

    return images_by_id, anns_by_image_id


def _norm_category_name(name: str) -> str:
    # Normalize for cross-source mapping (COCO categories vs torchvision weights meta).
    # Keep it simple and deterministic; this is demo-only.
    out = []
    for ch in str(name).strip().lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _load_coco_category_name_to_id(*, instances_json: Path) -> dict[str, int]:
    payload = json.loads(instances_json.read_text(encoding="utf-8"))
    cats = payload.get("categories") or []
    if not isinstance(cats, list):
        return {}
    out: dict[str, int] = {}
    for c in cats:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        cid = c.get("id")
        if not isinstance(name, str):
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        out[_norm_category_name(name)] = cid_i
    return out


def _coco_polygons_to_mask(*, w: int, h: int, segmentation: Any) -> Any:
    np, (Image, ImageDraw) = _require_deps()
    if not isinstance(segmentation, list):
        # RLE dict or invalid shape.
        return None

    img = Image.new("L", (int(w), int(h)), 0)
    draw = ImageDraw.Draw(img)
    drew_any = False
    for poly in segmentation:
        if not isinstance(poly, list):
            continue
        if len(poly) < 6:
            continue
        pts: list[tuple[float, float]] = []
        for k in range(0, len(poly) - 1, 2):
            try:
                x = float(poly[k])
                y = float(poly[k + 1])
            except Exception:
                continue
            pts.append((x, y))
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
            drew_any = True

    if not drew_any:
        return None
    return np.array(img) != 0


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
    coco_instances_json: str | Path | None = None,
    coco_images_dir: str | Path | None = None,
    yolo_root: str | Path | None = None,
    yolo_split: str = "val",
    inference: str = "none",
    device: str = "cpu",
    score_threshold: float = 0.5,
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

    gt_mask_kinds: dict[str, int] = {}

    inference_mode = str(inference).strip().lower()
    if inference_mode not in ("none", "auto", "torchvision"):
        raise ValueError(f"unknown inference: {inference} (expected: none|auto|torchvision)")
    if inference_mode != "none" and str(background).strip().lower() != "coco-instances":
        raise ValueError("inference is only supported for background=coco-instances")

    inference_meta: dict[str, Any] = {
        "mode": inference_mode,
        "used": False,
    }

    tv_model = None
    tv_device = None
    if inference_mode in ("auto", "torchvision"):
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore
            import torchvision.transforms.functional as TVF  # type: ignore

            weights_name: str | None = None
            weights_meta_categories: list[str] | None = None
            try:
                # Prefer v2 weights when available; fall back to v1 for older torchvision.
                from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights  # type: ignore

                weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                weights_name = getattr(weights, "name", None) or str(weights)
                meta = getattr(weights, "meta", None)
                if isinstance(meta, dict) and isinstance(meta.get("categories"), list):
                    weights_meta_categories = [str(x) for x in meta.get("categories") if isinstance(x, str)]
                tv_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
            except Exception:
                try:
                    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # type: ignore

                    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                    weights_name = getattr(weights, "name", None) or str(weights)
                    meta = getattr(weights, "meta", None)
                    if isinstance(meta, dict) and isinstance(meta.get("categories"), list):
                        weights_meta_categories = [str(x) for x in meta.get("categories") if isinstance(x, str)]
                    tv_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
                except Exception:
                    tv_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
                    weights_name = "pretrained=True"

            if str(device).strip().lower() == "auto":
                if torch.cuda.is_available():
                    tv_device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    tv_device = torch.device("mps")
                else:
                    tv_device = torch.device("cpu")
            else:
                tv_device = torch.device(str(device))

            tv_model.to(tv_device)
            tv_model.eval()

            inference_meta.update(
                {
                    "backend": getattr(tv_model, "__class__", type(tv_model)).__name__,
                    "device": str(tv_device),
                    "weights": weights_name,
                    "weights_categories": int(len(weights_meta_categories)) if weights_meta_categories else None,
                    "torch": getattr(torch, "__version__", None),
                    "torchvision": getattr(torchvision, "__version__", None),
                    "score_threshold": float(score_threshold),
                }
            )
            if weights_meta_categories:
                inference_meta["_weights_meta_categories"] = weights_meta_categories

            # Stash for inner loop without importing again.
            inference_meta["_TVF"] = TVF
        except Exception as exc:
            inference_meta.update({"available": False, "error": repr(exc)})
            if inference_mode == "torchvision":
                raise RuntimeError(
                    "inference=torchvision requested but torch/torchvision could not be initialized"
                ) from exc

    bg_mode = str(background).strip().lower()
    if bg_mode not in ("synthetic", "coco128", "coco-instances", "yolo-bbox"):
        raise ValueError(f"unknown background: {background} (expected: synthetic|coco128|coco-instances|yolo-bbox)")

    coco_pairs: list[tuple[Path, Path]] | None = None
    if bg_mode == "coco128":
        coco_pairs = _iter_coco128_pairs(coco128_root=Path("data") / "coco128")
        if int(num_images) > len(coco_pairs):
            num_images = len(coco_pairs)
        coco_pairs = rng.sample(coco_pairs, k=int(num_images))

    yolo_pairs: list[tuple[Path, Path]] | None = None
    yolo_meta: dict[str, Any] = {}
    if bg_mode == "yolo-bbox":
        if yolo_root is None:
            raise ValueError(
                "background=yolo-bbox requires yolo_root (dataset root with images/<split> and labels/<split>)"
            )
        yolo_root_path = Path(yolo_root)
        split = str(yolo_split).strip()
        images_base = yolo_root_path / "images" / split
        labels_base = yolo_root_path / "labels" / split
        yolo_pairs = _iter_yolo_pairs(images_base=images_base, labels_base=labels_base)
        if int(num_images) > len(yolo_pairs):
            num_images = len(yolo_pairs)
        yolo_pairs = rng.sample(yolo_pairs, k=int(num_images))
        yolo_meta = {"yolo_root": str(yolo_root_path), "yolo_split": split}

    coco_images_by_id: dict[int, dict[str, Any]] | None = None
    coco_anns_by_image_id: dict[int, list[dict[str, Any]]] | None = None
    coco_image_ids: list[int] | None = None
    coco_images_root: Path | None = None
    coco_name_to_id: dict[str, int] = {}
    if bg_mode == "coco-instances":
        if coco_instances_json is None:
            coco_instances_json = Path("data") / "coco" / "annotations" / "instances_val2017.json"
        if coco_images_dir is None:
            coco_images_dir = Path("data") / "coco" / "images" / "val2017"

        instances_path = Path(coco_instances_json)
        images_root = Path(coco_images_dir)
        if not instances_path.exists():
            raise FileNotFoundError(f"instances json not found: {instances_path}")
        if not images_root.exists():
            raise FileNotFoundError(f"images dir not found: {images_root}")
        coco_images_by_id, coco_anns_by_image_id = _load_coco_instances(instances_json=instances_path)
        coco_images_root = images_root
        coco_name_to_id = _load_coco_category_name_to_id(instances_json=instances_path)

        candidate_ids: list[int] = []
        for image_id, anns in coco_anns_by_image_id.items():
            if image_id not in coco_images_by_id:
                continue
            has_poly = False
            for a in anns:
                if int(a.get("iscrowd") or 0) == 1:
                    continue
                if isinstance(a.get("segmentation"), list):
                    has_poly = True
                    break
            if has_poly:
                candidate_ids.append(int(image_id))
        if not candidate_ids:
            raise FileNotFoundError("no polygon segmentations found in instances json (or only crowd/RLE)")
        if int(num_images) > len(candidate_ids):
            num_images = len(candidate_ids)
        coco_image_ids = rng.sample(candidate_ids, k=int(num_images))

    for i in range(int(num_images)):
        image_name = f"img_{i:04d}.png"
        image_path = images_dir / image_name

        gt_paths: list[str] = []
        gt_classes: list[int] = []
        gt_masks_for_overlay: list[Any] = []
        pred_masks_for_overlay: list[Any] = []
        pred_instances: list[dict[str, Any]] = []

        if bg_mode == "coco-instances":
            assert coco_images_by_id is not None
            assert coco_anns_by_image_id is not None
            assert coco_image_ids is not None
            assert coco_images_root is not None

            image_id = int(coco_image_ids[i])
            im = coco_images_by_id.get(image_id) or {}
            file_name = str(im.get("file_name") or "")
            if not file_name:
                raise FileNotFoundError(f"COCO image file_name missing for id={image_id}")
            src_img = coco_images_root / file_name
            if not src_img.exists():
                raise FileNotFoundError(f"COCO image not found: {src_img}")

            img_rgb_raw = Image.open(src_img).convert("RGB")
            img_rgb = img_rgb_raw.copy()
            draw_rgb = ImageDraw.Draw(img_rgb)
            w, h = img_rgb.size

            anns = coco_anns_by_image_id.get(image_id) or []
            kept = 0
            for j, a in enumerate(anns):
                if kept >= max(1, int(max_instances)):
                    break
                if int(a.get("iscrowd") or 0) == 1:
                    continue
                seg = a.get("segmentation")
                gt_mask = _coco_polygons_to_mask(w=int(w), h=int(h), segmentation=seg)
                if gt_mask is None:
                    continue

                class_id = int(a.get("category_id") or 0)

                # Optional outline to make visual inspection easy.
                fill = (40, 140, 220) if (class_id % 2) == 0 else (220, 120, 40)
                if isinstance(seg, list):
                    for poly in seg:
                        if not isinstance(poly, list) or len(poly) < 6:
                            continue
                        pts = []
                        for k in range(0, len(poly) - 1, 2):
                            try:
                                pts.append((float(poly[k]), float(poly[k + 1])))
                            except Exception:
                                pass
                        if len(pts) >= 3:
                            draw_rgb.line(pts + [pts[0]], fill=fill, width=2)

                gt_path = gt_dir / f"gt_{i:04d}_{kept:02d}.png"
                Image.fromarray((gt_mask.astype("uint8") * 255), mode="L").save(gt_path)
                gt_paths.append(str(gt_path))
                gt_classes.append(int(class_id))
                gt_masks_for_overlay.append(gt_mask)

                kept += 1

            # Predictions
            if tv_model is not None and tv_device is not None:
                try:
                    import torch  # type: ignore

                    TVF = inference_meta.get("_TVF")
                    if TVF is None:
                        raise RuntimeError("torchvision transforms not available")

                    with torch.no_grad():
                        x = TVF.to_tensor(img_rgb_raw).to(tv_device)
                        out = tv_model([x])[0]

                    labels = out.get("labels")
                    scores = out.get("scores")
                    masks = out.get("masks")
                    if labels is None or scores is None or masks is None:
                        raise RuntimeError("unexpected torchvision output (missing labels/scores/masks)")

                    labels_list = [int(v) for v in labels.detach().cpu().tolist()]
                    scores_list = [float(v) for v in scores.detach().cpu().tolist()]
                    masks_t = masks.detach().cpu()

                    inference_meta["used"] = True
                    if coco_name_to_id:
                        inference_meta["coco_category_map"] = {
                            "available": True,
                            "num_categories": int(len(coco_name_to_id)),
                        }

                    kept_pred = 0
                    mapped = 0
                    fallback = 0
                    tv_categories = inference_meta.get("_weights_meta_categories")
                    if not isinstance(tv_categories, list):
                        tv_categories = None
                    for k in range(len(scores_list)):
                        if kept_pred >= max(1, int(max_instances)):
                            break
                        score = float(scores_list[k])
                        if score < float(score_threshold):
                            continue

                        m = masks_t[k, 0]
                        # Ensure mask matches image size (H, W)
                        if tuple(m.shape) != (int(h), int(w)):
                            m_img = Image.fromarray((m.numpy() * 255).astype("uint8"), mode="L").resize(
                                (int(w), int(h)), resample=Image.NEAREST
                            )
                            mask_bool = (np.array(m_img) > 127).astype(bool)
                        else:
                            mask_bool = (m.numpy() > 0.5).astype(bool)

                        class_id = int(labels_list[k])
                        if coco_name_to_id and tv_categories:
                            try:
                                # Torchvision labels are indices into weights.meta["categories"].
                                if 0 <= int(class_id) < len(tv_categories):
                                    name = str(tv_categories[int(class_id)])
                                    cid = coco_name_to_id.get(_norm_category_name(name))
                                    if cid is not None:
                                        class_id = int(cid)
                                        mapped += 1
                                    else:
                                        fallback += 1
                                else:
                                    fallback += 1
                            except Exception:
                                fallback += 1

                        pred_path_rel = Path("pred_masks") / f"pred_{i:04d}_{kept_pred:02d}.png"
                        Image.fromarray((mask_bool.astype("uint8") * 255), mode="L").save(run_dir / pred_path_rel)
                        pred_masks_for_overlay.append(mask_bool)
                        pred_instances.append(
                            {
                                "class_id": int(class_id),
                                "score": float(score),
                                "mask": str(pred_path_rel),
                            }
                        )
                        kept_pred += 1
                    if coco_name_to_id and tv_categories:
                        inference_meta["class_id_space"] = "coco_category_id"
                        inference_meta["class_id_mapping"] = {"mapped": int(mapped), "fallback": int(fallback)}
                except Exception as exc:
                    # Fall back to synthetic-ish predictions if inference fails mid-run.
                    inference_meta.setdefault("warnings", []).append(f"torchvision inference failed: {exc!r}")

            if not pred_instances:
                # Fallback: generate pseudo predictions from GT so evaluation has something to show.
                for j, gt_mask in enumerate(gt_masks_for_overlay):
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
                                "class_id": int(gt_classes[j]) if j < len(gt_classes) else 0,
                                "score": float(0.9 - 0.1 * rng.random()),
                                "mask": str(pred_path_rel),
                            }
                        )

            if kept == 0:
                # No usable polygons for this image; still write it out for transparency.
                img_rgb.save(image_path)
            else:
                img_rgb.save(image_path)

        elif bg_mode == "coco128":
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

                fill = (40, 140, 220) if (class_id % 2) == 0 else (220, 120, 40)

                # COCO128 in this repo is YOLO *bbox* labels (5 columns). Some users may
                # provide a YOLO-seg variant where each row is: class x1 y1 x2 y2 ...
                # (normalized polygon points). If polygons are present, use them to
                # generate masks that better match the image.
                if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    pts: list[tuple[float, float]] = []
                    for k in range(1, len(parts) - 1, 2):
                        try:
                            x = float(parts[k]) * float(w)
                            y = float(parts[k + 1]) * float(h)
                        except Exception:
                            continue
                        pts.append((x, y))
                    if len(pts) < 3:
                        continue
                    mask_img = Image.new("L", (int(w), int(h)), 0)
                    mask_draw = ImageDraw.Draw(mask_img)
                    mask_draw.polygon(pts, fill=255)
                    gt_mask = np.array(mask_img) != 0
                    draw_rgb.line(pts + [pts[0]], fill=fill, width=2)
                    gt_mask_kinds["yolo_seg_polygon"] = int(gt_mask_kinds.get("yolo_seg_polygon", 0)) + 1
                else:
                    xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    x0, y0, x1, y1 = _yolo_bbox_to_xyxy(w=w, h=h, xc=xc, yc=yc, bw=bw, bh=bh)

                    # bbox-only fallback: ellipse pseudo-mask inside the bbox.
                    mask_img = Image.new("L", (int(w), int(h)), 0)
                    mask_draw = ImageDraw.Draw(mask_img)
                    mask_draw.ellipse([x0, y0, x1, y1], fill=255)
                    gt_mask = np.array(mask_img) != 0

                    # Lightly annotate the image so masks are visually traceable.
                    draw_rgb.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), fill=None)
                    draw_rgb.ellipse([x0 + 1, y0 + 1, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)], outline=fill, fill=None)
                    gt_mask_kinds["yolo_bbox_ellipse"] = int(gt_mask_kinds.get("yolo_bbox_ellipse", 0)) + 1

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
        elif bg_mode == "yolo-bbox":
            assert yolo_pairs is not None
            src_img, src_lab = yolo_pairs[i]
            img_rgb = Image.open(src_img).convert("RGB")
            draw_rgb = ImageDraw.Draw(img_rgb)
            w, h = img_rgb.size

            lines = [ln.strip() for ln in src_lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
            lines = lines[: max(1, int(max_instances))]

            for j, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))

                fill = (40, 140, 220) if (class_id % 2) == 0 else (220, 120, 40)

                if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    pts: list[tuple[float, float]] = []
                    for k in range(1, len(parts) - 1, 2):
                        try:
                            x = float(parts[k]) * float(w)
                            y = float(parts[k + 1]) * float(h)
                        except Exception:
                            continue
                        pts.append((x, y))
                    if len(pts) < 3:
                        continue
                    mask_img = Image.new("L", (int(w), int(h)), 0)
                    mask_draw = ImageDraw.Draw(mask_img)
                    mask_draw.polygon(pts, fill=255)
                    gt_mask = np.array(mask_img) != 0
                    draw_rgb.line(pts + [pts[0]], fill=fill, width=2)
                    gt_mask_kinds["yolo_seg_polygon"] = int(gt_mask_kinds.get("yolo_seg_polygon", 0)) + 1
                else:
                    xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    x0, y0, x1, y1 = _yolo_bbox_to_xyxy(w=w, h=h, xc=xc, yc=yc, bw=bw, bh=bh)

                    mask_img = Image.new("L", (int(w), int(h)), 0)
                    mask_draw = ImageDraw.Draw(mask_img)
                    mask_draw.ellipse([x0, y0, x1, y1], fill=255)
                    gt_mask = np.array(mask_img) != 0

                    draw_rgb.ellipse([x0, y0, x1, y1], outline=(0, 0, 0), fill=None)
                    draw_rgb.ellipse(
                        [x0 + 1, y0 + 1, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)],
                        outline=fill,
                        fill=None,
                    )
                    gt_mask_kinds["yolo_bbox_ellipse"] = int(gt_mask_kinds.get("yolo_bbox_ellipse", 0)) + 1

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

    warnings = list(result.warnings)
    if bg_mode in ("coco128", "yolo-bbox"):
        if int(gt_mask_kinds.get("yolo_seg_polygon", 0)) == 0:
            prefix = "coco128" if bg_mode == "coco128" else "yolo-bbox"
            warnings.append(
                f"{prefix} labels are bbox-only; gt_masks are pseudo (ellipse-in-bbox) and will not match object boundaries"
            )

    if bg_mode == "coco-instances":
        if inference_mode == "none":
            warnings.append("inference disabled: predictions are synthetic (GT-derived)")
        elif not bool(inference_meta.get("used")):
            warnings.append("inference not used: falling back to synthetic (GT-derived) predictions")
            err = inference_meta.get("error")
            if err:
                warnings.append(f"inference init error: {err}")
        else:
            warnings.append(
                "inference used: torchvision Mask R-CNN; class_id is normalized to COCO category_id when weights meta provides category names"
            )

    report = {
        "kind": "instance_seg_demo",
        "schema_version": 1,
        "meta": {
            "seed": int(seed),
            "run_dir": str(run_dir),
            "background": str(bg_mode),
            "gt_mask_kinds": dict(gt_mask_kinds),
            "inference": {k: v for k, v in dict(inference_meta).items() if not str(k).startswith("_")},
            "dataset": dict(yolo_meta) if bg_mode == "yolo-bbox" else {},
        },
        "result": {
            "map50_95": float(result.map50_95),
            "map50": float(result.map50),
            "per_class": dict(result.per_class),
            "counts": dict(result.counts),
            "warnings": warnings,
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
