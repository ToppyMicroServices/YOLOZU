from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path
from typing import Any

from yolozu.demos.instance_seg import (
    _coco_polygons_to_mask,
    _load_coco_category_name_to_id,
    _load_coco_instances,
    _norm_category_name,
    _overlay_masks,
    _require_deps,
)
from yolozu.eval.instance_segmentation_eval import mask_iou


SUPPORTED_CORRUPTIONS = ("gaussian_blur", "gaussian_noise", "brightness", "contrast", "jpeg")


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _apply_corruption(image: Any, *, corruption: str, severity: int, seed: int) -> Any:
    np, (Image, _) = _require_deps()
    from PIL import ImageEnhance, ImageFilter

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
        noise = np.random.default_rng(int(seed)).normal(128.0, sigma, size=(height, width, 1)).clip(0, 255).astype("uint8")
        noise_rgb = np.repeat(noise, 3, axis=2)
        alpha = float(min(0.75, 0.12 * sev + rng.uniform(0.0, 0.02)))
        return Image.blend(rgb, Image.fromarray(noise_rgb, mode="RGB"), alpha)

    if corruption == "jpeg":
        quality = max(10, 95 - 15 * sev)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=int(quality), optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as tmp:
            return tmp.convert("RGB")

    raise ValueError(f"unsupported corruption: {corruption}")


def _load_torchvision_model(*, device: str) -> tuple[Any, Any, list[str], dict[str, Any]]:
    import torch  # type: ignore
    import torchvision  # type: ignore

    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights  # type: ignore

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        weights_name = getattr(weights, "name", None) or str(weights)
        meta = getattr(weights, "meta", None)
        categories = [str(x) for x in (meta.get("categories") or [])] if isinstance(meta, dict) else []
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    except (AttributeError, ImportError, OSError, RuntimeError):
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # type: ignore

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_name = getattr(weights, "name", None) or str(weights)
        meta = getattr(weights, "meta", None)
        categories = [str(x) for x in (meta.get("categories") or [])] if isinstance(meta, dict) else []
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    if str(device).strip().lower() == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(str(device))

    model.to(torch_device)
    model.eval()
    meta = {
        "backend": getattr(model, "__class__", type(model)).__name__,
        "device": str(torch_device),
        "weights": str(weights_name),
        "torch": getattr(torch, "__version__", None),
        "torchvision": getattr(torchvision, "__version__", None),
        "weights_categories": int(len(categories)),
    }
    return model, torch_device, categories, meta


def _predict_instances(
    *,
    model: Any,
    torch_device: Any,
    image: Any,
    categories: list[str],
    coco_name_to_id: dict[str, int],
    score_threshold: float,
    max_instances: int,
) -> list[dict[str, Any]]:
    import torch  # type: ignore
    import torchvision.transforms.functional as TVF  # type: ignore

    np, (Image, _) = _require_deps()
    width, height = image.size
    with torch.no_grad():
        x = TVF.to_tensor(image).to(torch_device)
        out = model([x])[0]

    labels = out.get("labels")
    scores = out.get("scores")
    masks = out.get("masks")
    if labels is None or scores is None or masks is None:
        raise RuntimeError("unexpected torchvision output (missing labels/scores/masks)")

    labels_list = [int(v) for v in labels.detach().cpu().tolist()]
    scores_list = [float(v) for v in scores.detach().cpu().tolist()]
    masks_t = masks.detach().cpu()

    preds: list[dict[str, Any]] = []
    for idx, score in enumerate(scores_list):
        if len(preds) >= max(1, int(max_instances)):
            break
        if float(score) < float(score_threshold):
            continue
        mask = masks_t[idx, 0]
        if tuple(mask.shape) != (int(height), int(width)):
            mask_img = Image.fromarray((mask.numpy() * 255).astype("uint8"), mode="L").resize(
                (int(width), int(height)), resample=Image.NEAREST
            )
            mask_bool = (np.array(mask_img) > 127).astype(bool)
        else:
            mask_bool = (mask.numpy() > 0.5).astype(bool)

        class_id = int(labels_list[idx])
        if 0 <= class_id < len(categories):
            mapped = coco_name_to_id.get(_norm_category_name(categories[class_id]))
            if mapped is not None:
                class_id = int(mapped)
        preds.append({"class_id": int(class_id), "score": float(score), "mask": mask_bool})

    preds.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("class_id", 0))))
    return preds


def _predict_instances_hflip_tta(
    *,
    model: Any,
    torch_device: Any,
    image: Any,
    categories: list[str],
    coco_name_to_id: dict[str, int],
    score_threshold: float,
    max_instances: int,
    corruption: str,
    severity: int,
    merge_iou: float = 0.3,
    mask_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    import torch  # type: ignore
    import torchvision.transforms.functional as TVF  # type: ignore

    np, _ = _require_deps()
    width, height = image.size
    with torch.no_grad():
        x = TVF.to_tensor(image).to(torch_device)
        out_base = model([x])[0]
        out_flip = model([TVF.hflip(x)])[0]
        out_brightness = None
        if str(corruption) == "brightness":
            from PIL import ImageEnhance

            lift_factor = min(1.75, 1.10 + 0.12 * max(1, min(5, int(severity))))
            bright_image = ImageEnhance.Brightness(image.convert("RGB")).enhance(float(lift_factor))
            out_brightness = model([TVF.to_tensor(bright_image).to(torch_device)])[0]

    # Reuse the converter logic with the already-produced torchvision outputs.
    def _from_out(out: Any) -> list[dict[str, Any]]:
        labels = out.get("labels")
        scores = out.get("scores")
        masks = out.get("masks")
        if labels is None or scores is None or masks is None:
            raise RuntimeError("unexpected torchvision output (missing labels/scores/masks)")
        labels_list = [int(v) for v in labels.detach().cpu().tolist()]
        scores_list = [float(v) for v in scores.detach().cpu().tolist()]
        masks_t = masks.detach().cpu()
        preds: list[dict[str, Any]] = []
        from PIL import Image

        for idx, score in enumerate(scores_list):
            if len(preds) >= max(1, int(max_instances)):
                break
            if float(score) < float(score_threshold):
                continue
            mask = masks_t[idx, 0]
            if tuple(mask.shape) != (int(height), int(width)):
                mask_img = Image.fromarray((mask.numpy() * 255).astype("uint8"), mode="L").resize(
                    (int(width), int(height)), resample=Image.NEAREST
                )
                mask_bool = (np.array(mask_img) > 127).astype(bool)
            else:
                mask_bool = (mask.numpy() > 0.5).astype(bool)
            class_id = int(labels_list[idx])
            if 0 <= class_id < len(categories):
                mapped = coco_name_to_id.get(_norm_category_name(categories[class_id]))
                if mapped is not None:
                    class_id = int(mapped)
            preds.append({"class_id": int(class_id), "score": float(score), "mask": mask_bool})
        preds.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("class_id", 0))))
        return preds

    base = _from_out(out_base)
    flip = _from_out(out_flip)
    for det in flip:
        det["mask"] = np.fliplr(det["mask"])

    def _mask_nms(items: list[dict[str, Any]], *, iou_threshold: float = 0.75) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for det in sorted(items, key=lambda item: (-float(item.get("score", 0.0)), int(item.get("class_id", 0)))):
            suppress = False
            for prev in kept:
                if int(prev.get("class_id", -1)) != int(det.get("class_id", -2)):
                    continue
                if mask_iou(prev["mask"], det["mask"]) >= float(iou_threshold):
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        return kept

    def _merge_prob(det_a: dict[str, Any], det_b: dict[str, Any]) -> dict[str, Any]:
        prob = (
            det_a["mask"].astype("float32") * float(det_a.get("score", 0.0))
            + det_b["mask"].astype("float32") * float(det_b.get("score", 0.0))
        ) / max(1e-6, float(det_a.get("score", 0.0)) + float(det_b.get("score", 0.0)))
        return {
            "class_id": int(det_a.get("class_id", 0)),
            "score": float(max(float(det_a.get("score", 0.0)), float(det_b.get("score", 0.0)))),
            "mask": (prob >= float(mask_threshold)),
        }

    def _merge_extra(
        current: list[dict[str, Any]],
        extra: list[dict[str, Any]],
        *,
        solo_threshold: float,
    ) -> list[dict[str, Any]]:
        merged_local: list[dict[str, Any]] = []
        used = [False] * len(extra)
        for det in current:
            best_idx = -1
            best_iou = 0.0
            for idx, alt in enumerate(extra):
                if used[idx] or int(alt.get("class_id", -1)) != int(det.get("class_id", -2)):
                    continue
                iou = mask_iou(det["mask"], alt["mask"])
                if iou > best_iou:
                    best_iou = float(iou)
                    best_idx = int(idx)
            if best_idx >= 0 and best_iou >= float(merge_iou):
                used[best_idx] = True
                merged_local.append(_merge_prob(det, extra[best_idx]))
            else:
                merged_local.append(det)

        for idx, alt in enumerate(extra):
            if used[idx] or float(alt.get("score", 0.0)) < float(solo_threshold):
                continue
            duplicate = False
            for prev in merged_local:
                if int(prev.get("class_id", -1)) != int(alt.get("class_id", -2)):
                    continue
                if mask_iou(prev["mask"], alt["mask"]) >= float(merge_iou):
                    duplicate = True
                    break
            if not duplicate:
                merged_local.append(alt)
        return _mask_nms(merged_local)

    merged = _merge_extra(base, flip, solo_threshold=max(float(score_threshold) + 0.10, 0.50))
    if out_brightness is not None:
        brightness_preds = _from_out(out_brightness)
        merged = _merge_extra(merged, brightness_preds, solo_threshold=max(float(score_threshold) + 0.05, 0.40))

    merged = _mask_nms(merged)
    merged.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("class_id", 0))))
    return merged[: max(1, int(max_instances))]


def _union_mask(instances: list[dict[str, Any]], *, shape: tuple[int, int] | None = None) -> Any:
    np, _ = _require_deps()
    if not instances:
        if shape is None:
            return np.zeros((1, 1), dtype=bool)
        return np.zeros(shape, dtype=bool)
    union = np.asarray(instances[0]["mask"], dtype=bool).copy()
    for item in instances[1:]:
        union |= np.asarray(item["mask"], dtype=bool)
    return union


def _visual_metrics(preds: list[dict[str, Any]], gt_instances: list[dict[str, Any]]) -> dict[str, float]:
    np, _ = _require_deps()
    ref_shape: tuple[int, int] | None = None
    for items in (preds, gt_instances):
        if items:
            ref_shape = tuple(np.asarray(items[0]["mask"], dtype=bool).shape)
            break
    pred_union = _union_mask(preds, shape=ref_shape)
    gt_union = _union_mask(gt_instances, shape=ref_shape)
    if pred_union.shape != gt_union.shape:
        raise ValueError("pred/gt mask shapes differ in visual metric calculation")

    inter = float(np.logical_and(pred_union, gt_union).sum())
    union = float(np.logical_or(pred_union, gt_union).sum())
    pred_area = float(pred_union.sum())
    gt_area = float(gt_union.sum())
    fp_area = float(np.logical_and(pred_union, np.logical_not(gt_union)).sum())
    fn_area = float(np.logical_and(gt_union, np.logical_not(pred_union)).sum())
    precision = inter / pred_area if pred_area > 0.0 else (1.0 if gt_area <= 0.0 else 0.0)
    recall = inter / gt_area if gt_area > 0.0 else 1.0
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    return {
        "canvas_iou": (inter / union) if union > 0.0 else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_area_ratio": (fp_area / gt_area) if gt_area > 0.0 else fp_area,
        "fn_area_ratio": (fn_area / gt_area) if gt_area > 0.0 else 0.0,
        "pred_area_ratio": (pred_area / gt_area) if gt_area > 0.0 else pred_area,
    }


def _focus_bbox(
    raw_preds: list[dict[str, Any]],
    tta_preds: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    *,
    min_size: int = 224,
    pad: int = 24,
) -> tuple[int, int, int, int] | None:
    np, _ = _require_deps()
    ref_shape = None
    for items in (raw_preds, tta_preds, gt_instances):
        if items:
            ref_shape = tuple(np.asarray(items[0]["mask"], dtype=bool).shape)
            break
    if ref_shape is None:
        return None
    h, w = int(ref_shape[0]), int(ref_shape[1])
    raw_match = _match_predictions(raw_preds, gt_instances)
    tta_match = _match_predictions(tta_preds, gt_instances)
    raw_matched_gt = set(int(x) for x in raw_match["matched_gt_indices"])
    tta_matched_gt = set(int(x) for x in tta_match["matched_gt_indices"])
    raw_unmatched_pred = set(int(x) for x in raw_match["unmatched_pred_indices"])
    tta_unmatched_pred = set(int(x) for x in tta_match["unmatched_pred_indices"])
    focus_parts: list[dict[str, Any]] = []
    for idx in sorted(tta_matched_gt - raw_matched_gt):
        focus_parts.append({"mask": gt_instances[int(idx)]["mask"]})
    for idx in sorted(raw_unmatched_pred - tta_unmatched_pred):
        focus_parts.append({"mask": raw_preds[int(idx)]["mask"]})

    raw_union = _union_mask(raw_preds, shape=ref_shape)
    tta_union = _union_mask(tta_preds, shape=ref_shape)
    gt_union = _union_mask(gt_instances, shape=ref_shape)
    focus = _union_mask(focus_parts, shape=ref_shape) if focus_parts else np.zeros(ref_shape, dtype=bool)
    if not bool(focus.any()):
        focus = np.logical_or(
            np.logical_and(np.logical_not(raw_union), np.logical_and(tta_union, gt_union)),
            np.logical_and(raw_union, np.logical_and(np.logical_not(tta_union), np.logical_not(gt_union))),
        )
    if not bool(focus.any()):
        focus = np.logical_xor(raw_union, tta_union)
    if not bool(focus.any()):
        focus = gt_union
    if not bool(focus.any()):
        return None
    ys, xs = np.nonzero(focus)
    x0 = max(0, int(xs.min()) - int(pad))
    y0 = max(0, int(ys.min()) - int(pad))
    x1 = min(w, int(xs.max()) + int(pad) + 1)
    y1 = min(h, int(ys.max()) + int(pad) + 1)
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < int(min_size):
        cx = (x0 + x1) // 2
        half = int(min_size) // 2
        x0 = max(0, cx - half)
        x1 = min(w, x0 + int(min_size))
        x0 = max(0, x1 - int(min_size))
    if box_h < int(min_size):
        cy = (y0 + y1) // 2
        half = int(min_size) // 2
        y0 = max(0, cy - half)
        y1 = min(h, y0 + int(min_size))
        y0 = max(0, y1 - int(min_size))
    return int(x0), int(y0), int(x1), int(y1)


def _overlay_delta(
    *,
    base_rgb_path: Path,
    output_path: Path,
    raw_preds: list[dict[str, Any]],
    tta_preds: list[dict[str, Any]],
    gt_instances: list[dict[str, Any]],
    alpha: float = 0.50,
) -> None:
    np, (Image, _) = _require_deps()
    base = Image.open(base_rgb_path).convert("RGBA")
    w, h = base.size
    ref_shape = (h, w)
    raw_union = _union_mask(raw_preds, shape=ref_shape)
    tta_union = _union_mask(tta_preds, shape=ref_shape)
    gt_union = _union_mask(gt_instances, shape=ref_shape)

    positive = np.logical_or(
        np.logical_and(np.logical_not(raw_union), np.logical_and(tta_union, gt_union)),
        np.logical_and(raw_union, np.logical_and(np.logical_not(tta_union), np.logical_not(gt_union))),
    )
    negative = np.logical_or(
        np.logical_and(raw_union, np.logical_and(np.logical_not(tta_union), gt_union)),
        np.logical_and(np.logical_not(gt_union), np.logical_and(tta_union, np.logical_not(raw_union))),
    )

    overlay = np.zeros((h, w, 4), dtype="uint8")
    overlay[positive, 1] = 255
    overlay[positive, 3] = int(max(0.0, min(1.0, float(alpha))) * 255)
    overlay[negative, 0] = 255
    overlay[negative, 3] = int(max(0.0, min(1.0, float(alpha))) * 255)

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _match_predictions(preds: list[dict[str, Any]], gt_instances: list[dict[str, Any]]) -> dict[str, Any]:
    used = [False] * len(gt_instances)
    matched_ious: list[float] = []
    matched_classes: list[int] = []
    matched_gt_indices: list[int] = []
    matched_pred_indices: list[int] = []
    tp = 0
    for pred_idx, pred in enumerate(preds):
        best_idx = -1
        best_iou = 0.0
        pred_cid = int(pred.get("class_id", -1))
        for idx, gt in enumerate(gt_instances):
            if used[idx] or int(gt.get("class_id", -2)) != pred_cid:
                continue
            iou = mask_iou(pred["mask"], gt["mask"])
            if iou > best_iou:
                best_iou = float(iou)
                best_idx = int(idx)
        if best_idx >= 0 and best_iou >= 0.5:
            used[best_idx] = True
            tp += 1
            matched_ious.append(float(best_iou))
            matched_classes.append(int(pred_cid))
            matched_gt_indices.append(int(best_idx))
            matched_pred_indices.append(int(pred_idx))
    fn = sum(0 if flag else 1 for flag in used)
    fp = max(0, len(preds) - tp)
    matched_pred_set = set(matched_pred_indices)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "matched_classes": matched_classes,
        "matched_gt_indices": matched_gt_indices,
        "matched_pred_indices": matched_pred_indices,
        "unmatched_gt_indices": [idx for idx, flag in enumerate(used) if not flag],
        "unmatched_pred_indices": [idx for idx in range(len(preds)) if idx not in matched_pred_set],
        "pred_instances": int(len(preds)),
        "gt_instances": int(len(gt_instances)),
    }


def _score_predictions(preds: list[dict[str, Any]], gt_instances: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _match_predictions(preds, gt_instances)
    return {
        "tp": int(summary["tp"]),
        "fp": int(summary["fp"]),
        "fn": int(summary["fn"]),
        "mean_iou": float(summary["mean_iou"]),
        "matched_classes": list(summary["matched_classes"]),
        "pred_instances": int(summary["pred_instances"]),
        "gt_instances": int(summary["gt_instances"]),
    }


def _select_best_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("no cases to select from")

    preferred = [
        case
        for case in cases
        if float((case.get("visual_delta") or {}).get("canvas_iou", 0.0)) > 0.0
        and float((case.get("visual_delta") or {}).get("fp_area_ratio", 0.0)) <= 0.0
    ]
    ranked_cases = preferred or cases

    def _rank(case: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
        delta = case.get("delta") or {}
        visual_delta = case.get("visual_delta") or {}
        return (
            float(visual_delta.get("canvas_iou", 0.0)),
            float(visual_delta.get("f1", 0.0)),
            float(-visual_delta.get("fp_area_ratio", 0.0)),
            float(-visual_delta.get("fn_area_ratio", 0.0)),
            float(delta.get("tp", 0.0)),
            float(-delta.get("fn", 0.0)),
            float(-delta.get("fp", 0.0)),
            float(delta.get("mean_iou", 0.0)),
        )

    return max(ranked_cases, key=_rank)


def run_instance_seg_tta_demo(
    *,
    run_dir: str | Path | None = None,
    seed: int = 0,
    coco_instances_json: str | Path | None = None,
    coco_images_dir: str | Path | None = None,
    device: str = "auto",
    score_threshold: float = 0.25,
    max_instances: int = 8,
    num_images: int = 8,
    corruption: str = "brightness",
    severity: int = 5,
    image_id: int | None = None,
    output_name: str = "instance_seg_tta_demo_report.json",
) -> Path:
    _, (Image, _) = _require_deps()

    if corruption not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"unsupported corruption: {corruption} (supported: {', '.join(SUPPORTED_CORRUPTIONS)})")
    if int(severity) < 1 or int(severity) > 5:
        raise ValueError("--severity must be in [1, 5]")

    if run_dir is None:
        run_dir_p = Path("demo_output") / "instance_seg_tta" / _utc_run_id()
    else:
        run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)

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

    images_by_id, anns_by_image_id = _load_coco_instances(instances_json=instances_path)
    coco_name_to_id = _load_coco_category_name_to_id(instances_json=instances_path)
    model, torch_device, categories, model_meta = _load_torchvision_model(device=str(device))

    candidate_ids: list[int] = []
    for cand_id, anns in anns_by_image_id.items():
        if cand_id not in images_by_id:
            continue
        has_poly = False
        for ann in anns:
            if int(ann.get("iscrowd") or 0) == 1:
                continue
            if isinstance(ann.get("segmentation"), list):
                has_poly = True
                break
        if has_poly:
            candidate_ids.append(int(cand_id))
    candidate_ids.sort()
    if image_id is not None:
        candidate_ids = [int(image_id)] if int(image_id) in candidate_ids else []
    else:
        rng = random.Random(int(seed))
        if int(num_images) < len(candidate_ids):
            candidate_ids = sorted(rng.sample(candidate_ids, k=int(num_images)))
    if not candidate_ids:
        raise FileNotFoundError("no polygon segmentation image candidates available for instance-seg TTA demo")

    cases: list[dict[str, Any]] = []

    for cand_id in candidate_ids:
        image_meta = images_by_id.get(int(cand_id)) or {}
        file_name = str(image_meta.get("file_name") or "")
        if not file_name:
            continue
        src_img = images_root / file_name
        if not src_img.exists():
            continue
        rgb = Image.open(src_img).convert("RGB")
        width, height = rgb.size

        gt_instances: list[dict[str, Any]] = []
        for ann in anns_by_image_id.get(int(cand_id), []) or []:
            if int(ann.get("iscrowd") or 0) == 1:
                continue
            mask = _coco_polygons_to_mask(w=int(width), h=int(height), segmentation=ann.get("segmentation"))
            if mask is None:
                continue
            gt_instances.append({"class_id": int(ann.get("category_id") or 0), "mask": mask})
        if not gt_instances:
            continue

        shifted = _apply_corruption(rgb, corruption=str(corruption), severity=int(severity), seed=int(seed) ^ int(cand_id))
        raw = _predict_instances(
            model=model,
            torch_device=torch_device,
            image=shifted,
            categories=categories,
            coco_name_to_id=coco_name_to_id,
            score_threshold=float(score_threshold),
            max_instances=int(max_instances),
        )
        tta = _predict_instances_hflip_tta(
            model=model,
            torch_device=torch_device,
            image=shifted,
            categories=categories,
            coco_name_to_id=coco_name_to_id,
            score_threshold=float(score_threshold),
            max_instances=int(max_instances),
            corruption=str(corruption),
            severity=int(severity),
        )
        raw_score = _score_predictions(raw, gt_instances)
        tta_score = _score_predictions(tta, gt_instances)
        raw_visual = _visual_metrics(raw, gt_instances)
        tta_visual = _visual_metrics(tta, gt_instances)
        delta = {
            "tp": int(tta_score["tp"] - raw_score["tp"]),
            "fp": int(tta_score["fp"] - raw_score["fp"]),
            "fn": int(tta_score["fn"] - raw_score["fn"]),
            "mean_iou": float(tta_score["mean_iou"] - raw_score["mean_iou"]),
        }
        visual_delta = {
            "canvas_iou": float(tta_visual["canvas_iou"] - raw_visual["canvas_iou"]),
            "precision": float(tta_visual["precision"] - raw_visual["precision"]),
            "recall": float(tta_visual["recall"] - raw_visual["recall"]),
            "f1": float(tta_visual["f1"] - raw_visual["f1"]),
            "fp_area_ratio": float(tta_visual["fp_area_ratio"] - raw_visual["fp_area_ratio"]),
            "fn_area_ratio": float(tta_visual["fn_area_ratio"] - raw_visual["fn_area_ratio"]),
            "pred_area_ratio": float(tta_visual["pred_area_ratio"] - raw_visual["pred_area_ratio"]),
        }
        case = {
            "image_id": int(cand_id),
            "file_name": file_name,
            "raw": raw_score,
            "tta": tta_score,
            "raw_visual": raw_visual,
            "tta_visual": tta_visual,
            "delta": delta,
            "visual_delta": visual_delta,
            "_raw_preds": raw,
            "_tta_preds": tta,
            "_gt_instances": gt_instances,
            "_shifted_image": shifted,
            "_original_image": rgb,
        }
        cases.append(case)

    if not cases:
        raise RuntimeError("instance-seg TTA demo did not produce any comparable cases")

    selected = _select_best_case(cases)
    selected_dir = run_dir_p / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_original_path = selected_dir / "original.png"
    selected_shifted_path = selected_dir / "shifted.png"
    overlay_raw_path = selected_dir / "overlay_raw.png"
    overlay_tta_path = selected_dir / "overlay_tta.png"
    overlay_delta_path = selected_dir / "overlay_delta.png"
    overlay_raw_focus_path = selected_dir / "overlay_raw_focus.png"
    overlay_tta_focus_path = selected_dir / "overlay_tta_focus.png"

    selected["_original_image"].save(selected_original_path)
    selected["_shifted_image"].save(selected_shifted_path)
    selected_gt_masks = [dict(item)["mask"] for item in selected["_gt_instances"]]
    selected_raw_masks = [dict(item)["mask"] for item in selected["_raw_preds"]]
    selected_tta_masks = [dict(item)["mask"] for item in selected["_tta_preds"]]

    _overlay_masks(
        base_rgb_path=selected_shifted_path,
        output_path=overlay_raw_path,
        gt_masks=selected_gt_masks,
        pred_masks=selected_raw_masks,
    )
    _overlay_masks(
        base_rgb_path=selected_shifted_path,
        output_path=overlay_tta_path,
        gt_masks=selected_gt_masks,
        pred_masks=selected_tta_masks,
    )
    _overlay_delta(
        base_rgb_path=selected_shifted_path,
        output_path=overlay_delta_path,
        raw_preds=selected["_raw_preds"],
        tta_preds=selected["_tta_preds"],
        gt_instances=selected["_gt_instances"],
    )
    focus_box = _focus_bbox(selected["_raw_preds"], selected["_tta_preds"], selected["_gt_instances"])
    if focus_box is not None:
        x0, y0, x1, y1 = focus_box
        from PIL import Image

        with Image.open(overlay_raw_path) as raw_img:
            raw_img.crop((int(x0), int(y0), int(x1), int(y1))).save(overlay_raw_focus_path)
        with Image.open(overlay_tta_path) as tta_img:
            tta_img.crop((int(x0), int(y0), int(x1), int(y1))).save(overlay_tta_focus_path)

    sanitized_cases: list[dict[str, Any]] = []
    for case in cases:
        sanitized_cases.append(
            {
                "image_id": int(case["image_id"]),
                "file_name": str(case["file_name"]),
                "raw": dict(case["raw"]),
                "tta": dict(case["tta"]),
                "raw_visual": dict(case["raw_visual"]),
                "tta_visual": dict(case["tta_visual"]),
                "delta": dict(case["delta"]),
                "visual_delta": dict(case["visual_delta"]),
            }
        )

    report = {
        "kind": "instance_seg_tta_demo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interface_contract": "predictions interface contract",
        "settings": {
            "run_dir": str(run_dir_p),
            "coco_instances_json": str(instances_path),
            "coco_images_dir": str(images_root),
            "seed": int(seed),
            "num_images": int(num_images),
            "image_id": (int(image_id) if image_id is not None else "auto"),
            "corruption": str(corruption),
            "severity": int(severity),
            "device": str(torch_device),
            "score_threshold": float(score_threshold),
            "max_instances": int(max_instances),
            "tta_mode": ("brightness_lift+hflip_mask_fusion" if str(corruption) == "brightness" else "hflip_mask_fusion"),
        },
        "model": model_meta,
        "selected": {
            "image_id": int(selected["image_id"]),
            "file_name": str(selected["file_name"]),
            "raw": dict(selected["raw"]),
            "tta": dict(selected["tta"]),
            "raw_visual": dict(selected["raw_visual"]),
            "tta_visual": dict(selected["tta_visual"]),
            "delta": dict(selected["delta"]),
            "visual_delta": dict(selected["visual_delta"]),
        },
        "scan_summary": sanitized_cases,
        "artifacts": {
            "original": str(selected_original_path),
            "shifted": str(selected_shifted_path),
            "overlay_raw": str(overlay_raw_path),
            "overlay_tta": str(overlay_tta_path),
            "overlay_delta": str(overlay_delta_path),
            "overlay_raw_focus": str(overlay_raw_focus_path),
            "overlay_tta_focus": str(overlay_tta_focus_path),
        },
    }
    out_path = run_dir_p / str(output_name)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
