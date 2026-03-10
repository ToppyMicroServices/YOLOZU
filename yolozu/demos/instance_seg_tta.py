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
    except Exception:
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
    merge_iou: float = 0.3,
    mask_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    import torch  # type: ignore
    import torchvision.transforms.functional as TVF  # type: ignore

    np, _ = _require_deps()
    width, height = image.size
    with torch.no_grad():
        x = TVF.to_tensor(image).to(torch_device)
        out_base = model([x])[0]
        out_flip = model([TVF.hflip(x)])[0]

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

    merged: list[dict[str, Any]] = []
    used = [False] * len(flip)
    for det in base:
        best_idx = -1
        best_iou = 0.0
        for idx, alt in enumerate(flip):
            if used[idx] or int(alt.get("class_id", -1)) != int(det.get("class_id", -2)):
                continue
            iou = mask_iou(det["mask"], alt["mask"])
            if iou > best_iou:
                best_iou = float(iou)
                best_idx = int(idx)
        if best_idx >= 0 and best_iou >= float(merge_iou):
            used[best_idx] = True
            alt = flip[best_idx]
            prob = (
                det["mask"].astype("float32") * float(det.get("score", 0.0))
                + alt["mask"].astype("float32") * float(alt.get("score", 0.0))
            ) / max(1e-6, float(det.get("score", 0.0)) + float(alt.get("score", 0.0)))
            merged.append(
                {
                    "class_id": int(det.get("class_id", 0)),
                    "score": float(max(float(det.get("score", 0.0)), float(alt.get("score", 0.0)))),
                    "mask": (prob >= float(mask_threshold)),
                }
            )
        else:
            merged.append(det)

    for idx, alt in enumerate(flip):
        if not used[idx]:
            merged.append(alt)

    merged.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("class_id", 0))))
    return merged[: max(1, int(max_instances))]


def _score_predictions(preds: list[dict[str, Any]], gt_instances: list[dict[str, Any]]) -> dict[str, Any]:
    used = [False] * len(gt_instances)
    matched_ious: list[float] = []
    matched_classes: list[int] = []
    tp = 0
    for pred in preds:
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
    fn = sum(0 if flag else 1 for flag in used)
    fp = max(0, len(preds) - tp)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "matched_classes": matched_classes,
        "pred_instances": int(len(preds)),
        "gt_instances": int(len(gt_instances)),
    }


def _select_best_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("no cases to select from")

    def _rank(case: dict[str, Any]) -> tuple[float, float, float, float]:
        delta = case.get("delta") or {}
        return (
            float(delta.get("tp", 0.0)),
            float(-delta.get("fn", 0.0)),
            float(-delta.get("fp", 0.0)),
            float(delta.get("mean_iou", 0.0)),
        )

    return max(cases, key=_rank)


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
        )
        raw_score = _score_predictions(raw, gt_instances)
        tta_score = _score_predictions(tta, gt_instances)
        delta = {
            "tp": int(tta_score["tp"] - raw_score["tp"]),
            "fp": int(tta_score["fp"] - raw_score["fp"]),
            "fn": int(tta_score["fn"] - raw_score["fn"]),
            "mean_iou": float(tta_score["mean_iou"] - raw_score["mean_iou"]),
        }
        case = {
            "image_id": int(cand_id),
            "file_name": file_name,
            "raw": raw_score,
            "tta": tta_score,
            "delta": delta,
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

    sanitized_cases: list[dict[str, Any]] = []
    for case in cases:
        sanitized_cases.append(
            {
                "image_id": int(case["image_id"]),
                "file_name": str(case["file_name"]),
                "raw": dict(case["raw"]),
                "tta": dict(case["tta"]),
                "delta": dict(case["delta"]),
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
            "tta_mode": "hflip_mask_fusion",
        },
        "model": model_meta,
        "selected": {
            "image_id": int(selected["image_id"]),
            "file_name": str(selected["file_name"]),
            "raw": dict(selected["raw"]),
            "tta": dict(selected["tta"]),
            "delta": dict(selected["delta"]),
        },
        "scan_summary": sanitized_cases,
        "artifacts": {
            "original": str(selected_original_path),
            "shifted": str(selected_shifted_path),
            "overlay_raw": str(overlay_raw_path),
            "overlay_tta": str(overlay_tta_path),
        },
    }
    out_path = run_dir_p / str(output_name)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
