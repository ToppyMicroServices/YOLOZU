"""ONNX Runtime inference and prediction export.

Runs ONNX Runtime inference on image datasets with letterbox
preprocessing, optional NMS, and raw-output decoding, then writes
schema-correct ``predictions.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform

__all__ = ["export_predictions_onnxrt", "write_predictions_json"]
import time
from pathlib import Path
from typing import Any

from yolozu.dataset import build_manifest
from yolozu.letterbox import compute_letterbox, input_xyxy_to_orig_xyxy, orig_xyxy_to_cxcywh_norm
from yolozu.predictions import validate_predictions_entries


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(data)


def _resolve_boxes_and_scores(outputs: dict[str, object], *, boxes_key: str, scores_key: str, class_key: str | None):
    boxes = outputs.get(boxes_key)
    scores = outputs.get(scores_key)
    if boxes is None:
        raise ValueError(f"missing boxes output: {boxes_key}")
    if scores is None:
        raise ValueError(f"missing scores output: {scores_key}")
    class_ids = outputs.get(class_key) if class_key else None
    return boxes, scores, class_ids


def _split_combined_output(values, *, fmt: str):
    if fmt != "xyxy_score_class":
        raise ValueError(f"unsupported combined format: {fmt}")
    arr = values
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"unsupported combined output shape: {arr.shape}")
    boxes = arr[:, :4]
    scores = arr[:, 4]
    class_ids = arr[:, 5]
    return boxes, scores, class_ids


def _normalize_raw_output(values, *, fmt: str):
    if fmt != "yolo_84":
        raise ValueError(f"unsupported raw format: {fmt}")
    arr = values
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    if arr.shape[0] in (84, 85):
        arr = arr.T
    elif arr.shape[1] not in (84, 85):
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    return arr


def _xywh_to_xyxy(boxes, *, np):
    x, y, w, h = boxes.T
    x1 = x - (w / 2.0)
    y1 = y - (h / 2.0)
    x2 = x + (w / 2.0)
    y2 = y + (h / 2.0)
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy_one_to_many(box, boxes, *, np):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _nms(boxes, scores, *, iou_thresh: float, max_det: int, np):
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < int(max_det):
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy_one_to_many(boxes[i], boxes[order[1:]], np=np)
        order = order[1:][ious <= float(iou_thresh)]
    return np.array(keep, dtype=np.int64)


def _decode_raw_output(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
    np,
):
    data = _normalize_raw_output(raw, fmt="yolo_84")
    if data.shape[1] <= 4:
        raise ValueError(f"raw output has no class scores: {data.shape}")
    boxes_xywh = data[:, :4]
    scores_all = data[:, 4:]
    class_ids = np.argmax(scores_all, axis=1)
    scores = np.max(scores_all, axis=1)

    keep = scores >= float(min_score)
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh, np=np)
    if boxes_xyxy.size == 0:
        return boxes_xyxy, scores, class_ids

    max_nms = 30000
    if boxes_xyxy.shape[0] > max_nms:
        order = scores.argsort()[::-1][:max_nms]
        boxes_xyxy = boxes_xyxy[order]
        scores = scores[order]
        class_ids = class_ids[order]

    if not agnostic:
        max_wh = 7680.0
        offsets = class_ids.astype(np.float32) * max_wh
        boxes_nms = boxes_xyxy.copy()
        boxes_nms[:, 0] += offsets
        boxes_nms[:, 1] += offsets
        boxes_nms[:, 2] += offsets
        boxes_nms[:, 3] += offsets
    else:
        boxes_nms = boxes_xyxy

    keep_idx = _nms(boxes_nms, scores, iou_thresh=iou_thresh, max_det=max_det, np=np)
    return boxes_xyxy[keep_idx], scores[keep_idx], class_ids[keep_idx]


def _decode_raw_ultralytics(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
    np,
):
    try:
        import torch  # type: ignore
        from ultralytics.utils import nms as u_nms  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics + torch are required for raw-postprocess=ultralytics") from exc

    arr = raw
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    if arr.shape[1] in (84, 85) and arr.shape[0] not in (84, 85):
        arr = arr.T
    if arr.shape[0] not in (84, 85):
        raise ValueError(f"unsupported raw output shape: {arr.shape}")
    pred = torch.as_tensor(arr[None, ...], dtype=torch.float32)

    outputs = u_nms.non_max_suppression(
        pred,
        conf_thres=float(min_score),
        iou_thres=float(iou_thresh),
        classes=None,
        agnostic=bool(agnostic),
        max_det=int(max_det),
        nc=0,
        end2end=False,
        rotated=False,
        return_idxs=False,
    )
    if not outputs or outputs[0] is None or len(outputs[0]) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    out = outputs[0].detach().cpu().numpy()
    boxes = out[:, :4]
    scores = out[:, 4]
    class_ids = out[:, 5].astype(int)
    return boxes, scores, class_ids


def _preprocess_pil(*, image_path: str, input_size: int, np):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image loading") from exc

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    letterbox = compute_letterbox(orig_w=int(orig_w), orig_h=int(orig_h), input_size=int(input_size))
    left = int(letterbox.pad_x)
    top = int(letterbox.pad_y)

    if (orig_w, orig_h) != (letterbox.new_w, letterbox.new_h):
        img = img.resize((int(letterbox.new_w), int(letterbox.new_h)), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (int(input_size), int(input_size)), color=(114, 114, 114))
    canvas.paste(img, (int(left), int(top)))

    x = np.asarray(canvas).astype(np.float32) / 255.0  # (H,W,C)
    x = np.transpose(x, (2, 0, 1))  # (C,H,W)
    x = np.expand_dims(x, axis=0)  # (1,C,H,W)
    return x, (int(orig_w), int(orig_h)), letterbox


def export_predictions_onnxrt(
    *,
    dataset_root: str | Path,
    split: str | None = None,
    max_images: int | None = None,
    onnx: str | Path | None,
    input_name: str = "images",
    boxes_output: str = "boxes",
    scores_output: str = "scores",
    class_output: str | None = None,
    combined_output: str | None = None,
    combined_format: str = "xyxy_score_class",
    raw_output: str | None = None,
    raw_format: str = "yolo_84",
    raw_postprocess: str = "native",
    boxes_format: str = "xyxy",
    boxes_scale: str = "norm",
    min_score: float = 0.001,
    topk: int = 300,
    nms_iou: float = 0.7,
    agnostic_nms: bool = False,
    imgsz: int = 640,
    dry_run: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    manifest = build_manifest(dataset_root, split=split)
    records = list(manifest.get("images") or [])
    if max_images is not None:
        records = records[: max(0, int(max_images))]

    if dry_run:
        predictions = []
        for r in records:
            if not isinstance(r, dict):
                continue
            image = r.get("image")
            if not isinstance(image, str) or not image:
                continue
            predictions.append({"image": image, "detections": []})
        validate_predictions_entries(predictions, strict=bool(strict))
        config_fingerprint = {
            "exporter": "onnxruntime",
            "dry_run": True,
            "imgsz": int(imgsz),
            "dataset": str(dataset_root),
            "split": split,
            "max_images": max_images,
        }
        meta = {
            "schema_version": 1,
            "timestamp": _now_utc(),
            "generator": "yolozu onnxrt export --dry-run",
            "images": int(len(predictions)),
            "run": {"config_fingerprint": config_fingerprint, "config_hash": _sha256_json(config_fingerprint)},
        }
        return {"schema_version": 1, "predictions": predictions, "meta": meta}

    if not onnx:
        raise ValueError("--onnx is required unless --dry-run is set")

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for onnxruntime exporter") from exc

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required (install with `pip install 'yolozu[onnxrt]'`)") from exc

    model_path = Path(onnx)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"onnx model not found: {model_path}")

    providers = None
    try:
        providers = ort.get_available_providers()
    except (AttributeError, RuntimeError, TypeError):
        providers = None

    sess = ort.InferenceSession(str(model_path), providers=providers)

    predictions: list[dict[str, Any]] = []
    input_size = int(imgsz)

    for record in records:
        if not isinstance(record, dict):
            continue
        image_path = record.get("image")
        if not isinstance(image_path, str) or not image_path:
            continue

        x, (orig_w, orig_h), letterbox = _preprocess_pil(image_path=image_path, input_size=input_size, np=np)

        raw_outputs = sess.run(None, {str(input_name): x})
        names = [o.name for o in sess.get_outputs()]
        outputs = dict(zip(names, raw_outputs))

        if combined_output:
            combined = outputs.get(str(combined_output))
            if combined is None:
                raise ValueError(f"missing combined output: {combined_output}")
            boxes_t, scores_t, class_t = _split_combined_output(np.asarray(combined), fmt=str(combined_format))
        elif raw_output:
            raw = outputs.get(str(raw_output))
            if raw is None:
                raise ValueError(f"missing raw output: {raw_output}")
            if str(raw_postprocess) == "ultralytics":
                boxes_t, scores_t, class_t = _decode_raw_ultralytics(
                    np.asarray(raw),
                    min_score=float(min_score),
                    iou_thresh=float(nms_iou),
                    max_det=int(topk),
                    agnostic=bool(agnostic_nms),
                    np=np,
                )
            else:
                boxes_t, scores_t, class_t = _decode_raw_output(
                    np.asarray(raw),
                    min_score=float(min_score),
                    iou_thresh=float(nms_iou),
                    max_det=int(topk),
                    agnostic=bool(agnostic_nms),
                    np=np,
                )
        else:
            boxes_t, scores_t, class_t = _resolve_boxes_and_scores(
                outputs, boxes_key=str(boxes_output), scores_key=str(scores_output), class_key=class_output
            )

        boxes = np.asarray(boxes_t)
        scores = np.asarray(scores_t)
        class_ids = None if class_t is None else np.asarray(class_t)

        if scores.ndim == 2:
            class_ids = np.argmax(scores, axis=1)
            scores = np.max(scores, axis=1)
        elif scores.ndim != 1:
            raise ValueError(f"unsupported scores shape: {scores.shape}")

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"unsupported boxes shape: {boxes.shape}")
        if class_ids is None:
            raise ValueError("class ids missing: provide --class-output or use (N,C) scores")

        scores = scores.astype(float)
        class_ids = class_ids.astype(int)

        if raw_output:
            idx = list(range(len(scores)))
        else:
            idx = [i for i, s in enumerate(scores.tolist()) if float(s) >= float(min_score)]
            idx.sort(key=lambda i: float(scores[i]), reverse=True)
            idx = idx[: max(0, int(topk))]

        detections: list[dict[str, Any]] = []
        use_ultra_scale = bool(raw_output and str(raw_postprocess) == "ultralytics")
        if use_ultra_scale:
            try:
                import torch  # type: ignore
                from ultralytics.utils import ops as u_ops  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("ultralytics + torch are required for raw-postprocess=ultralytics") from exc

        for i in idx:
            b = boxes[i].tolist()
            if str(boxes_format) != "xyxy":
                raise ValueError("only --boxes-format xyxy is supported")

            if raw_output and str(boxes_scale) != "abs":
                raise ValueError("--raw-output expects --boxes-scale abs (input-space pixels)")

            if str(boxes_scale) == "norm":
                x1, y1, x2, y2 = (
                    float(b[0]) * input_size,
                    float(b[1]) * input_size,
                    float(b[2]) * input_size,
                    float(b[3]) * input_size,
                )
            else:
                x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

            if use_ultra_scale:
                scaled = u_ops.scale_boxes(
                    (input_size, input_size),
                    torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
                    (orig_h, orig_w),
                )
                sx1, sy1, sx2, sy2 = scaled[0].tolist()
                orig_xyxy = (float(sx1), float(sy1), float(sx2), float(sy2))
            else:
                orig_xyxy = input_xyxy_to_orig_xyxy(
                    (x1, y1, x2, y2), letterbox=letterbox, orig_w=int(orig_w), orig_h=int(orig_h)
                )
            bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=int(orig_w), orig_h=int(orig_h))
            detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})

        predictions.append({"image": image_path, "detections": detections})

    validate_predictions_entries(predictions, strict=bool(strict))

    config_fingerprint = {
        "exporter": "onnxruntime",
        "imgsz": int(imgsz),
        "dataset": str(dataset_root),
        "split": split,
        "max_images": max_images,
        "onnx": str(model_path),
        "onnx_sha256": _sha256(model_path),
        "input_name": str(input_name),
        "boxes_output": str(boxes_output),
        "scores_output": str(scores_output),
        "class_output": class_output,
        "combined_output": combined_output,
        "combined_format": str(combined_format),
        "raw_output": raw_output,
        "raw_format": str(raw_format),
        "raw_postprocess": str(raw_postprocess),
        "boxes_format": str(boxes_format),
        "boxes_scale": str(boxes_scale),
        "min_score": float(min_score),
        "topk": int(topk),
        "nms_iou": float(nms_iou),
        "agnostic_nms": bool(agnostic_nms),
        "providers": providers,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    meta = {
        "schema_version": 1,
        "timestamp": _now_utc(),
        "generator": "yolozu onnxrt export",
        "images": int(len(predictions)),
        "run": {"config_fingerprint": config_fingerprint, "config_hash": _sha256_json(config_fingerprint)},
    }
    return {"schema_version": 1, "predictions": predictions, "meta": meta}


def write_predictions_json(*, output: str | Path, payload: dict[str, Any], force: bool = False) -> Path:
    from yolozu.export import write_predictions_json as _write

    return _write(output=output, payload=payload, force=bool(force))
