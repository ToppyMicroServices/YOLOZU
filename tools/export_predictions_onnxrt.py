import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest
from yolozu.image_size import get_image_size
from yolozu.letterbox import compute_letterbox, input_xyxy_to_orig_xyxy, orig_xyxy_to_cxcywh_norm
from yolozu.predictions import validate_predictions_entries


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    p.add_argument("--input-name", default="images", help="ONNX input name (default: images).")
    p.add_argument("--boxes-output", default="boxes", help="Output name for boxes tensor (default: boxes).")
    p.add_argument("--scores-output", default="scores", help="Output name for scores tensor (default: scores).")
    p.add_argument("--class-output", default=None, help="Optional output name for class_id tensor (default: none).")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional single output name with (N,6) or (1,N,6) entries [x1,y1,x2,y2,score,class_id].",
    )
    p.add_argument(
        "--combined-format",
        choices=("xyxy_score_class",),
        default="xyxy_score_class",
        help="Layout for --combined-output (default: xyxy_score_class).",
    )
    p.add_argument(
        "--raw-output",
        default=None,
        help="Optional single output name with raw head output (e.g., 1x84x8400) to decode + NMS.",
    )
    p.add_argument(
        "--raw-format",
        choices=("yolo_84",),
        default="yolo_84",
        help="Layout for --raw-output (default: yolo_84).",
    )
    p.add_argument(
        "--raw-postprocess",
        choices=("native", "ultralytics"),
        default="native",
        help="Postprocess for --raw-output (default: native).",
    )
    p.add_argument(
        "--boxes-format",
        choices=("xyxy",),
        default="xyxy",
        help="Box layout produced by the model in input-image space (default: xyxy).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="norm",
        help="Whether boxes are in pixels (abs) or normalized [0,1] (norm) wrt input_size (default: norm).",
    )
    p.add_argument("--min-score", type=float, default=0.001, help="Score threshold (no NMS).")
    p.add_argument("--topk", type=int, default=300, help="Keep top-K detections per image (no NMS).")
    p.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold for NMS when decoding raw output.")
    p.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS when decoding raw output.",
    )
    p.add_argument("--output", default="reports/predictions_onnxrt.json", help="Where to write predictions JSON.")
    p.add_argument("--wrap", action="store_true", help="Wrap as {predictions:[...], meta:{...}}.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-correct JSON without running inference.")
    p.add_argument("--strict", action="store_true", help="Strict prediction schema validation before writing.")
    return p.parse_args(argv)


def _resolve_onnx_input_size(input_shape: object, *, default_size: int = 640) -> tuple[int, str | None]:
    def _dim_to_int(value: object) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    shape = list(input_shape) if isinstance(input_shape, (list, tuple)) else []
    model_h = _dim_to_int(shape[2]) if len(shape) >= 3 else None
    model_w = _dim_to_int(shape[3]) if len(shape) >= 4 else None

    if model_h is None or model_w is None:
        return int(default_size), None
    if model_h != model_w:
        raise SystemExit(f"ONNX input is non-square ({model_h}x{model_w}); export_predictions_onnxrt expects square. Use correct input.")
    warning = None
    if int(model_h) != int(default_size):
        warning = f"[warn] using ONNX-fixed input size {model_h}x{model_h} (override not supported for this model)"
    return int(model_h), warning


def _validate_decode_contract(*, exporter_name: str, boxes_format: str, boxes_scale: str, raw_output: str | None) -> None:
    if boxes_format != "xyxy":
        raise ValueError(f"only --boxes-format xyxy is supported in this {exporter_name}")
    if raw_output and boxes_scale != "abs":
        raise ValueError("--raw-output expects --boxes-scale abs (input-space pixels)")


def _now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _xywh_to_xyxy(boxes):
    x, y, w, h = boxes.T
    x1 = x - (w / 2.0)
    y1 = y - (h / 2.0)
    x2 = x + (w / 2.0)
    y2 = y + (h / 2.0)
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy_one_to_many(box, boxes):
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


def _nms(boxes, scores, *, iou_thresh: float, max_det: int):
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    try:
        import torch  # type: ignore

        try:
            from ultralytics.utils.nms import TorchNMS  # type: ignore

            keep = TorchNMS.nms(
                torch.as_tensor(boxes, dtype=torch.float32),
                torch.as_tensor(scores, dtype=torch.float32),
                float(iou_thresh),
            )
            keep = keep[: int(max_det)].cpu().numpy().astype(np.int64)
            return keep
        except Exception:
            try:
                import torchvision  # type: ignore

                keep = torchvision.ops.nms(
                    torch.as_tensor(boxes, dtype=torch.float32),
                    torch.as_tensor(scores, dtype=torch.float32),
                    float(iou_thresh),
                )
                keep = keep[: int(max_det)].cpu().numpy().astype(np.int64)
                return keep
            except Exception:
                # Fall back to the local NMS implementation when torchvision NMS is unavailable.
                pass
    except Exception:
        # Torch/TorchVision are optional for the fast-path NMS helper.
        pass

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= float(iou_thresh)]
    return np.array(keep, dtype=np.int64)


def _decode_raw_output(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
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

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
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

    keep_idx = _nms(boxes_nms, scores, iou_thresh=iou_thresh, max_det=max_det)
    return boxes_xyxy[keep_idx], scores[keep_idx], class_ids[keep_idx]


def _decode_raw_ultralytics(
    raw,
    *,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
):
    try:
        import torch  # type: ignore
        from ultralytics.utils import nms as u_nms  # type: ignore
    except Exception as exc:  # pragma: no cover
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


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_size = 640  # default fallback (YOLO26 protocol)
    dataset_root = repo_root / args.dataset
    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: args.max_images]

    predictions = []
    if args.dry_run:
        for record in records:
            predictions.append({"image": record["image"], "detections": []})
        validate_predictions_entries(predictions, strict=args.strict)
        payload = (
            {
                "predictions": predictions,
                "meta": {
                    "timestamp": _now_utc(),
                    "exporter": "onnxruntime",
                    "dry_run": True,
                    "protocol_id": "yolo26",
                    "imgsz": input_size,
                    "dataset": args.dataset,
                    "split": manifest["split"],
                    "max_images": args.max_images,
                    "note": "dry-run mode: no inference performed",
                },
            }
            if args.wrap
            else predictions
        )
        out_path = repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(out_path)
        return

    if not args.onnx:
        raise SystemExit("--onnx is required unless --dry-run is set")

    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for onnxruntime exporter")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required (pip install onnxruntime)") from exc

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for image loading (pip install opencv-python)") from exc

    model_path = Path(args.onnx)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not model_path.exists():
        raise SystemExit(f"onnx model not found: {model_path}")

    providers = None
    try:
        providers = ort.get_available_providers()
    except Exception:
        providers = None

    sess = ort.InferenceSession(str(model_path), providers=providers)

    # Derive fixed input size from ONNX when static.
    input_size, shape_warning = _resolve_onnx_input_size(sess.get_inputs()[0].shape, default_size=640)
    if shape_warning:
        print(shape_warning, file=sys.stderr)

    _validate_decode_contract(
        exporter_name="skeleton",
        boxes_format=str(args.boxes_format),
        boxes_scale=str(args.boxes_scale),
        raw_output=args.raw_output,
    )

    def preprocess(image_path: str):
        w, h = get_image_size(image_path)
        letterbox = compute_letterbox(orig_w=w, orig_h=h, input_size=input_size)

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"failed to load image: {image_path}")

        pad_w = float(input_size) - float(letterbox.new_w)
        pad_h = float(input_size) - float(letterbox.new_h)
        pad_x = pad_w / 2.0
        pad_y = pad_h / 2.0
        left = int(letterbox.pad_x)
        top = int(letterbox.pad_y)
        right = int(round(pad_x + 0.1))
        bottom = int(round(pad_y + 0.1))

        if (img.shape[1], img.shape[0]) != (letterbox.new_w, letterbox.new_h):
            img = cv2.resize(img, (letterbox.new_w, letterbox.new_h), interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        img = img[..., ::-1]  # BGR to RGB

        x = img.astype(np.float32) / 255.0  # (H,W,C)
        x = np.transpose(x, (2, 0, 1))  # (C,H,W)
        x = np.expand_dims(x, axis=0)  # (1,C,H,W)
        return x, (w, h), letterbox

    for record in records:
        image_path = record["image"]
        x, (orig_w, orig_h), letterbox = preprocess(image_path)

        raw_outputs = sess.run(None, {args.input_name: x})
        names = [o.name for o in sess.get_outputs()]
        outputs = dict(zip(names, raw_outputs))
        if args.combined_output:
            combined = outputs.get(args.combined_output)
            if combined is None:
                raise ValueError(f"missing combined output: {args.combined_output}")
            boxes_t, scores_t, class_t = _split_combined_output(
                np.asarray(combined), fmt=str(args.combined_format)
            )
        elif args.raw_output:
            raw = outputs.get(args.raw_output)
            if raw is None:
                raise ValueError(f"missing raw output: {args.raw_output}")
            if args.raw_postprocess == "ultralytics":
                boxes_t, scores_t, class_t = _decode_raw_ultralytics(
                    np.asarray(raw),
                    min_score=float(args.min_score),
                    iou_thresh=float(args.nms_iou),
                    max_det=int(args.topk),
                    agnostic=bool(args.agnostic_nms),
                )
            else:
                boxes_t, scores_t, class_t = _decode_raw_output(
                    np.asarray(raw),
                    min_score=float(args.min_score),
                    iou_thresh=float(args.nms_iou),
                    max_det=int(args.topk),
                    agnostic=bool(args.agnostic_nms),
                )
        else:
            boxes_t, scores_t, class_t = _resolve_boxes_and_scores(
                outputs, boxes_key=args.boxes_output, scores_key=args.scores_output, class_key=args.class_output
            )

        boxes = np.asarray(boxes_t)
        scores = np.asarray(scores_t)
        class_ids = None if class_t is None else np.asarray(class_t)

        if scores.ndim == 2:
            # scores: (N, C) -> pick best class
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

        if args.raw_output:
            idx = list(range(len(scores)))
        else:
            idx = [i for i, s in enumerate(scores.tolist()) if float(s) >= float(args.min_score)]
            idx.sort(key=lambda i: float(scores[i]), reverse=True)
            idx = idx[: max(0, int(args.topk))]

        detections = []
        use_ultra_scale = bool(args.raw_output and args.raw_postprocess == "ultralytics")
        if use_ultra_scale:
            try:
                import torch  # type: ignore
                from ultralytics.utils import ops as u_ops  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("ultralytics + torch are required for raw-postprocess=ultralytics") from exc
        for i in idx:
            b = boxes[i].tolist()
            if args.boxes_scale == "norm":
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
                orig_xyxy = input_xyxy_to_orig_xyxy((x1, y1, x2, y2), letterbox=letterbox, orig_w=orig_w, orig_h=orig_h)
            bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)

            detections.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})

        predictions.append({"image": image_path, "detections": detections})

    validate_predictions_entries(predictions, strict=args.strict)

    meta = {
        "timestamp": _now_utc(),
        "exporter": "onnxruntime",
        "protocol_id": "yolo26",
        "imgsz": input_size,
        "dataset": args.dataset,
        "split": manifest["split"],
        "max_images": args.max_images,
        "onnx": str(model_path),
        "onnx_sha256": _sha256(model_path),
        "input_name": args.input_name,
        "boxes_output": args.boxes_output,
        "scores_output": args.scores_output,
        "class_output": args.class_output,
        "combined_output": args.combined_output,
        "combined_format": args.combined_format,
        "raw_output": args.raw_output,
        "raw_format": args.raw_format,
        "raw_postprocess": args.raw_postprocess,
        "boxes_format": args.boxes_format,
        "boxes_scale": args.boxes_scale,
        "min_score": args.min_score,
        "topk": args.topk,
        "nms_iou": args.nms_iou,
        "agnostic_nms": args.agnostic_nms,
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

    payload = {"predictions": predictions, "meta": meta} if args.wrap else predictions
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(out_path)


if __name__ == "__main__":
    main()
