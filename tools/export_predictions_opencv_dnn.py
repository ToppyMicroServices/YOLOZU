#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

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


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="export_predictions_opencv_dnn",
        description="Run OpenCV-DNN (cv2.dnn) inference on an ONNX model and export YOLOZU predictions JSON.",
    )
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")
    p.add_argument("--input-size", type=int, default=640, help="Square input size (letterbox) (default: 640).")
    p.add_argument(
        "--output-names",
        default=None,
        help="Comma-separated list of output layer names (default: OpenCV unconnected outputs).",
    )
    p.add_argument("--output-index", type=int, default=0, help="Which output to treat as the raw head (default: 0).")
    p.add_argument(
        "--raw-format",
        choices=("yolo_84", "yolo_85_obj"),
        default="yolo_84",
        help="Raw head format (default: yolo_84 for YOLOv8-style 1x84xN). Use yolo_85_obj for YOLOv5-style 85 with objectness.",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="abs",
        help="Whether raw boxes are in input pixels (abs) or 0..1 (norm). (default: abs)",
    )
    p.add_argument("--min-score", type=float, default=0.25, help="Min score filter before NMS (default: 0.25).")
    p.add_argument("--nms-iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45).")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300).")
    p.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS (default: class-wise).")
    p.add_argument("--swap-rb", action="store_true", help="Swap BGR↔RGB in preprocessing (default: off).")

    p.add_argument(
        "--dnn-backend",
        default=None,
        help="Optional OpenCV DNN backend (e.g. default, opencv, cuda). Applied when OpenCV supports it.",
    )
    p.add_argument(
        "--dnn-target",
        default=None,
        help="Optional OpenCV DNN target (e.g. cpu, cuda, cuda_fp16). Applied when OpenCV supports it.",
    )

    p.add_argument("--strict", action="store_true", help="Strictly validate output predictions JSON.")
    p.add_argument("--output", default="reports/pred_opencv_dnn.json", help="Where to write predictions JSON.")
    p.add_argument("--meta-output", default=None, help="Where to write metadata JSON (default: <output>.meta.json).")
    p.add_argument("--dump-io", default=None, help="Optional JSON path to dump input/output tensor IO summary.")
    p.add_argument("--dry-run", action="store_true", help="Write outputs without importing/running OpenCV DNN.")
    return p.parse_args(argv)


def _iou_xyxy_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, *, iou_thresh: float, max_det: int) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < int(max_det):
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= float(iou_thresh)]
    return np.array(keep, dtype=np.int64)


def _normalize_raw_yolo84(raw: np.ndarray) -> np.ndarray:
    # Accept common OpenCV-DNN outputs:
    # - (1,84,N) -> (N,84)
    # - (1,N,84) -> (N,84)
    # - (84,N)   -> (N,84)
    # - (N,84)   -> (N,84)
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        a = arr[0]
        if a.shape[0] == 84 and a.shape[1] != 84:
            return np.transpose(a, (1, 0))
        if a.shape[-1] == 84:
            return a.reshape(-1, 84)
    if arr.ndim == 2:
        if arr.shape[0] == 84 and arr.shape[1] != 84:
            return np.transpose(arr, (1, 0))
        if arr.shape[1] == 84:
            return arr
    raise ValueError(f"unsupported raw output shape for yolo_84: {tuple(arr.shape)}")


def _normalize_raw_yolo85_obj(raw: np.ndarray) -> np.ndarray:
    # Accept common YOLOv5-style OpenCV-DNN outputs:
    # - (1, N, 85) -> (N, 85)
    # - (N, 85)    -> (N, 85)
    # - (1, 85, N) -> (N, 85)
    # - (85, N)    -> (N, 85)
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        a = arr[0]
        if a.shape[0] == 85 and a.shape[1] != 85:
            return np.transpose(a, (1, 0))
        if a.shape[-1] == 85:
            return a.reshape(-1, 85)
    if arr.ndim == 2:
        if arr.shape[0] == 85 and arr.shape[1] != 85:
            return np.transpose(arr, (1, 0))
        if arr.shape[1] == 85:
            return arr
    raise ValueError(f"unsupported raw output shape for yolo_85_obj: {tuple(arr.shape)}")


def _decode_yolo84(
    raw: np.ndarray,
    *,
    boxes_scale: str,
    input_size: int,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _normalize_raw_yolo84(raw).astype(np.float32, copy=False)
    boxes_xywh = data[:, :4]
    class_scores = data[:, 4:]
    class_ids = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores.max(axis=1)

    keep = scores >= float(min_score)
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if boxes_scale == "norm":
        boxes_xywh = boxes_xywh * float(input_size)

    # xywh -> xyxy (in input space)
    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    if boxes_xyxy.size == 0:
        return boxes_xyxy, scores, class_ids

    if agnostic:
        idx = _nms(boxes_xyxy, scores, iou_thresh=float(iou_thresh), max_det=int(max_det))
    else:
        kept: list[int] = []
        for cid in np.unique(class_ids):
            mask = class_ids == cid
            idx_c = _nms(
                boxes_xyxy[mask],
                scores[mask],
                iou_thresh=float(iou_thresh),
                max_det=int(max_det),
            )
            # map back to global indices
            global_idx = np.nonzero(mask)[0][idx_c]
            kept.extend(int(i) for i in global_idx.tolist())
        kept = sorted(kept, key=lambda i: float(scores[i]), reverse=True)[: int(max_det)]
        idx = np.asarray(kept, dtype=np.int64)

    return boxes_xyxy[idx], scores[idx], class_ids[idx]


def _decode_yolo85_obj(
    raw: np.ndarray,
    *,
    boxes_scale: str,
    input_size: int,
    min_score: float,
    iou_thresh: float,
    max_det: int,
    agnostic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # YOLOv5-style: [cx, cy, w, h, obj, class_scores...]
    data = _normalize_raw_yolo85_obj(raw).astype(np.float32, copy=False)
    if data.shape[1] < 6:
        raise ValueError(f"yolo_85_obj expects at least 6 columns (got {data.shape[1]})")
    boxes_xywh = data[:, :4]
    obj = data[:, 4]
    class_scores = data[:, 5:]
    class_ids = class_scores.argmax(axis=1).astype(np.int64)
    cls = class_scores.max(axis=1)
    scores = obj * cls

    keep = scores >= float(min_score)
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if boxes_scale == "norm":
        boxes_xywh = boxes_xywh * float(input_size)

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    if boxes_xyxy.size == 0:
        return boxes_xyxy, scores, class_ids

    if agnostic:
        idx = _nms(boxes_xyxy, scores, iou_thresh=float(iou_thresh), max_det=int(max_det))
    else:
        kept: list[int] = []
        for cid in np.unique(class_ids):
            mask = class_ids == cid
            idx_c = _nms(
                boxes_xyxy[mask],
                scores[mask],
                iou_thresh=float(iou_thresh),
                max_det=int(max_det),
            )
            global_idx = np.nonzero(mask)[0][idx_c]
            kept.extend(int(i) for i in global_idx.tolist())
        kept = sorted(kept, key=lambda i: float(scores[i]), reverse=True)[: int(max_det)]
        idx = np.asarray(kept, dtype=np.int64)

    return boxes_xyxy[idx], scores[idx], class_ids[idx]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if np is None and not bool(args.dry_run):
        raise SystemExit("numpy is required for OpenCV-DNN exporter (or use --dry-run)")

    dataset_root = _resolve(str(args.dataset))
    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    output_path = _resolve(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _resolve(str(args.meta_output)) if args.meta_output else output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "timestamp": _now_utc(),
        "exporter": "opencv_dnn",
        "dataset": str(args.dataset),
        "split": manifest["split"],
        "max_images": args.max_images,
        "onnx": args.onnx,
        "input_size": int(args.input_size),
        "output_names": args.output_names,
        "output_index": int(args.output_index),
        "raw_format": args.raw_format,
        "boxes_scale": args.boxes_scale,
        "min_score": float(args.min_score),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "agnostic_nms": bool(args.agnostic_nms),
        "swap_rb": bool(args.swap_rb),
        "dnn_backend": args.dnn_backend,
        "dnn_target": args.dnn_target,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": sys.version,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "dry_run": bool(args.dry_run),
    }
    if args.dump_io:
        meta["dump_io"] = str(args.dump_io)

    if args.dry_run:
        predictions = [{"image": str(r["image"]), "detections": []} for r in records]
        validate_predictions_entries(predictions, strict=bool(args.strict))
        output_path.write_text(json.dumps({"predictions": predictions}, indent=2, sort_keys=True), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        if args.dump_io:
            dump_path = _resolve(str(args.dump_io))
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(
                json.dumps(
                    {
                        "input": {"name": "images", "shape": [1, 3, int(args.input_size), int(args.input_size)], "dtype": "float32"},
                        "outputs": [],
                        "dry_run": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        print(output_path)
        return 0

    if not args.onnx:
        raise SystemExit("--onnx is required unless --dry-run is set")

    onnx_path = _resolve(str(args.onnx))
    if not onnx_path.exists():
        raise SystemExit(f"onnx not found: {onnx_path}")
    meta["onnx"] = str(onnx_path)
    meta["onnx_sha256"] = _sha256(onnx_path)

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit("OpenCV is required (install opencv-python) unless --dry-run is set") from exc

    meta["cv2_version"] = getattr(cv2, "__version__", None)

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    if args.dnn_backend:
        b = str(args.dnn_backend).lower()
        try:
            if b == "opencv" or b == "default":
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            elif b == "cuda":
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            elif b == "openvino" and hasattr(cv2.dnn, "DNN_BACKEND_INFERENCE_ENGINE"):
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        except cv2.error:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if args.dnn_target:
        t = str(args.dnn_target).lower()
        try:
            if t == "cpu" or t == "default":
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            elif t == "cuda":
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            elif t == "cuda_fp16":
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            elif t == "opencl" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL"):
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            elif t == "opencl_fp16" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL_FP16"):
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        except cv2.error:
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    output_names: list[str]
    if args.output_names:
        output_names = [s.strip() for s in str(args.output_names).split(",") if s.strip()]
    else:
        output_names = list(net.getUnconnectedOutLayersNames())
    if not output_names:
        raise SystemExit("OpenCV net has no output names; pass --output-names explicitly")
    meta["output_names_effective"] = output_names
    input_size = int(args.input_size)
    io_probe: dict[str, Any] = {
        "input": {"name": "images", "shape": [1, 3, int(input_size), int(input_size)], "dtype": "float32"},
        "outputs": [],
    }
    try:
        probe_blob = np.zeros((1, 3, int(input_size), int(input_size)), dtype=np.float32)
        net.setInput(probe_blob)
        probe_outs = net.forward(output_names)
        io_probe["outputs"] = [
            {"name": str(name), "shape": list(np.asarray(out).shape), "dtype": str(np.asarray(out).dtype)}
            for name, out in zip(output_names, probe_outs)
        ]
    except Exception as exc:
        io_probe["probe_error"] = str(exc)

    predictions: list[dict[str, Any]] = []

    for record in records:
        image_path = str(record["image"])
        orig_w, orig_h = get_image_size(image_path)
        letterbox = compute_letterbox(orig_w=int(orig_w), orig_h=int(orig_h), input_size=input_size)

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

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=bool(args.swap_rb),
            crop=False,
        )
        net.setInput(blob)
        outs = net.forward(output_names)

        idx = int(args.output_index)
        if idx < 0 or idx >= len(outs):
            raise SystemExit(f"--output-index out of range: {idx} (have {len(outs)} outputs)")
        raw = outs[idx]

        if str(args.raw_format) == "yolo_85_obj":
            boxes_xyxy, scores, class_ids = _decode_yolo85_obj(
                raw,
                boxes_scale=str(args.boxes_scale),
                input_size=input_size,
                min_score=float(args.min_score),
                iou_thresh=float(args.nms_iou),
                max_det=int(args.max_det),
                agnostic=bool(args.agnostic_nms),
            )
        else:
            boxes_xyxy, scores, class_ids = _decode_yolo84(
                raw,
                boxes_scale=str(args.boxes_scale),
                input_size=input_size,
                min_score=float(args.min_score),
                iou_thresh=float(args.nms_iou),
                max_det=int(args.max_det),
                agnostic=bool(args.agnostic_nms),
            )

        dets: list[dict[str, Any]] = []
        for i in range(int(boxes_xyxy.shape[0])):
            x1, y1, x2, y2 = (float(boxes_xyxy[i, 0]), float(boxes_xyxy[i, 1]), float(boxes_xyxy[i, 2]), float(boxes_xyxy[i, 3]))
            orig_xyxy = input_xyxy_to_orig_xyxy((x1, y1, x2, y2), letterbox=letterbox, orig_w=int(orig_w), orig_h=int(orig_h))
            bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=int(orig_w), orig_h=int(orig_h))
            dets.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": bbox})
        predictions.append({"image": image_path, "detections": dets})

    validate_predictions_entries(predictions, strict=bool(args.strict))
    output_path.write_text(json.dumps({"predictions": predictions}, indent=2, sort_keys=True), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if args.dump_io:
        dump_path = _resolve(str(args.dump_io))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(io_probe, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
