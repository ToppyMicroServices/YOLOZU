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
from typing import Any, Literal

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
        prog="export_predictions_opencv_dnn_rtdetr",
        description="Run OpenCV-DNN (cv2.dnn) inference on an RT-DETR ONNX model and export YOLOZU predictions JSON (no NMS).",
    )
    p.add_argument("--dataset", required=True, help="YOLO-format COCO root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--onnx", default=None, help="Path to ONNX model (required unless --dry-run).")

    p.add_argument("--imgsz", type=int, default=640, help="Square input size (default: 640).")
    p.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Use letterbox preprocessing to keep aspect ratio (default: off = stretch resize).",
    )
    p.add_argument(
        "--letterbox-fill",
        default="114,114,114",
        help="Letterbox fill color as 'R,G,B' (default: 114,114,114).",
    )
    p.add_argument(
        "--input-color",
        choices=("RGB", "BGR"),
        default="RGB",
        help="Model input color order (default: RGB).",
    )
    p.add_argument("--scale", type=float, default=1.0 / 255.0, help="Input scaling factor (default: 1/255).")
    p.add_argument("--mean", default=None, help="Optional mean as 'R,G,B' (applied after scaling).")
    p.add_argument("--std", default=None, help="Optional std as 'R,G,B' (applied after mean).")

    p.add_argument(
        "--outputs",
        default=None,
        help="Comma-separated output layer names. Default: OpenCV unconnected outputs.",
    )
    p.add_argument(
        "--boxes-output",
        default=None,
        help="Output name for boxes tensor (auto-detected when omitted).",
    )
    p.add_argument(
        "--logits-output",
        default=None,
        help="Output name for class logits/prob tensor (auto-detected when omitted).",
    )
    p.add_argument("--labels-output", default=None, help="Output name for class_id tensor (optional).")
    p.add_argument("--scores-output", default=None, help="Output name for score tensor (optional).")
    p.add_argument("--print-outputs", action="store_true", help="Print output names + shapes and exit.")

    p.add_argument(
        "--boxes-format",
        choices=("cxcywh", "xyxy"),
        default="cxcywh",
        help="Boxes coordinate format produced by the model (default: cxcywh).",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("norm", "abs"),
        default="norm",
        help="Boxes coordinate scale in model output (default: norm in 0..1 relative to input).",
    )
    p.add_argument(
        "--scores-activation",
        choices=("softmax", "sigmoid", "none"),
        default="softmax",
        help="Activation to apply to logits output (default: softmax).",
    )
    p.add_argument(
        "--background-class",
        choices=("none", "last", "zero"),
        default="last",
        help="Which class index represents background (ignored when decoding logits). (default: last)",
    )
    p.add_argument("--score-thr", type=float, default=0.01, help="Score threshold (default: 0.01).")
    p.add_argument("--topk", type=int, default=300, help="Max detections per image (default: 300).")

    p.add_argument(
        "--dnn-backend",
        default="opencv",
        help="OpenCV DNN backend selector (default: opencv).",
    )
    p.add_argument("--dnn-target", default="cpu", help="OpenCV DNN target selector (default: cpu).")

    p.add_argument("--strict", action="store_true", help="Strictly validate output predictions JSON.")
    p.add_argument("--output", default="reports/pred_rtdetr_opencv_dnn.json", help="Where to write predictions JSON.")
    p.add_argument("--meta-output", default=None, help="Where to write metadata JSON (default: <output>.meta.json).")
    p.add_argument("--dump-io", default=None, help="Optional JSON path to dump input/output tensor IO summary.")
    p.add_argument("--dry-run", action="store_true", help="Write outputs without importing/running OpenCV DNN.")
    return p.parse_args(argv)


def _parse_rgb_triplet(value: str, *, where: str) -> list[float]:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{where} must be 'R,G,B' (got {value!r})")
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception as exc:
            raise SystemExit(f"{where} must be numeric (got {value!r})") from exc
    return out


def _normalize_boxes_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        if a.shape[1] == 4:
            return a
        if a.shape[0] == 4 and a.shape[1] != 4:
            return np.transpose(a, (1, 0))
    raise ValueError(f"unsupported boxes shape (expected Nx4): {tuple(np.asarray(arr).shape)}")


def _normalize_logits_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2:
        # Heuristic: prefer (N,C) with C reasonably small.
        if a.shape[0] >= 1 and a.shape[1] >= 2:
            return a
    raise ValueError(f"unsupported logits shape (expected NxC): {tuple(np.asarray(arr).shape)}")


def _normalize_vector(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 1:
        return a
    raise ValueError(f"unsupported vector shape: {tuple(np.asarray(arr).shape)}")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    s = e.sum(axis=1, keepdims=True)
    return np.where(s > 0.0, e / s, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


def _decode_logits(
    logits: np.ndarray,
    *,
    activation: Literal["softmax", "sigmoid", "none"],
    background_class: Literal["none", "last", "zero"],
) -> tuple[np.ndarray, np.ndarray]:
    if activation == "softmax":
        probs = _softmax(logits)
    elif activation == "sigmoid":
        probs = _sigmoid(logits)
    else:
        probs = logits.astype(np.float32, copy=False)

    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError(f"invalid probs shape: {tuple(probs.shape)}")

    scores = probs
    if background_class != "none" and scores.shape[1] >= 2:
        scores = scores.copy()
        if background_class == "last":
            scores[:, -1] = -1.0
        elif background_class == "zero":
            scores[:, 0] = -1.0

    class_ids = scores.argmax(axis=1).astype(np.int64)
    best = scores.max(axis=1).astype(np.float32)
    return class_ids, best


def _cxcywh_to_xyxy(cx: np.ndarray, cy: np.ndarray, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def _preprocess_entry(*, args: argparse.Namespace) -> dict[str, Any]:
    method = "letterbox" if bool(args.keep_aspect) else "resize"
    fill = _parse_rgb_triplet(str(args.letterbox_fill), where="--letterbox-fill")
    out: dict[str, Any] = {
        "method": method,
        "input_color": str(args.input_color),
        "normalize": "0_1" if abs(float(args.scale) - (1.0 / 255.0)) < 1e-9 else "custom",
        "resize_interp": "linear",
        "input_size": {"width": int(args.imgsz), "height": int(args.imgsz)},
    }
    if bool(args.keep_aspect):
        out["letterbox_fill"] = [int(round(v)) for v in fill]
    if args.mean is not None:
        out["mean"] = _parse_rgb_triplet(str(args.mean), where="--mean")
    if args.std is not None:
        out["std"] = _parse_rgb_triplet(str(args.std), where="--std")
    out["keep_aspect"] = bool(args.keep_aspect)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if np is None and not bool(args.dry_run):
        raise SystemExit("numpy is required for OpenCV-DNN RT-DETR exporter (or use --dry-run)")

    dataset_root = _resolve(str(args.dataset))
    manifest = build_manifest(dataset_root, split=str(args.split) if args.split else None)
    records = list(manifest["images"])
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    output_path = _resolve(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _resolve(str(args.meta_output)) if args.meta_output else output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    preprocess = _preprocess_entry(args=args)

    meta: dict[str, Any] = {
        "timestamp": _now_utc(),
        "exporter": "opencv_dnn_rtdetr",
        "dataset": str(args.dataset),
        "split": manifest["split"],
        "max_images": args.max_images,
        "onnx": args.onnx,
        "export_settings": {
            "imgsz": int(args.imgsz),
            "bbox_format": "cxcywh_norm",
            "score_threshold": float(args.score_thr),
            "preprocess": preprocess,
        },
        "outputs": {
            "names": args.outputs,
            "boxes_output": args.boxes_output,
            "logits_output": args.logits_output,
            "labels_output": args.labels_output,
            "scores_output": args.scores_output,
        },
        "decode": {
            "boxes_format": str(args.boxes_format),
            "boxes_scale": str(args.boxes_scale),
            "scores_activation": str(args.scores_activation),
            "background_class": str(args.background_class),
            "topk": int(args.topk),
        },
        "dnn": {"backend": str(args.dnn_backend), "target": str(args.dnn_target)},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
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
        predictions: list[dict[str, Any]] = []
        for rec in records:
            predictions.append(
                {
                    "image": str(rec["image"]),
                    "detections": [],
                    "preprocess": preprocess,
                    "image_size": {"width": int(args.imgsz), "height": int(args.imgsz)},
                }
            )
        validate_predictions_entries(predictions, strict=bool(args.strict))
        output_path.write_text(json.dumps({"predictions": predictions}, indent=2, sort_keys=True), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        if args.dump_io:
            dump_path = _resolve(str(args.dump_io))
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(
                json.dumps(
                    {
                        "input": {"name": "images", "shape": [1, 3, int(args.imgsz), int(args.imgsz)], "dtype": "float32"},
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
    try:
        b = str(args.dnn_backend).lower()
        if b in {"opencv", "default"}:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        elif b == "cuda":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        elif b == "openvino" and hasattr(cv2.dnn, "DNN_BACKEND_INFERENCE_ENGINE"):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    except cv2.error:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    try:
        t = str(args.dnn_target).lower()
        if t in {"cpu", "default"}:
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

    if args.outputs:
        output_names = [s.strip() for s in str(args.outputs).split(",") if s.strip()]
    else:
        output_names = list(net.getUnconnectedOutLayersNames())
    if not output_names:
        raise SystemExit("OpenCV net has no output names; pass --outputs explicitly")
    meta["output_names_effective"] = output_names

    # Probe output shapes if requested.
    if bool(args.print_outputs):
        dummy = np.zeros((1, 3, int(args.imgsz), int(args.imgsz)), dtype=np.float32)
        net.setInput(dummy)
        outs = net.forward(output_names)
        for name, out in zip(output_names, outs):
            print(f"{name}: shape={tuple(np.asarray(out).shape)} dtype={np.asarray(out).dtype}")
        return 0

    # Auto-detect outputs by probing a forward pass on a dummy tensor.
    dummy = np.zeros((1, 3, int(args.imgsz), int(args.imgsz)), dtype=np.float32)
    net.setInput(dummy)
    probe_outs = {name: out for name, out in zip(output_names, net.forward(output_names))}
    io_probe = {
        "input": {"name": "images", "shape": [1, 3, int(args.imgsz), int(args.imgsz)], "dtype": "float32"},
        "outputs": [
            {"name": str(name), "shape": list(np.asarray(out).shape), "dtype": str(np.asarray(out).dtype)}
            for name, out in probe_outs.items()
        ],
    }

    def _auto_pick_boxes() -> str:
        for name, out in probe_outs.items():
            try:
                b = _normalize_boxes_2d(np.asarray(out))
            except Exception:
                continue
            if b.shape[1] == 4:
                return name
        raise SystemExit("failed to auto-detect boxes output; pass --boxes-output")

    def _auto_pick_logits(exclude: set[str]) -> str | None:
        for name, out in probe_outs.items():
            if name in exclude:
                continue
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] >= 2 and arr.shape[-1] != 4:
                return name
        for name, out in probe_outs.items():
            if name in exclude:
                continue
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[-1] >= 2 and arr.shape[-1] != 4:
                return name
        return None

    boxes_name = str(args.boxes_output) if args.boxes_output else None
    if boxes_name is None:
        boxes_name = _auto_pick_boxes()
    used = {boxes_name} if boxes_name else set()

    logits_name = str(args.logits_output) if args.logits_output else None
    if logits_name is None:
        logits_name = _auto_pick_logits(used)
    if logits_name:
        used.add(logits_name)

    labels_name = str(args.labels_output) if args.labels_output else None
    scores_name = str(args.scores_output) if args.scores_output else None

    meta["outputs_effective"] = {"boxes": boxes_name, "logits": logits_name, "labels": labels_name, "scores": scores_name}

    # Prepare per-image inference loop.
    imgsz = int(args.imgsz)
    keep_aspect = bool(args.keep_aspect)
    fill = [int(round(v)) for v in _parse_rgb_triplet(str(args.letterbox_fill), where="--letterbox-fill")]
    input_color = str(args.input_color)
    mean = _parse_rgb_triplet(str(args.mean), where="--mean") if args.mean is not None else None
    std = _parse_rgb_triplet(str(args.std), where="--std") if args.std is not None else None
    scale = float(args.scale)

    def _make_blob(image_path: str):
        orig_w, orig_h = get_image_size(image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"failed to load image: {image_path}")

        if input_color == "RGB":
            img = img[..., ::-1]  # BGR -> RGB

        letterbox = None
        if keep_aspect:
            letterbox = compute_letterbox(orig_w=int(orig_w), orig_h=int(orig_h), input_size=imgsz)
            pad_w = float(imgsz) - float(letterbox.new_w)
            pad_h = float(imgsz) - float(letterbox.new_h)
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
                value=tuple(fill),
            )
        else:
            if (img.shape[1], img.shape[0]) != (imgsz, imgsz):
                img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        x = img.astype(np.float32) * float(scale)  # (H,W,C) in input_color order
        if mean is not None:
            x = x - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        if std is not None:
            x = x / np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        x = np.transpose(x, (2, 0, 1))  # (C,H,W)
        x = np.expand_dims(x, axis=0)  # (1,C,H,W)
        return x, (int(orig_w), int(orig_h)), letterbox

    predictions: list[dict[str, Any]] = []

    for rec in records:
        image_path = str(rec["image"])
        blob, (orig_w, orig_h), letterbox = _make_blob(image_path)

        net.setInput(blob)
        # Forward using the same requested name list to keep ordering consistent.
        outs = net.forward(output_names)
        per = {name: out for name, out in zip(output_names, outs)}

        boxes_arr = _normalize_boxes_2d(np.asarray(per[str(boxes_name)]))
        if str(args.boxes_scale) == "norm":
            boxes_arr = boxes_arr * float(imgsz)

        if labels_name is not None and scores_name is not None:
            labels_arr = _normalize_vector(np.asarray(per[labels_name])).astype(np.int64)
            scores_arr = _normalize_vector(np.asarray(per[scores_name])).astype(np.float32)
        elif logits_name is not None:
            logits_arr = _normalize_logits_2d(np.asarray(per[logits_name]))
            labels_arr, scores_arr = _decode_logits(
                logits_arr,
                activation=str(args.scores_activation),
                background_class=str(args.background_class),
            )
        else:
            raise SystemExit("need either --logits-output (or auto logits) OR both --labels-output and --scores-output")

        if boxes_arr.shape[0] != labels_arr.shape[0] or boxes_arr.shape[0] != scores_arr.shape[0]:
            raise SystemExit(
                f"mismatched output lengths: boxes={boxes_arr.shape[0]} labels={labels_arr.shape[0]} scores={scores_arr.shape[0]}"
            )

        score_thr = float(args.score_thr)
        keep = scores_arr >= score_thr
        boxes_keep = boxes_arr[keep]
        labels_keep = labels_arr[keep]
        scores_keep = scores_arr[keep]

        if boxes_keep.size == 0:
            dets: list[dict[str, Any]] = []
        else:
            # Sort by score desc and keep topk.
            order = np.argsort(scores_keep)[::-1]
            if int(args.topk) > 0:
                order = order[: int(args.topk)]
            boxes_keep = boxes_keep[order]
            labels_keep = labels_keep[order]
            scores_keep = scores_keep[order]

            dets = []
            for i in range(int(boxes_keep.shape[0])):
                b = boxes_keep[i]
                if str(args.boxes_format) == "xyxy":
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    xyxy_in = (x1, y1, x2, y2)
                else:
                    cx, cy, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    xyxy = _cxcywh_to_xyxy(
                        np.asarray([cx], dtype=np.float32),
                        np.asarray([cy], dtype=np.float32),
                        np.asarray([w], dtype=np.float32),
                        np.asarray([h], dtype=np.float32),
                    )[0]
                    xyxy_in = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

                if letterbox is not None:
                    orig_xyxy = input_xyxy_to_orig_xyxy(xyxy_in, letterbox=letterbox, orig_w=orig_w, orig_h=orig_h)
                else:
                    # Stretch-resize mapping from input space (imgsz x imgsz) back to original.
                    x1, y1, x2, y2 = xyxy_in
                    sx = float(orig_w) / float(imgsz)
                    sy = float(orig_h) / float(imgsz)
                    orig_xyxy = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

                bbox = orig_xyxy_to_cxcywh_norm(orig_xyxy, orig_w=orig_w, orig_h=orig_h)
                dets.append({"class_id": int(labels_keep[i]), "score": float(scores_keep[i]), "bbox": bbox})

        predictions.append(
            {
                "image": image_path,
                "detections": dets,
                "preprocess": preprocess,
                "image_size": {"width": imgsz, "height": imgsz},
            }
        )

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
