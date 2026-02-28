#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="export_predictions_opencv_dnn_unified",
        description="Unified OpenCV-DNN exporter (YOLO/RT-DETR) to YOLOZU predictions.json.",
    )
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Dataset split (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    p.add_argument("--onnx", required=True, help="ONNX model path.")
    p.add_argument("--imgsz", type=int, default=640, help="Input size (default: 640).")
    p.add_argument(
        "--preprocess",
        choices=("yolo_letterbox_640", "rtdetr_resize_640", "rtdetr_letterbox_640"),
        default=None,
        help="Preprocessing preset (default: inferred from --decode).",
    )
    p.add_argument(
        "--decode",
        choices=("auto", "yolo_84", "yolo_85_obj", "rtdetr"),
        default="auto",
        help="Decode preset (default: auto).",
    )
    p.add_argument("--score-thr", type=float, default=0.01, help="Score threshold.")
    p.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU for YOLO decode.")
    p.add_argument("--topk", type=int, default=300, help="Max detections per image.")
    p.add_argument("--dnn-backend", default="opencv", help="OpenCV DNN backend selector.")
    p.add_argument("--dnn-target", default="cpu", help="OpenCV DNN target selector.")
    p.add_argument("--output", required=True, help="Output predictions JSON.")
    p.add_argument("--meta-output", default=None, help="Meta JSON output (default: <output>.meta.json).")
    p.add_argument("--dump-io", default=None, help="Write IO probe JSON (tensor names/shapes/dtypes).")
    p.add_argument("--strict", action="store_true", help="Strict predictions validation.")
    p.add_argument("--dry-run", action="store_true", help="Write schema-valid predictions without OpenCV runtime.")
    return p.parse_args(argv)


def _resolve_decode(args: argparse.Namespace) -> str:
    if str(args.decode) != "auto":
        return str(args.decode)
    if str(args.preprocess or "").startswith("rtdetr_"):
        return "rtdetr"
    return "yolo_84"


def _resolve_preprocess(args: argparse.Namespace, decode: str) -> str:
    if args.preprocess:
        return str(args.preprocess)
    return "rtdetr_resize_640" if decode == "rtdetr" else "yolo_letterbox_640"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    decode = _resolve_decode(args)
    preprocess = _resolve_preprocess(args, decode)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    meta_path = Path(args.meta_output) if args.meta_output else output_path.with_suffix(output_path.suffix + ".meta.json")
    if not meta_path.is_absolute():
        meta_path = (repo_root / meta_path).resolve()
    dump_io_path = Path(args.dump_io).resolve() if args.dump_io else None

    if decode == "rtdetr":
        cmd = [
            sys.executable,
            "tools/export_predictions_opencv_dnn_rtdetr.py",
            "--dataset",
            str(args.dataset),
            "--onnx",
            str(args.onnx),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--topk",
            str(int(args.topk)),
            "--dnn-backend",
            str(args.dnn_backend),
            "--dnn-target",
            str(args.dnn_target),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_path),
        ]
        if preprocess == "rtdetr_letterbox_640":
            cmd.append("--keep-aspect")
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if dump_io_path:
            cmd.extend(["--dump-io", str(dump_io_path)])
        if args.strict:
            cmd.append("--strict")
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd)
    else:
        raw_format = "yolo_85_obj" if decode == "yolo_85_obj" else "yolo_84"
        cmd = [
            sys.executable,
            "tools/export_predictions_opencv_dnn.py",
            "--dataset",
            str(args.dataset),
            "--onnx",
            str(args.onnx),
            "--input-size",
            str(int(args.imgsz)),
            "--raw-format",
            raw_format,
            "--min-score",
            str(float(args.score_thr)),
            "--nms-iou",
            str(float(args.nms_iou)),
            "--max-det",
            str(int(args.topk)),
            "--dnn-backend",
            str(args.dnn_backend),
            "--dnn-target",
            str(args.dnn_target),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_path),
        ]
        if preprocess == "yolo_letterbox_640":
            cmd.append("--swap-rb")
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if dump_io_path:
            cmd.extend(["--dump-io", str(dump_io_path)])
        if args.strict:
            cmd.append("--strict")
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd)

    meta = _load_json(meta_path) if meta_path.exists() else {}
    unified = {
        "exporter": "opencv_dnn_unified",
        "backend": str(args.dnn_backend),
        "target": str(args.dnn_target),
        "decode": decode,
        "preprocess_preset": preprocess,
        "dump_io": str(dump_io_path) if dump_io_path else None,
    }
    if isinstance(meta, dict):
        meta["unified"] = unified
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
