from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from yolozu.core.cli_args import (
    require_float_in_range,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_int,
)
from yolozu.predictions import normalize_predictions_payload
from yolozu.predictions.predictions_transform import apply_ttt_lite, summarize_task_coverage
from yolozu.predictions.schema_governance import CURRENT_SCHEMA_VERSION
from yolozu.tta.cli_options import add_ttt_arguments, build_ttt_cli_args
from yolozu.tta.presets import apply_ttt_preset_args

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = "reports/predictions.json"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str | Path) -> str | None:
    try:
        p = Path(path)
        return sha256_bytes(p.read_bytes())
    except Exception:
        return None


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(data)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def ensure_wrapper(payload: Any) -> dict[str, Any]:
    entries, meta = normalize_predictions_payload(payload)
    schema_version = (
        payload.get("schema_version", CURRENT_SCHEMA_VERSION)
        if isinstance(payload, dict) and "predictions" in payload
        else CURRENT_SCHEMA_VERSION
    )
    return {
        "schema_version": schema_version,
        "predictions": entries,
        "meta": dict(meta or {}),
    }


def validate_compile_report(
    report: Any,
    *,
    enabled: bool,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool | None,
    allow_fallback: bool,
) -> None:
    if not isinstance(report, dict):
        raise ValueError("missing meta.inference.torch_compile evidence")
    requested = report.get("requested")
    actual = report.get("actual")
    runtime = report.get("evidence")
    if not isinstance(requested, dict) or not isinstance(actual, dict):
        raise ValueError("compile evidence must separate requested and actual states")
    if not isinstance(runtime, dict):
        raise ValueError("compile evidence is missing runtime evidence")

    expected_request = {
        "enabled": bool(enabled),
        "backend": str(backend) if enabled else None,
        "mode": str(mode) if enabled else None,
        "fullgraph": bool(fullgraph) if enabled else None,
        "dynamic": dynamic if enabled else None,
        "allow_fallback": bool(allow_fallback) if enabled else False,
    }
    if requested != expected_request:
        raise ValueError(
            f"compile request evidence mismatch: expected {expected_request}, got {requested}"
        )

    status = actual.get("status")
    required_runtime_keys = {
        "compile_api_available",
        "setup_completed",
        "first_execution_completed",
        "fallback_execution_completed",
        "counter_source",
        "counter_delta",
        "graph_count",
        "graph_break_count",
        "captured_call_count",
    }
    missing = sorted(required_runtime_keys - set(runtime))
    if missing:
        raise ValueError(f"compile runtime evidence missing keys: {missing}")

    if not enabled:
        expected_actual = {
            "status": "not_requested",
            "backend": "eager",
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
        }
        if actual != expected_actual:
            raise ValueError(
                "compile was not requested but actual state does not match "
                f"eager execution: {actual}"
            )
        expected_runtime = {
            "compile_api_available": None,
            "setup_completed": False,
            "first_execution_completed": False,
            "fallback_execution_completed": False,
            "counter_source": None,
            "counter_delta": None,
            "graph_count": None,
            "graph_break_count": None,
            "captured_call_count": None,
        }
        for key, value in expected_runtime.items():
            if runtime.get(key) != value:
                raise ValueError(
                    f"compile was not requested but evidence.{key}={runtime.get(key)!r}"
                )
        if report.get("failure") is not None:
            raise ValueError("compile was not requested but failure evidence is present")
        return

    if status == "compiled":
        expected_actual = {
            "status": "compiled",
            "backend": str(backend),
            "mode": str(mode),
            "fullgraph": bool(fullgraph),
            "dynamic": dynamic,
        }
        if actual != expected_actual:
            raise ValueError(
                f"actual compile settings mismatch: expected {expected_actual}, got {actual}"
            )
        if runtime.get("compile_api_available") is not True:
            raise ValueError("compiled status requires an available torch.compile API")
        if runtime.get("setup_completed") is not True:
            raise ValueError("compiled status requires completed compile setup")
        if runtime.get("first_execution_completed") is not True:
            raise ValueError("compiled status requires a completed first execution")
        if runtime.get("fallback_execution_completed") is not False:
            raise ValueError("compiled status cannot include eager fallback execution")
        if report.get("failure") is not None:
            raise ValueError("compiled status cannot include compile failure evidence")
        return

    if status == "fallback" and allow_fallback:
        expected_actual = {
            "status": "fallback",
            "backend": "eager",
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
        }
        if actual != expected_actual:
            raise ValueError(
                f"fallback actual state mismatch: expected {expected_actual}, got {actual}"
            )
        if runtime.get("first_execution_completed") is not False:
            raise ValueError("fallback status cannot claim compiled first execution")
        if runtime.get("fallback_execution_completed") is not True:
            raise ValueError("fallback status requires completed eager execution")
        failure = report.get("failure")
        if not isinstance(failure, dict):
            raise ValueError("fallback status requires compile failure evidence")
        failure_phase = failure.get("phase")
        if failure_phase not in {"setup", "first_execution"}:
            raise ValueError("fallback compile failure phase is invalid")
        if not isinstance(runtime.get("compile_api_available"), bool):
            raise ValueError("fallback status requires compile API availability evidence")
        if runtime.get("setup_completed") is not (
            failure_phase == "first_execution"
        ):
            raise ValueError("fallback setup evidence does not match its failure phase")
        if not isinstance(failure.get("type"), str) or not isinstance(
            failure.get("message"), str
        ):
            raise ValueError("fallback compile failure evidence is incomplete")
        return

    raise ValueError(
        f"requested compile execution was not established (actual status: {status!r})"
    )


def validate_export_numeric_args(args: argparse.Namespace) -> None:
    try:
        require_non_negative_int(args.max_images, flag_name="--max-images")
        require_positive_int(args.infer_batch_size, flag_name="--infer-batch-size")
        require_positive_int(args.topk, flag_name="--topk")
        require_positive_int(args.max_detections, flag_name="--max-detections")
        require_float_in_range(args.min_score, flag_name="--min-score", minimum=0.0, maximum=1.0)
        require_float_in_range(
            args.score_threshold,
            flag_name="--score-threshold",
            minimum=0.0,
            maximum=1.0,
        )
        require_float_in_range(args.score_thr, flag_name="--score-thr", minimum=0.0, maximum=1.0)
        require_float_in_range(args.nms_iou, flag_name="--nms-iou", minimum=0.0, maximum=1.0)
        require_float_in_range(args.tta_model_merge_iou, flag_name="--tta-model-merge-iou", minimum=0.0, maximum=1.0)
        require_non_negative_float(args.ttt_aux_pose_weight, flag_name="--ttt-aux-pose-weight")
        require_non_negative_float(args.ttt_aux_keypoints_weight, flag_name="--ttt-aux-keypoints-weight")
        require_non_negative_float(args.ttt_aux_depth_weight, flag_name="--ttt-aux-depth-weight")
        require_non_negative_float(args.ttt_aux_seg_weight, flag_name="--ttt-aux-seg-weight")
        require_non_negative_float(args.ttt_lite_entropy_weight, flag_name="--ttt-lite-entropy-weight")
        if float(args.ttt_aux_temperature) <= 0.0:
            raise ValueError("--ttt-aux-temperature must be > 0")
        if float(args.ttt_lite_temperature) <= 0.0:
            raise ValueError("--ttt-lite-temperature must be > 0")
    except ValueError as exc:
        msg = str(exc)
        msg = msg.replace("--topk must be > 0", "--topk must be >= 1")
        msg = msg.replace("--max-detections must be > 0", "--max-detections must be >= 1")
        raise SystemExit(msg) from exc


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def compile_dynamic_value(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return None
    return normalized == "true"


def output_config_hash(path: Path) -> str | None:
    try:
        payload = ensure_wrapper(load_json(path))
    except Exception:
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    run = meta.get("run")
    if not isinstance(run, dict):
        return None
    got = run.get("config_hash")
    return got if isinstance(got, str) and got else None


def ensure_output_matches(path: Path, *, expected_config_hash: str) -> None:
    got = output_config_hash(path)
    if got is None:
        raise SystemExit(f"output exists but missing meta.run.config_hash: {path} (use --force to overwrite)")
    if got != expected_config_hash:
        raise SystemExit(
            "output exists but does not match current config_hash:\n"
            f"  path: {path}\n"
            f"  expected: {expected_config_hash}\n"
            f"  got: {got}\n"
            "Use --force to overwrite, or choose a different --output/--run-dir/--cache-dir."
        )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


_ACCEL_EXPORT_SCRIPTS = {
    "onnxrt": "tools/export_predictions_onnxrt.py",
    "trt": "tools/export_predictions_trt.py",
    "executorch": "tools/export_predictions_executorch.py",
}

_NON_ACCEL_EXPORT_SCRIPTS = {
    "yolox": "tools/export_predictions_yolox.py",
    "opencv-dnn": "tools/export_predictions_opencv_dnn_unified.py",
    "opencv-dnn-rtdetr": "tools/export_predictions_opencv_dnn_rtdetr.py",
    "opencv-dnn-yolo": "tools/export_predictions_opencv_dnn.py",
}


def ensure_exporter_script_exists(*, backend: str) -> str:
    relpath = _ACCEL_EXPORT_SCRIPTS.get(str(backend)) or _NON_ACCEL_EXPORT_SCRIPTS.get(str(backend))
    if not isinstance(relpath, str) or not relpath:
        raise SystemExit(f"internal error: unknown accelerator backend: {backend}")
    script_path = REPO_ROOT / relpath
    if not script_path.is_file():
        raise SystemExit(
            f"export backend '{backend}' is declared but missing exporter script: {relpath}. "
            "Install/update repository sources or switch backend."
        )
    return relpath


def validate_torch_only_flags(*, args: argparse.Namespace, backend: str) -> None:
    compile_opts_changed = bool(
        bool(args.torch_compile)
        or str(args.torch_compile_backend) != "inductor"
        or str(args.torch_compile_mode) != "reduce-overhead"
        or bool(getattr(args, "torch_compile_fullgraph", False))
        or str(getattr(args, "torch_compile_dynamic", "auto")) != "auto"
        or bool(getattr(args, "allow_compile_fallback", False))
    )
    accel_opts_changed = bool(
        compile_opts_changed
        or str(getattr(args, "torch_amp", "off")) != "off"
        or bool(getattr(args, "torch_channels_last", False))
        or not bool(getattr(args, "torch_inference_mode", True))
    )
    infer_batch_changed = int(args.infer_batch_size) != 1
    ttt_lite_non_torch = bool(getattr(args, "ttt_lite_non_torch", False))
    disallowed_ttt = bool(args.ttt) and not ttt_lite_non_torch
    if args.tta or disallowed_ttt or int(args.lora_r) > 0 or accel_opts_changed or infer_batch_changed:
        raise SystemExit(
            f"--tta/--ttt/--lora-* are only supported for --backend dummy/torch (got: {backend}); "
            "--torch-compile* and --infer-batch-size require --backend torch; "
            "--torch-amp/--torch-channels-last/--[no-]torch-inference-mode also require --backend torch; "
            "for non-torch backends use --ttt-lite-non-torch with --ttt to enable score-only adaptation"
        )


def require_backend_model(*, args: argparse.Namespace, backend: str) -> str:
    model = args.model
    if not model:
        raise SystemExit(f"--model is required for --backend {backend}")
    return str(model)


def build_accel_backend_config(*, args: argparse.Namespace, backend: str, dataset_fp: str) -> dict[str, Any]:
    validate_torch_only_flags(args=args, backend=backend)
    model = require_backend_model(args=args, backend=backend)
    if backend == "onnxrt":
        return {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "model": model,
            "model_sha256": sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    if backend == "trt":
        return {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "engine": model,
            "engine_sha256": sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    if backend == "executorch":
        runtime_output_json = getattr(args, "runtime_output_json", None)
        return {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "model": model,
            "model_sha256": sha256_file(model),
            "runtime_output_json": str(runtime_output_json) if runtime_output_json else None,
            "runtime_output_json_sha256": sha256_file(runtime_output_json) if runtime_output_json else None,
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    raise SystemExit(f"internal error: unsupported accelerator backend: {backend}")


def build_accel_backend_command(*, args: argparse.Namespace, backend: str, dataset: str, out_path: Path) -> list[str]:
    model = require_backend_model(args=args, backend=backend)
    script = ensure_exporter_script_exists(backend=backend)
    cmd = [
        sys.executable,
        script,
        "--dataset",
        str(dataset),
        "--output",
        str(out_path),
        "--wrap",
    ]
    if backend == "onnxrt":
        cmd.extend(
            [
                "--onnx",
                model,
                "--input-name",
                str(args.input_name),
                "--combined-output",
                str(args.combined_output),
                "--boxes-scale",
                str(args.boxes_scale),
            ]
        )
    elif backend == "trt":
        cmd.extend(
            [
                "--engine",
                model,
                "--input-name",
                str(args.input_name),
                "--combined-output",
                str(args.combined_output),
                "--boxes-scale",
                str(args.boxes_scale),
            ]
        )
    elif backend == "executorch":
        cmd.extend(["--model", model])
        if getattr(args, "runtime_output_json", None):
            cmd.extend(["--runtime-output-json", str(args.runtime_output_json)])
        cmd.extend(["--boxes-scale", str(args.boxes_scale)])
    else:
        raise SystemExit(f"internal error: unsupported accelerator backend: {backend}")

    cmd.extend(["--min-score", str(float(args.min_score)), "--topk", str(int(args.topk))])
    if args.split:
        cmd.extend(["--split", str(args.split)])
    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.dry_run:
        cmd.append("--dry-run")
    if backend == "executorch" and args.strict:
        cmd.append("--strict")
    return cmd


def parse_common_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=("dummy", "torch", "onnxrt", "trt", "executorch", "yolox", "opencv-dnn", "opencv-dnn-rtdetr", "opencv-dnn-yolo"),
        default="dummy",
        help="Inference backend (default: dummy).",
    )
    p.add_argument("--dataset", default=None, help="YOLO-format dataset root (defaults to data/coco128).")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("--output", default=DEFAULT_PREDICTIONS_PATH, help="Predictions JSON output path.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. When set and --output is default, writes <run-dir>/predictions.json.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Enable fingerprinted run cache. When set and --output is default, writes into --cache-dir/<config_hash>/predictions.json.",
    )
    p.add_argument("--cache-dir", default="runs/yolozu_runs", help="Cache root directory (default: runs/yolozu_runs).")
    p.add_argument("--notes", default=None, help="Notes to store in meta.run.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed to store in meta.run.")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    p.add_argument("--dry-run", action="store_true", help="Backend dry-run when supported (onnxrt/trt).")
    p.add_argument("--strict", action="store_true", help="Enable strict predictions validation when backend supports it.")

    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="Torch config path (rtdetr_pose).")
    p.add_argument("--checkpoint", default=None, help="Torch checkpoint path (optional).")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    p.add_argument("--infer-batch-size", type=int, default=1, help="Torch inference batch size (default: 1).")
    p.add_argument("--image-size", type=int, nargs="+", default=None, help="Torch image size (one or two ints).")
    p.add_argument("--score-threshold", type=float, default=0.3, help="Torch score threshold (default: 0.3).")
    p.add_argument("--max-detections", type=int, default=50, help="Torch max detections (default: 50).")
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Request evidenced torch.compile execution for torch backend inference."
        ),
    )
    p.add_argument("--torch-compile-backend", default="inductor", help="torch.compile backend (default: inductor).")
    p.add_argument(
        "--torch-compile-mode",
        default="reduce-overhead",
        help="torch.compile mode (default: reduce-overhead).",
    )
    p.add_argument(
        "--torch-compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a single torch.compile graph (default: false).",
    )
    p.add_argument(
        "--torch-compile-dynamic",
        choices=("auto", "true", "false"),
        default="auto",
        help="torch.compile dynamic-shape policy: auto, true, or false (default: auto).",
    )
    p.add_argument(
        "--allow-compile-fallback",
        action="store_true",
        help=(
            "Allow explicit eager fallback when requested torch.compile setup or "
            "first execution fails; records actual.status=fallback."
        ),
    )
    p.add_argument(
        "--torch-amp",
        choices=("off", "fp16", "bf16"),
        default="off",
        help="Torch autocast dtype for inference (default: off).",
    )
    p.add_argument(
        "--torch-channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use channels_last memory format for torch inference tensors (default: false).",
    )
    p.add_argument(
        "--torch-inference-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.inference_mode instead of torch.no_grad (default: true).",
    )
    p.add_argument("--lora-r", type=int, default=0, help="Enable LoRA by setting rank r>0 (default: 0 disables).")
    p.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling (default: r).")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout on inputs (default: 0.0).")
    p.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    p.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base weights and train LoRA params only (default: false).",
    )
    p.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
    )
    p.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    p.add_argument(
        "--tta-mode",
        choices=("postprocess", "model"),
        default="postprocess",
        help="TTA mode for torch backend: postprocess (default) or model-space branch merge.",
    )
    p.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    p.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    p.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    p.add_argument(
        "--tta-keypoint-swap-pairs",
        default=None,
        help="Optional keypoint swap pairs like '1:2,3:4' for horizontal flip semantics.",
    )
    p.add_argument(
        "--tta-model-merge-iou",
        type=float,
        default=0.55,
        help="IoU threshold for --tta-mode model branch merge (default: 0.55).",
    )
    p.add_argument(
        "--tta-flip-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When TTA is enabled, horizontally flip keypoints x coordinates (default: true).",
    )
    p.add_argument(
        "--tta-flip-pose-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When TTA is enabled, horizontally flip pose offsets x component (default: true).",
    )
    p.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")
    add_ttt_arguments(p, include_enable_flag=True)
    p.add_argument(
        "--ttt-lite-non-torch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow score-only TTT-lite for non-torch backends when --ttt is requested.",
    )
    p.add_argument(
        "--ttt-lite-temperature",
        type=float,
        default=1.0,
        help="Temperature for non-torch TTT-lite score scaling (default: 1.0).",
    )
    p.add_argument(
        "--ttt-lite-entropy-weight",
        type=float,
        default=0.0,
        help="Entropy penalty weight for non-torch TTT-lite (default: 0.0).",
    )
    p.add_argument(
        "--ttt-lite-minmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-image min-max normalization in non-torch TTT-lite (default: true).",
    )

    p.add_argument("--model", default=None, help="Model path (.onnx for onnxrt, .plan for trt, .pte for executorch).")
    p.add_argument("--exp", default=None, help="YOLOX exp file path for --backend yolox.")
    p.add_argument("--weights", default=None, help="YOLOX checkpoint path for --backend yolox.")
    p.add_argument("--input-name", default="images", help="Input tensor/binding name (default: images).")
    p.add_argument("--combined-output", default="output0", help="Combined output name (default: output0).")
    p.add_argument(
        "--runtime-output-json",
        default=None,
        help="ExecuTorch runtime output JSON to decode for --backend executorch non-dry runs.",
    )
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="abs",
        help="Combined boxes scale (default: abs).",
    )
    p.add_argument("--min-score", type=float, default=0.0, help="Score threshold (default: 0.0).")
    p.add_argument("--topk", type=int, default=300, help="Top-K per image (default: 300).")

    p.add_argument("--onnx", default=None, help="ONNX model path for --backend opencv-dnn-rtdetr (alias: --model).")
    p.add_argument("--imgsz", type=int, default=640, help="Input size for opencv-dnn-rtdetr backend (default: 640).")
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.01,
        help="Score threshold for opencv-dnn-rtdetr backend (default: 0.01).",
    )
    p.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Keep aspect ratio via letterbox in opencv-dnn-rtdetr backend (default: False).",
    )
    p.add_argument(
        "--dnn-backend",
        default="opencv",
        help="OpenCV DNN backend for opencv-dnn-rtdetr (e.g. opencv, cuda) (default: opencv).",
    )
    p.add_argument(
        "--dnn-target",
        default="cpu",
        help="OpenCV DNN target for opencv-dnn-rtdetr (e.g. cpu, cuda, cuda_fp16) (default: cpu).",
    )
    p.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for opencv-dnn-yolo backend (default: 0.45).",
    )
    p.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS for opencv-dnn-yolo backend.")
    p.add_argument(
        "--raw-format",
        choices=("yolo_84", "yolo_85_obj"),
        default="yolo_84",
        help="Raw head layout for opencv-dnn-yolo backend (default: yolo_84).",
    )
    p.add_argument(
        "--decode",
        choices=("auto", "yolo_84", "yolo_85_obj", "rtdetr"),
        default="auto",
        help="Unified OpenCV decode selection for --backend opencv-dnn (default: auto).",
    )
    p.add_argument(
        "--preprocess",
        choices=("yolo_letterbox_640", "rtdetr_resize_640", "rtdetr_letterbox_640"),
        default=None,
        help="Unified OpenCV preprocess preset for --backend opencv-dnn.",
    )
    p.add_argument(
        "--dump-io",
        default=None,
        help="Optional IO probe dump path for OpenCV backends (input/output tensor names/shapes/dtypes).",
    )


def export_with_backend(
    args: argparse.Namespace,
    *,
    subprocess_or_die: Callable[[list[str]], str],
    base_run_meta: Callable[..., dict[str, Any]],
    dataset_override: str | None = None,
    dataset_meta: str | None = None,
) -> Path:
    validate_export_numeric_args(args)

    dataset = dataset_override or (args.dataset if args.dataset else str(REPO_ROOT / "data" / "coco128"))
    dataset_fp = dataset_meta or dataset

    backend = str(args.backend)
    compile_fullgraph = bool(getattr(args, "torch_compile_fullgraph", False))
    compile_dynamic = str(getattr(args, "torch_compile_dynamic", "auto"))
    allow_compile_fallback = bool(getattr(args, "allow_compile_fallback", False))
    if allow_compile_fallback and not bool(args.torch_compile):
        raise SystemExit("--allow-compile-fallback requires --torch-compile")
    if (
        str(args.torch_compile_backend) != "inductor"
        or str(args.torch_compile_mode) != "reduce-overhead"
        or compile_fullgraph
        or compile_dynamic != "auto"
    ) and not bool(args.torch_compile):
        raise SystemExit("--torch-compile-* options require --torch-compile")

    adapter = None
    config_fp: dict[str, Any]

    if backend in ("dummy", "torch"):
        if backend == "dummy":
            compile_opts_changed = bool(
                bool(args.torch_compile)
                or str(args.torch_compile_backend) != "inductor"
                or str(args.torch_compile_mode) != "reduce-overhead"
                or compile_fullgraph
                or compile_dynamic != "auto"
                or allow_compile_fallback
                or str(args.torch_amp) != "off"
                or bool(args.torch_channels_last)
                or not bool(args.torch_inference_mode)
            )
            if compile_opts_changed or int(args.infer_batch_size) != 1:
                raise SystemExit(
                    "--torch-compile*/--torch-amp/--torch-channels-last/--[no-]torch-inference-mode "
                    "and --infer-batch-size are only supported for --backend torch"
                )
        adapter = "dummy" if backend == "dummy" else "rtdetr_pose"
        lora_enabled = bool(backend == "torch" and int(args.lora_r) > 0)
        tta_enabled = bool(args.tta)
        ttt_enabled = bool(args.ttt)
        torch_compile_enabled = bool(backend == "torch" and bool(args.torch_compile))
        if ttt_enabled:
            apply_ttt_preset_args(args)
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "adapter": adapter,
            "config": str(args.config) if backend == "torch" else None,
            "config_sha256": sha256_file(REPO_ROOT / str(args.config)) if backend == "torch" else None,
            "checkpoint": str(args.checkpoint) if backend == "torch" else None,
            "checkpoint_sha256": sha256_file(args.checkpoint) if backend == "torch" and args.checkpoint else None,
            "device": str(args.device) if backend == "torch" else None,
            "image_size": list(args.image_size) if backend == "torch" and args.image_size else None,
            "score_threshold": float(args.score_threshold) if backend == "torch" else None,
            "max_detections": int(args.max_detections) if backend == "torch" else None,
            "infer_batch_size": int(args.infer_batch_size) if backend == "torch" else None,
            "torch_compile": {
                "requested": {
                    "enabled": torch_compile_enabled,
                    "backend": (
                        str(args.torch_compile_backend)
                        if torch_compile_enabled
                        else None
                    ),
                    "mode": (
                        str(args.torch_compile_mode)
                        if torch_compile_enabled
                        else None
                    ),
                    "fullgraph": compile_fullgraph if torch_compile_enabled else None,
                    "dynamic": (
                        compile_dynamic_value(compile_dynamic)
                        if torch_compile_enabled
                        else None
                    ),
                    "allow_fallback": (
                        allow_compile_fallback if torch_compile_enabled else False
                    ),
                },
            },
            "torch_amp": (str(args.torch_amp) if backend == "torch" else None),
            "torch_channels_last": (bool(args.torch_channels_last) if backend == "torch" else None),
            "torch_inference_mode": (bool(args.torch_inference_mode) if backend == "torch" else None),
            "lora": {
                "enabled": lora_enabled,
                "r": int(args.lora_r) if lora_enabled else 0,
                "alpha": float(args.lora_alpha) if lora_enabled and args.lora_alpha is not None else None,
                "dropout": float(args.lora_dropout) if lora_enabled else None,
                "target": str(args.lora_target) if lora_enabled else None,
                "freeze_base": bool(args.lora_freeze_base) if lora_enabled else None,
                "train_bias": str(args.lora_train_bias) if lora_enabled else None,
            },
            "tta": {
                "enabled": tta_enabled,
                "mode": (str(args.tta_mode) if tta_enabled else "postprocess"),
                "seed": args.tta_seed if tta_enabled else None,
                "flip_prob": float(args.tta_flip_prob) if tta_enabled else None,
                "norm_only": bool(args.tta_norm_only) if tta_enabled else None,
                "keypoint_swap_pairs": (str(args.tta_keypoint_swap_pairs) if tta_enabled and args.tta_keypoint_swap_pairs else None),
                "model_merge_iou": (float(args.tta_model_merge_iou) if tta_enabled else None),
                "flip_keypoints": bool(args.tta_flip_keypoints) if tta_enabled else None,
                "flip_pose_offsets": bool(args.tta_flip_pose_offsets) if tta_enabled else None,
            },
            "ttt": {
                "enabled": ttt_enabled,
                "preset": args.ttt_preset if ttt_enabled else None,
                "method": str(args.ttt_method) if ttt_enabled else None,
                "reset": str(args.ttt_reset) if ttt_enabled else None,
                "steps": int(args.ttt_steps) if ttt_enabled else None,
                "batch_size": int(args.ttt_batch_size) if ttt_enabled else None,
                "lr": float(args.ttt_lr) if ttt_enabled else None,
                "stop_on_non_finite": bool(args.ttt_stop_on_non_finite) if ttt_enabled else None,
                "rollback_on_stop": bool(args.ttt_rollback_on_stop) if ttt_enabled else None,
                "max_grad_norm": float(args.ttt_max_grad_norm) if ttt_enabled and args.ttt_max_grad_norm is not None else None,
                "max_update_norm": float(args.ttt_max_update_norm) if ttt_enabled and args.ttt_max_update_norm is not None else None,
                "max_total_update_norm": (
                    float(args.ttt_max_total_update_norm)
                    if ttt_enabled and args.ttt_max_total_update_norm is not None
                    else None
                ),
                "max_loss_ratio": float(args.ttt_max_loss_ratio) if ttt_enabled and args.ttt_max_loss_ratio is not None else None,
                "max_loss_increase": (
                    float(args.ttt_max_loss_increase)
                    if ttt_enabled and args.ttt_max_loss_increase is not None
                    else None
                ),
                "update_filter": str(args.ttt_update_filter) if ttt_enabled else None,
                "include": list(args.ttt_include) if ttt_enabled and args.ttt_include else None,
                "exclude": list(args.ttt_exclude) if ttt_enabled and args.ttt_exclude else None,
                "max_batches": int(args.ttt_max_batches) if ttt_enabled else None,
                "seed": args.ttt_seed if ttt_enabled else None,
                "sdft_task": (str(args.ttt_sdft_task) if ttt_enabled and args.ttt_sdft_task else None),
                "mim": {
                    "mask_prob": float(args.ttt_mask_prob) if ttt_enabled else None,
                    "patch_size": int(args.ttt_patch_size) if ttt_enabled else None,
                    "mask_value": float(args.ttt_mask_value) if ttt_enabled else None,
                },
                "cotta": {
                    "ema_momentum": float(args.ttt_cotta_ema_momentum) if ttt_enabled else None,
                    "augmentations": list(args.ttt_cotta_augmentations) if ttt_enabled and args.ttt_cotta_augmentations else None,
                    "aggregation": str(args.ttt_cotta_aggregation) if ttt_enabled else None,
                    "restore_prob": float(args.ttt_cotta_restore_prob) if ttt_enabled else None,
                    "restore_interval": int(args.ttt_cotta_restore_interval) if ttt_enabled else None,
                },
                "eata": {
                    "conf_min": float(args.ttt_eata_conf_min) if ttt_enabled else None,
                    "entropy_min": float(args.ttt_eata_entropy_min) if ttt_enabled else None,
                    "entropy_max": float(args.ttt_eata_entropy_max) if ttt_enabled else None,
                    "min_valid_dets": int(args.ttt_eata_min_valid_dets) if ttt_enabled else None,
                    "anchor_lambda": float(args.ttt_eata_anchor_lambda) if ttt_enabled else None,
                    "selected_ratio_min": float(args.ttt_eata_selected_ratio_min) if ttt_enabled else None,
                    "max_skip_streak": int(args.ttt_eata_max_skip_streak) if ttt_enabled else None,
                },
                "sar": {
                    "rho": float(args.ttt_sar_rho) if ttt_enabled else None,
                    "adaptive": bool(args.ttt_sar_adaptive) if ttt_enabled else None,
                    "first_step_scale": float(args.ttt_sar_first_step_scale) if ttt_enabled else None,
                },
                "aux": {
                    "pose_weight": float(args.ttt_aux_pose_weight) if ttt_enabled else None,
                    "keypoints_weight": float(args.ttt_aux_keypoints_weight) if ttt_enabled else None,
                    "depth_weight": float(args.ttt_aux_depth_weight) if ttt_enabled else None,
                    "seg_weight": float(args.ttt_aux_seg_weight) if ttt_enabled else None,
                    "temperature": float(args.ttt_aux_temperature) if ttt_enabled else None,
                },
            },
        }
    elif backend in ("onnxrt", "trt", "executorch"):
        config_fp = build_accel_backend_config(args=args, backend=backend, dataset_fp=str(dataset_fp))
    elif backend == "yolox":
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "exp": str(args.exp) if args.exp else None,
            "weights": str(args.weights) if args.weights else None,
            "weights_sha256": sha256_file(args.weights) if args.weights else None,
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_thr": float(args.nms_iou),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    elif backend == "opencv-dnn":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_iou": float(args.nms_iou),
            "topk": int(args.topk),
            "decode": str(args.decode),
            "preprocess": str(args.preprocess) if args.preprocess else None,
            "dnn_backend": str(args.dnn_backend),
            "dnn_target": str(args.dnn_target),
            "dump_io": str(args.dump_io) if args.dump_io else None,
            "dry_run": bool(args.dry_run),
        }
    elif backend == "opencv-dnn-rtdetr":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-rtdetr")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "keep_aspect": bool(args.keep_aspect),
            "dnn_backend": str(args.dnn_backend),
            "dnn_target": str(args.dnn_target),
            "topk": int(args.topk),
            "dump_io": str(args.dump_io) if args.dump_io else None,
        }
    elif backend == "opencv-dnn-yolo":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-yolo")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_iou": float(args.nms_iou),
            "agnostic_nms": bool(args.agnostic_nms),
            "raw_format": str(args.raw_format),
            "dump_io": str(args.dump_io) if args.dump_io else None,
        }
    else:
        raise SystemExit(f"unknown backend: {backend}")

    if backend not in ("dummy", "torch"):
        config_fp["ttt_lite"] = {
            "enabled": bool(args.ttt and args.ttt_lite_non_torch),
            "temperature": float(args.ttt_lite_temperature),
            "entropy_weight": float(args.ttt_lite_entropy_weight),
            "minmax_norm": bool(args.ttt_lite_minmax),
        }

    config_hash = sha256_json(config_fp)

    out_path = resolve_path(args.output)

    run_dir = None
    if args.run_dir and args.output == DEFAULT_PREDICTIONS_PATH:
        run_dir = resolve_path(args.run_dir)
        out_path = run_dir / "predictions.json"

    cache_out = None
    if args.cache:
        cache_out = resolve_path(args.cache_dir) / config_hash / "predictions.json"
        if args.output == DEFAULT_PREDICTIONS_PATH and not args.run_dir:
            out_path = cache_out

    def _validate_existing_compile_evidence(path: Path) -> None:
        if backend != "torch":
            return
        try:
            existing = ensure_wrapper(load_json(path))
            existing_meta = existing.get("meta")
            existing_inference = (
                existing_meta.get("inference")
                if isinstance(existing_meta, dict)
                else None
            )
            existing_report = (
                existing_inference.get("torch_compile")
                if isinstance(existing_inference, dict)
                else None
            )
            validate_compile_report(
                existing_report,
                enabled=bool(args.torch_compile),
                backend=str(args.torch_compile_backend),
                mode=str(args.torch_compile_mode),
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic_value(compile_dynamic),
                allow_fallback=allow_compile_fallback,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"existing output has invalid torch.compile evidence: {path}: {exc} "
                "(use --force to replace it)"
            ) from exc

    if out_path.exists() and not args.force:
        ensure_output_matches(out_path, expected_config_hash=config_hash)
        _validate_existing_compile_evidence(out_path)
        return out_path

    if cache_out is not None and cache_out.exists() and not args.force:
        ensure_output_matches(cache_out, expected_config_hash=config_hash)
        _validate_existing_compile_evidence(cache_out)
        if cache_out != out_path:
            copy_file(cache_out, out_path)
        return out_path

    if args.force:
        for stale_path in {out_path, cache_out}:
            if stale_path is not None and stale_path.exists():
                stale_path.unlink()

    if backend in ("dummy", "torch"):
        if adapter is None:
            raise SystemExit("internal error: missing adapter")
        cmd = [
            sys.executable,
            "tools/export_predictions.py",
            "--adapter",
            adapter,
            "--dataset",
            str(dataset),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])

        if args.tta:
            cmd.append("--tta")
        cmd.extend(["--tta-mode", str(args.tta_mode)])
        if args.tta_seed is not None:
            cmd.extend(["--tta-seed", str(int(args.tta_seed))])
        cmd.extend(["--tta-flip-prob", str(float(args.tta_flip_prob))])
        cmd.extend(["--tta-model-merge-iou", str(float(args.tta_model_merge_iou))])
        if args.tta_norm_only:
            cmd.append("--tta-norm-only")
        if args.tta_keypoint_swap_pairs:
            cmd.extend(["--tta-keypoint-swap-pairs", str(args.tta_keypoint_swap_pairs)])
        cmd.append("--tta-flip-keypoints" if bool(args.tta_flip_keypoints) else "--no-tta-flip-keypoints")
        cmd.append("--tta-flip-pose-offsets" if bool(args.tta_flip_pose_offsets) else "--no-tta-flip-pose-offsets")
        if args.tta_log_out:
            cmd.extend(["--tta-log-out", str(args.tta_log_out)])

        if args.ttt:
            cmd.extend(build_ttt_cli_args(args, include_enable_flag=True))

        if backend == "torch":
            cmd.extend(
                [
                    "--config",
                    str(args.config),
                    "--device",
                    str(args.device),
                    "--score-threshold",
                    str(float(args.score_threshold)),
                    "--max-detections",
                    str(int(args.max_detections)),
                    "--infer-batch-size",
                    str(int(args.infer_batch_size)),
                    "--torch-compile-backend",
                    str(args.torch_compile_backend),
                    "--torch-compile-mode",
                    str(args.torch_compile_mode),
                    "--torch-compile-dynamic",
                    compile_dynamic,
                    "--torch-amp",
                    str(args.torch_amp),
                    "--lora-r",
                    str(int(args.lora_r)),
                    "--lora-dropout",
                    str(float(args.lora_dropout)),
                    "--lora-target",
                    str(args.lora_target),
                    "--lora-train-bias",
                    str(args.lora_train_bias),
                ]
            )
            if args.checkpoint:
                cmd.extend(["--checkpoint", str(args.checkpoint)])
            if args.image_size:
                cmd.extend(["--image-size", *[str(int(x)) for x in args.image_size]])
            if args.lora_alpha is not None:
                cmd.extend(["--lora-alpha", str(float(args.lora_alpha))])
            if bool(args.torch_compile):
                cmd.append("--torch-compile")
            if compile_fullgraph:
                cmd.append("--torch-compile-fullgraph")
            if allow_compile_fallback:
                cmd.append("--allow-compile-fallback")
            cmd.append("--torch-channels-last" if bool(args.torch_channels_last) else "--no-torch-channels-last")
            cmd.append("--torch-inference-mode" if bool(args.torch_inference_mode) else "--no-torch-inference-mode")
            cmd.append("--lora-freeze-base" if bool(args.lora_freeze_base) else "--no-lora-freeze-base")

        subprocess_or_die(cmd)
    elif backend in ("onnxrt", "trt", "executorch"):
        validate_torch_only_flags(args=args, backend=backend)
        cmd = build_accel_backend_command(args=args, backend=backend, dataset=str(dataset), out_path=out_path)
        subprocess_or_die(cmd)
    elif backend == "yolox":
        validate_torch_only_flags(args=args, backend=backend)
        script = ensure_exporter_script_exists(backend=backend)
        cmd = [
            sys.executable,
            script,
            "--dataset",
            str(dataset),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--nms-thr",
            str(float(args.nms_iou)),
            "--topk",
            str(int(args.topk)),
            "--device",
            str(args.device),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.exp:
            cmd.extend(["--exp", str(args.exp)])
        if args.weights:
            cmd.extend(["--weights", str(args.weights)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        subprocess_or_die(cmd)
    elif backend == "opencv-dnn":
        validate_torch_only_flags(args=args, backend=backend)
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn")
        script = ensure_exporter_script_exists(backend=backend)
        cmd = [
            sys.executable,
            script,
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--nms-iou",
            str(float(args.nms_iou)),
            "--topk",
            str(int(args.topk)),
            "--decode",
            str(args.decode),
            "--dnn-backend",
            str(args.dnn_backend),
            "--dnn-target",
            str(args.dnn_target),
            "--output",
            str(out_path),
        ]
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.preprocess:
            cmd.extend(["--preprocess", str(args.preprocess)])
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.strict:
            cmd.append("--strict")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        subprocess_or_die(cmd)
    elif backend == "opencv-dnn-rtdetr":
        validate_torch_only_flags(args=args, backend=backend)
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-rtdetr")
        script = ensure_exporter_script_exists(backend=backend)
        cmd = [
            sys.executable,
            script,
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.keep_aspect:
            cmd.append("--keep-aspect")
        if args.dnn_backend:
            cmd.extend(["--dnn-backend", str(args.dnn_backend)])
        if args.dnn_target:
            cmd.extend(["--dnn-target", str(args.dnn_target)])
        if args.topk is not None:
            cmd.extend(["--topk", str(int(args.topk))])
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        subprocess_or_die(cmd)
    elif backend == "opencv-dnn-yolo":
        validate_torch_only_flags(args=args, backend=backend)
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-yolo")
        script = ensure_exporter_script_exists(backend=backend)
        cmd = [
            sys.executable,
            script,
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--input-size",
            str(int(args.imgsz)),
            "--min-score",
            str(float(args.score_thr)),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.nms_iou is not None:
            cmd.extend(["--nms-iou", str(float(args.nms_iou))])
        if args.agnostic_nms:
            cmd.append("--agnostic-nms")
        if args.raw_format:
            cmd.extend(["--raw-format", str(args.raw_format)])
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        subprocess_or_die(cmd)
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend: {backend}")

    payload = ensure_wrapper(load_json(out_path))
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    if backend == "torch":
        inference = meta.get("inference")
        report = (
            inference.get("torch_compile")
            if isinstance(inference, dict)
            else None
        )
        try:
            validate_compile_report(
                report,
                enabled=bool(args.torch_compile),
                backend=str(args.torch_compile_backend),
                mode=str(args.torch_compile_mode),
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic_value(compile_dynamic),
                allow_fallback=allow_compile_fallback,
            )
        except ValueError as exc:
            if out_path.exists():
                out_path.unlink()
            raise SystemExit(f"invalid torch.compile evidence: {exc}") from exc

    if backend not in ("dummy", "torch") and bool(args.ttt) and bool(args.ttt_lite_non_torch):
        before_scores: list[float] = []
        for entry in payload.get("predictions") or []:
            if not isinstance(entry, dict):
                continue
            for det in entry.get("detections") or []:
                if not isinstance(det, dict):
                    continue
                try:
                    before_scores.append(float(det.get("score", 0.0)))
                except Exception:
                    before_scores.append(0.0)
        lite = apply_ttt_lite(
            payload.get("predictions") or [],
            enabled=True,
            temperature=float(args.ttt_lite_temperature),
            entropy_weight=float(args.ttt_lite_entropy_weight),
            minmax_norm=bool(args.ttt_lite_minmax),
            preserve_raw_score_key="score_raw",
        )
        payload["predictions"] = lite.entries
        after_scores: list[float] = []
        changed = 0
        for entry in lite.entries:
            if not isinstance(entry, dict):
                continue
            for det in entry.get("detections") or []:
                if not isinstance(det, dict):
                    continue
                try:
                    score_new = float(det.get("score", 0.0))
                except Exception:
                    score_new = 0.0
                after_scores.append(score_new)
                try:
                    score_old = float(det.get("score_raw", score_new))
                except Exception:
                    score_old = score_new
                if abs(score_new - score_old) > 1e-8:
                    changed += 1

        ttt_meta = meta.get("ttt")
        if not isinstance(ttt_meta, dict):
            ttt_meta = {}
        ttt_meta.update(
            {
                "enabled": True,
                "method": "lite_non_torch",
                "steps": 0,
                "batch_size": 0,
                "lr": 0.0,
                "update_filter": "score_only",
                "include": None,
                "exclude": None,
                "max_batches": 0,
                "seed": args.seed,
                "mim": {"mask_prob": 0.0, "patch_size": 0, "mask_value": 0.0},
                "report": {
                    "mode": "lite_non_torch",
                    "temperature": float(args.ttt_lite_temperature),
                    "entropy_weight": float(args.ttt_lite_entropy_weight),
                    "minmax_norm": bool(args.ttt_lite_minmax),
                    "detections": int(len(after_scores)),
                    "changed_scores": int(changed),
                    "warnings": list(lite.warnings),
                    "mean_score_before": (
                        float(sum(before_scores) / max(1, len(before_scores))) if before_scores else 0.0
                    ),
                    "mean_score_after": (
                        float(sum(after_scores) / max(1, len(after_scores))) if after_scores else 0.0
                    ),
                },
            }
        )
        meta["ttt"] = ttt_meta

    meta["task_coverage"] = summarize_task_coverage(payload.get("predictions") or [])
    meta["run"] = base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
    write_json(out_path, payload)

    if cache_out is not None and cache_out != out_path:
        copy_file(out_path, cache_out)

    meta_dir = cache_out.parent if cache_out is not None else run_dir
    if meta_dir is not None:
        write_json(meta_dir / "run_config.json", {"config_hash": config_hash, "config_fingerprint": config_fp})

    return out_path
