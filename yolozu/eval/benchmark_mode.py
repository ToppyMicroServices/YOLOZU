"""Ultralytics-parity benchmark mode.

This module powers both ``yolozu benchmark`` and ``tools/benchmark_model.py``.

Phase 1 established:
- the CLI surface,
- stable report and artifact wiring,
- explicit skipped statuses for unsupported formats,
- reproducibility metadata,
- a clearly labeled synthetic latency probe.

Phase 2 adds real backend orchestration for ``torch``, ``onnx``, ``engine``,
and ``torchscript`` by delegating to existing exporter/eval tools when the
requested artifacts and runtime dependencies are present.

Phase 2.1 promotes ``torchscript`` from accepted planning surface to a real
detect-task orchestration lane backed by a declared combined-output decode
path.

Phase 2.2 adds explicit task semantics for ``detect``, ``segmentation``,
``classification``, ``obb``, ``keypoints``/``pose``, ``depth``, and
``pose6d`` so the benchmark report no longer tells a detection-only story.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from yolozu.eval.benchmark import measure_latency
from yolozu.eval.depth_eval import compare_depth_arrays, load_depth_array, load_mask_array
from yolozu.eval.keypoints_parity import compare_keypoints_predictions
from yolozu.eval.pose_parity import compare_pose_predictions
from yolozu.eval.segmentation_parity import compare_segmentation_predictions
from yolozu.eval.metrics_report import append_jsonl, now_utc_iso, write_json
from yolozu.predictions.predictions_parity import compare_predictions
from yolozu.predictions.segmentation_predictions import load_segmentation_predictions_entries

repo_root = Path(__file__).resolve().parents[2]

PHASE1_FORMATS = ("torch", "onnx", "engine", "torchscript", "executorch", "opencv_dnn")
REAL_BACKEND_FORMATS = ("torch", "onnx", "engine", "torchscript")
BENCHMARK_UNWIRED_FORMATS = {"executorch", "opencv_dnn"}
BENCHMARK_UNWIRED_TASKS = {"classification", "obb"}
TASK_ALIASES = {
    "detect": "detect",
    "detection": "detect",
    "seg": "segmentation",
    "segmentation": "segmentation",
    "classify": "classification",
    "classification": "classification",
    "cls": "classification",
    "obb": "obb",
    "keypoints": "keypoints",
    "pose": "keypoints",
    "depth": "depth",
    "pose6d": "pose6d",
    "6dof": "pose6d",
    "pose_6d": "pose6d",
    "pose-6d": "pose6d",
}
TASK_CHOICES = tuple(sorted(TASK_ALIASES))
TASK_SEMANTICS = {
    "detect": {
        "display_name": "Detection",
        "metric_family": "bbox_map",
        "expected_metric_keys": ["mAP50-95", "mAP50", "AR@100"],
        "support_level": "real_for_torch_onnx_engine_torchscript",
        "ultralytics_surface": True,
        "yolozu_native_extension": False,
        "notes": "Default benchmark semantics. Real backend orchestration is available for torch/onnx/engine/torchscript when artifacts and runtimes are present.",
    },
    "segmentation": {
        "display_name": "Segmentation",
        "metric_family": "mask_map",
        "expected_metric_keys": ["mask_mAP50-95", "mask_mAP50", "mask_AR"],
        "support_level": "artifact_backed_real_for_torch_onnx_engine",
        "ultralytics_surface": True,
        "yolozu_native_extension": False,
        "notes": "Segmentation uses artifact-backed real evaluation and parity for torch/onnx/engine backend predictions artifacts.",
    },
    "classification": {
        "display_name": "Classification",
        "metric_family": "topk_accuracy",
        "expected_metric_keys": ["top1", "top5", "accuracy"],
        "support_level": "unsupported_skipped",
        "ultralytics_surface": True,
        "yolozu_native_extension": False,
        "notes": "Classification is explicit in the benchmark task matrix, but real benchmark orchestration is not shipped; benchmark runs report this lane as skipped rather than writing placeholder artifacts.",
    },
    "obb": {
        "display_name": "Oriented Bounding Boxes",
        "metric_family": "obb_map",
        "expected_metric_keys": ["obb_mAP50-95", "obb_mAP50", "obb_AR"],
        "support_level": "unsupported_skipped",
        "ultralytics_surface": True,
        "yolozu_native_extension": False,
        "notes": "OBB is visible in the benchmark surface and report schema, but backend/eval wiring is not shipped; benchmark runs report this lane as skipped rather than writing placeholder artifacts.",
    },
    "keypoints": {
        "display_name": "Keypoints / Pose",
        "metric_family": "oks_map",
        "expected_metric_keys": ["OKS_mAP", "PCK", "keypoint_AR"],
        "support_level": "artifact_backed_real_for_torch_onnx_engine",
        "ultralytics_surface": True,
        "yolozu_native_extension": False,
        "notes": "The CLI accepts both --task keypoints and --task pose and records a canonical keypoints task with pose alias metadata. Current benchmark execution uses backend-specific predictions artifacts for eval/parity rather than pretending YOLOZU executed the backend inference itself.",
    },
    "depth": {
        "display_name": "Monocular Depth",
        "metric_family": "depth_error",
        "expected_metric_keys": ["abs_rel", "rmse", "delta1"],
        "support_level": "artifact_backed_real_for_torch_onnx_engine",
        "ultralytics_surface": False,
        "yolozu_native_extension": True,
        "notes": "Depth is a YOLOZU-native benchmark extension with artifact-backed real evaluation and parity for torch/onnx/engine depth outputs.",
    },
    "pose6d": {
        "display_name": "6DoF Pose",
        "metric_family": "pose6d_error",
        "expected_metric_keys": ["ADD", "ADDS", "reprojection_error"],
        "support_level": "artifact_backed_real_for_torch_onnx_engine",
        "ultralytics_surface": False,
        "yolozu_native_extension": True,
        "notes": "6DoF pose is a YOLOZU-native benchmark extension with artifact-backed real evaluation and parity for torch/onnx/engine prediction artifacts.",
    },
}
FLAG_DEFAULTS = {
    "half": False,
    "int8": False,
    "batch": 1,
    "dynamic": False,
    "nms": False,
    "simplify": False,
    "opset": 17,
    "workspace": 4.0,
    "fraction": 1.0,
}
FORMAT_FLAG_RULES = {
    "torch": {
        "supported_nondefault_flags": {"half", "batch", "nms"},
        "notes": "Torch benchmark orchestration currently forwards --half, --batch, and --nms.",
    },
    "onnx": {
        "supported_nondefault_flags": set(),
        "notes": "ONNX benchmark mode currently consumes an existing ONNX artifact and does not honor export-oriented knobs.",
    },
    "engine": {
        "supported_nondefault_flags": set(),
        "notes": "TensorRT benchmark mode currently consumes an existing engine artifact and does not honor export-oriented knobs.",
    },
    "torchscript": {
        "supported_nondefault_flags": set(),
        "notes": "TorchScript detect benchmarking consumes an existing TorchScript artifact through the declared combined-output decode path; export-oriented flags remain unsupported.",
    },
    "executorch": {
        "supported_nondefault_flags": set(),
        "notes": "ExecuTorch is planning/synthetic-only in the current phase; export-oriented flags remain unsupported.",
    },
    "opencv_dnn": {
        "supported_nondefault_flags": set(),
        "notes": "OpenCV DNN is planning/synthetic-only in the current phase; export-oriented flags remain unsupported.",
    },
}


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip() or None
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path_hint(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_file(path)
    h = hashlib.sha256()
    count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(child.relative_to(path)).replace("\\", "/")
        try:
            size = child.stat().st_size
        except OSError:
            size = -1
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("utf-8"))
        h.update(b"\n")
        count += 1
        if count >= 2048:
            break
    if count == 0:
        return None
    return h.hexdigest()


def _resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _support_status_for_format(fmt: str, *, device: str, task_label: str = "detect") -> tuple[bool, str | None]:
    if task_label in BENCHMARK_UNWIRED_TASKS:
        return False, "benchmark_task_not_wired"
    if fmt in BENCHMARK_UNWIRED_FORMATS:
        return False, "benchmark_format_not_wired"
    if task_label in {"segmentation", "keypoints", "depth", "pose6d"} and fmt in REAL_BACKEND_FORMATS:
        return True, None
    device_l = str(device or "").strip().lower()
    wants_gpu = any(tok in device_l for tok in ("cuda", "gpu", "trt", "tensorrt"))
    system = platform.system().lower()

    if fmt == "torch":
        return (_module_available("ultralytics"), None if _module_available("ultralytics") else "missing_runtime_dependency")
    if fmt == "torchscript":
        return (_module_available("torch"), None if _module_available("torch") else "missing_runtime_dependency")
    if fmt == "onnx":
        return (
            _module_available("onnxruntime"),
            None if _module_available("onnxruntime") else "missing_runtime_dependency",
        )
    if fmt == "opencv_dnn":
        return (_module_available("cv2"), None if _module_available("cv2") else "missing_runtime_dependency")
    if fmt == "executorch":
        return (
            _module_available("executorch"),
            None if _module_available("executorch") else "missing_runtime_dependency",
        )
    if fmt == "engine":
        if system != "linux":
            return False, "platform_not_supported"
        if not wants_gpu:
            return False, "gpu_required"
        has_cuda_binding = _module_available("pycuda") or _module_available("cuda")
        has_tensorrt = _module_available("tensorrt")
        return (
            has_cuda_binding and has_tensorrt,
            None if (has_cuda_binding and has_tensorrt) else "missing_runtime_dependency",
        )
    return False, "unsupported_format"


def _expand_requested_formats(value: str | None, *, device: str, task_label: str = "detect") -> list[str]:
    requested = _split_csv(value)
    if not requested:
        requested = ["all"]
    if requested == ["all"]:
        expanded = [
            fmt for fmt in PHASE1_FORMATS if _support_status_for_format(fmt, device=device, task_label=task_label)[0]
        ]
        return expanded or list(PHASE1_FORMATS)
    out: list[str] = []
    for fmt in requested:
        if fmt == "all":
            out.extend([item for item in PHASE1_FORMATS if item not in out])
            continue
        if fmt not in PHASE1_FORMATS:
            raise ValueError(
                f"unsupported --format value: {fmt} (expected one of: {', '.join(PHASE1_FORMATS)} or all)"
            )
        if fmt not in out:
            out.append(fmt)
    return out


def _normalize_task_label(value: str | None) -> tuple[str, str]:
    requested = str(value or "detect").strip()
    key = requested.lower()
    canonical = TASK_ALIASES.get(key)
    if canonical is None:
        expected = ", ".join(TASK_CHOICES)
        raise ValueError(f"unsupported --task value: {requested} (expected one of: {expected})")
    return canonical, requested


def _task_semantics(value: str | None) -> dict[str, Any]:
    canonical, requested = _normalize_task_label(value)
    base = TASK_SEMANTICS[canonical]
    return {
        "canonical_task": canonical,
        "requested_task": requested,
        "display_name": base["display_name"],
        "metric_family": base["metric_family"],
        "expected_metric_keys": list(base["expected_metric_keys"]),
        "support_level": base["support_level"],
        "ultralytics_surface": bool(base["ultralytics_surface"]),
        "yolozu_native_extension": bool(base["yolozu_native_extension"]),
        "accepted_aliases": sorted(alias for alias, target in TASK_ALIASES.items() if target == canonical),
        "notes": base["notes"],
    }


def _task_execution_semantics(
    task_label: str,
    *,
    fmt: str,
    benchmark_source: str,
    dry_run: bool,
) -> dict[str, Any]:
    task_meta = TASK_SEMANTICS[task_label]
    if task_label in BENCHMARK_UNWIRED_TASKS or fmt in BENCHMARK_UNWIRED_FORMATS:
        execution_mode = "unsupported_skipped"
    elif dry_run:
        execution_mode = "dry_run_planning"
    elif task_label == "detect" and fmt in REAL_BACKEND_FORMATS and benchmark_source == "dataset_pass_wall_time":
        execution_mode = "real_backend_eval"
    elif task_label in {"segmentation", "keypoints", "depth", "pose6d"} and fmt in REAL_BACKEND_FORMATS and benchmark_source == "artifact_eval":
        execution_mode = "real_artifact_eval"
    else:
        execution_mode = "synthetic_planning_only"

    artifact_expectation: dict[str, str]
    if execution_mode in {"real_backend_eval", "real_artifact_eval"}:
        artifact_expectation = {
            "predictions": "real",
            "eval": "real",
            "parity": "real_when_comparable",
        }
    elif execution_mode == "dry_run_planning":
        artifact_expectation = {
            "predictions": "placeholder",
            "eval": "placeholder",
            "parity": "placeholder",
        }
    elif execution_mode == "unsupported_skipped":
        artifact_expectation = {
            "predictions": "skipped",
            "eval": "skipped",
            "parity": "skipped",
        }
    else:
        artifact_expectation = {
            "predictions": "placeholder",
            "eval": "placeholder",
            "parity": "placeholder",
        }

    note = task_meta["notes"]
    if task_label == "segmentation" and execution_mode == "real_artifact_eval":
        note = (
            f"{note} Current segmentation benchmarking is artifact-backed: backend-specific mask predictions artifacts are "
            "evaluated with tools/eval_segmentation.py and compared directly, without pretending YOLOZU performed the "
            "underlying backend inference itself."
        )
    elif task_label == "keypoints" and execution_mode == "real_artifact_eval":
        note = (
            f"{note} Current keypoints benchmarking is artifact-backed: backend-specific predictions artifacts are "
            "evaluated with tools/eval_keypoints.py and compared directly, without pretending YOLOZU performed the "
            "underlying backend inference itself."
        )
    elif task_label == "depth" and execution_mode == "real_artifact_eval":
        note = (
            f"{note} Current depth benchmarking is artifact-backed: backend-specific depth maps are evaluated and "
            "compared directly, without pretending YOLOZU performed the underlying depth inference itself."
        )
    elif task_label == "pose6d" and execution_mode == "real_artifact_eval":
        note = (
            f"{note} Current 6DoF benchmarking is artifact-backed: backend-specific predictions artifacts are "
            "evaluated with tools/eval_pose.py and compared directly, without pretending YOLOZU performed the "
            "underlying backend inference itself."
        )
    elif task_label != "detect":
        note = f"{note} Current benchmark execution remains planning/synthetic-only until a dedicated backend/eval path lands."

    return {
        "execution_mode": execution_mode,
        "real_backend_supported_now": bool(execution_mode in {"real_backend_eval", "real_artifact_eval"}),
        "artifact_expectation": artifact_expectation,
        "eval_expectation": {
            "metric_family": task_meta["metric_family"],
            "expected_metric_keys": list(task_meta["expected_metric_keys"]),
        },
        "notes": note,
    }


def _support_status_from_result(status: str, execution_semantics: dict[str, Any]) -> str:
    if status in {"skipped", "failed", "dry_run"}:
        return "skipped"
    mode = str(execution_semantics.get("execution_mode") or "")
    if mode == "real_backend_eval":
        return "real"
    if mode == "real_artifact_eval":
        return "artifact-backed"
    return "skipped"


def _support_reason_from_result(
    *,
    support_status: str,
    status: str,
    skip_reason: str | None,
    execution_semantics: dict[str, Any],
) -> str:
    if skip_reason:
        return str(skip_reason)
    mode = str(execution_semantics.get("execution_mode") or "")
    if support_status == "skipped":
        return mode or str(status)
    return mode


def _annotate_result_support(result: dict[str, Any], *, runtime_available: bool, runtime_reason: str | None) -> None:
    execution_semantics = result.get("execution_semantics") or {}
    status = str(result.get("status") or "")
    skip_reason = result.get("skip_reason")
    support_status = _support_status_from_result(status, execution_semantics)
    result["support_status"] = support_status
    result["support_reason"] = _support_reason_from_result(
        support_status=support_status,
        status=status,
        skip_reason=str(skip_reason) if skip_reason else None,
        execution_semantics=execution_semantics,
    )
    result["artifact_status"] = dict(execution_semantics.get("artifact_expectation") or {})
    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    runtime.update(
        {
            "available": bool(runtime_available),
            "reason": runtime_reason,
            "latency_source": result.get("latency_source"),
            "runtime_lock": (result.get("run_meta") or {}).get("runtime_lock"),
        }
    )
    result["runtime"] = runtime


def _support_summary(results: list[dict[str, Any]], requested_formats: list[str]) -> dict[str, Any]:
    by_format = {str(item.get("format")): str(item.get("support_status")) for item in results}
    counts = {"real": 0, "artifact-backed": 0, "skipped": 0}
    for value in by_format.values():
        if value in counts:
            counts[value] += 1
    missing = [fmt for fmt in requested_formats if fmt not in by_format]
    return {
        "allowed_statuses": ["real", "artifact-backed", "skipped"],
        "requested_formats": list(requested_formats),
        "reported_formats": [str(item.get("format")) for item in results],
        "missing_formats": missing,
        "counts": counts,
        "by_format": by_format,
    }


def _nondefault_flag_values(args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, default in FLAG_DEFAULTS.items():
        value = getattr(args, name, default)
        if value != default:
            out[name] = value
    return out


def _validate_benchmark_args(args: Any, requested_formats: list[str], *, task_label: str) -> None:
    nondefault = _nondefault_flag_values(args)
    if not nondefault:
        nondefault = {}
    dry_run = bool(getattr(args, "dry_run", False))

    for fmt in requested_formats:
        rule = FORMAT_FLAG_RULES.get(fmt, {"supported_nondefault_flags": set(), "notes": None})
        unsupported = sorted(name for name in nondefault if name not in rule["supported_nondefault_flags"])
        if unsupported:
            joined = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
            note = f" {rule['notes']}" if rule.get("notes") else ""
            raise ValueError(f"{joined} not supported for --format {fmt}.{note}")

        benchmark_source = _selected_benchmark_source(args, fmt=fmt, task_label=task_label)
        if task_label in {"segmentation", "keypoints", "depth", "pose6d"} and not dry_run and benchmark_source == "dataset_pass_wall_time":
            raise ValueError(
                f"--task {task_label} uses artifact-backed evaluation; use --latency-source auto or artifact_eval"
            )


def _artifact_path(base: str | None, *, fmt: str, default_name: str) -> Path:
    if not base:
        return (repo_root / "reports" / default_name.format(format=fmt)).resolve()
    text = str(base)
    if "{format}" in text:
        return _resolve_path(text.format(format=fmt)) or (repo_root / "reports" / default_name.format(format=fmt))
    path = _resolve_path(text)
    assert path is not None
    if path.suffix:
        return path.with_name(f"{path.stem}_{fmt}{path.suffix}")
    return (path / default_name.format(format=fmt)).resolve()


def _write_placeholder(
    path: Path,
    *,
    kind: str,
    fmt: str,
    status: str,
    reason: str | None,
    run_meta: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "format": fmt,
        "status": status,
        "reason": reason,
        "timestamp": now_utc_iso(),
        "run_meta": run_meta,
    }
    write_json(path, payload)


def _prediction_protocol(args: Any) -> str | None:
    protocol = getattr(args, "protocol", None)
    if protocol in {"nms_applied", "e2e_nms_free"}:
        return str(protocol)
    return None


def _selected_benchmark_source(args: Any, *, fmt: str, task_label: str) -> str:
    requested = str(getattr(args, "latency_source", "auto") or "auto")
    if requested != "auto":
        return requested
    if task_label in {"segmentation", "keypoints", "depth", "pose6d"} and fmt in REAL_BACKEND_FORMATS:
        return "artifact_eval"
    if fmt in REAL_BACKEND_FORMATS:
        return "dataset_pass_wall_time"
    return "synthetic_step"


def _resolve_model_artifact(args: Any, *, fmt: str, task_label: str) -> tuple[str | None, str | None]:
    override_map = {
        "torch": getattr(args, "torch_model", None),
        "onnx": getattr(args, "onnx_model", None),
        "engine": getattr(args, "engine_model", None),
        "torchscript": getattr(args, "torchscript_model", None),
    }
    candidate = override_map.get(fmt) or getattr(args, "model", None)
    if not candidate:
        return None, "model_artifact_required"
    text = str(candidate)
    suffix = Path(text).suffix.lower()

    if task_label in {"segmentation", "keypoints", "depth", "pose6d"}:
        return text, None

    if fmt == "torch":
        if override_map.get(fmt):
            return text, None
        if suffix in {".pt", ".pth", ".ckpt"} or not suffix:
            return text, None
        return None, "model_artifact_mismatch"
    if fmt == "onnx":
        if override_map.get(fmt):
            return text, None
        if suffix == ".onnx":
            return text, None
        return None, "model_artifact_required"
    if fmt == "engine":
        if override_map.get(fmt):
            return text, None
        if suffix in {".engine", ".plan"}:
            return text, None
        return None, "model_artifact_required"
    if fmt == "torchscript":
        if override_map.get(fmt):
            return text, None
        if suffix in {".torchscript", ".ts", ".pt", ".pth"}:
            return text, None
        return None, "model_artifact_required"
    return None, "unsupported_format"


def _depth_mask_path(args: Any) -> Path | None:
    return _resolve_path(getattr(args, "depth_mask", None))


def _depth_align(args: Any) -> str:
    return str(getattr(args, "depth_align", "median_scale") or "median_scale")


def _tool_path(name: str) -> str:
    return str((repo_root / "tools" / name).resolve())


def _run_command(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, float(time.perf_counter() - start)


def _result_tail(proc: subprocess.CompletedProcess[str], *, max_chars: int = 2000) -> str:
    tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if len(tail) <= max_chars:
        return tail
    return tail[-max_chars:]


def _load_json_payload(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _prediction_entry_count(path: Path) -> int:
    payload = _load_json_payload(path)
    if isinstance(payload, dict):
        preds = payload.get("predictions")
        if isinstance(preds, list):
            return len(preds)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _eval_metrics(path: Path) -> dict[str, Any] | None:
    payload = _load_json_payload(path)
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return payload


def _parity_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    results = report.get("results") or []
    failure_images = 0
    total_failures = 0
    matched = 0
    extra_candidate = 0
    score_abs_max = 0.0
    bbox_abs_max = 0.0

    for item in results:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok", False)):
            failure_images += 1
        counts = item.get("counts") or {}
        matched += int(counts.get("matched") or 0)
        extra_candidate += int(counts.get("extra_cand") or 0)
        failures = item.get("failures") or []
        total_failures += len(failures)
        for failure in failures:
            if not isinstance(failure, dict) or failure.get("type") != "value_mismatch":
                continue
            ref = failure.get("ref") or {}
            cand = failure.get("cand") or {}
            try:
                score_abs_max = max(score_abs_max, abs(float(ref.get("score")) - float(cand.get("score"))))
            except (TypeError, ValueError):
                continue
            ref_bbox = ref.get("bbox") or {}
            cand_bbox = cand.get("bbox") or {}
            for key in ("cx", "cy", "w", "h"):
                try:
                    bbox_abs_max = max(bbox_abs_max, abs(float(ref_bbox.get(key)) - float(cand_bbox.get(key))))
                except (TypeError, ValueError):
                    continue

    return {
        "images": int(report.get("images") or 0),
        "ok": bool(report.get("ok", False)),
        "failure_images": int(failure_images),
        "total_failures": int(total_failures),
        "matched": int(matched),
        "extra_candidate": int(extra_candidate),
        "score_abs_max": float(score_abs_max),
        "bbox_abs_max": float(bbox_abs_max),
    }


def _write_parity_reference(
    path: Path,
    *,
    fmt: str,
    candidate_backends: list[str],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "benchmark_parity_reference",
        "format": fmt,
        "status": "reference",
        "reference_backend": fmt,
        "candidate_backends": list(candidate_backends),
        "summary": {
            "reference_backend": fmt,
            "candidate_backends": list(candidate_backends),
            "comparisons": int(len(candidate_backends)),
        },
        "timestamp": now_utc_iso(),
        "run_meta": run_meta,
    }
    write_json(path, payload)
    return payload


def _select_parity_reference(
    results: list[dict[str, Any]],
    *,
    args: Any,
    eligible: callable,
) -> dict[str, Any] | None:
    preferred = str(getattr(args, "parity_reference_backend", "auto") or "auto").strip().lower()
    if preferred and preferred != "auto":
        for item in results:
            if eligible(item) and str(item.get("format", "")).lower() == preferred:
                return item
    for item in results:
        if eligible(item) and str(item.get("format")) == "torch":
            return item
    for item in results:
        if eligible(item):
            return item
    return None


def _single_pass_metrics(elapsed_s: float, *, images: int) -> dict[str, Any]:
    total_ms = max(float(elapsed_s), 0.0) * 1000.0
    denom = max(int(images), 1)
    per_image_ms = total_ms / float(denom)
    fps = float(images / elapsed_s) if (images > 0 and elapsed_s > 0.0) else 0.0
    rounded = round(per_image_ms, 3)
    return {
        "total_sec": round(float(elapsed_s), 6),
        "images": int(images),
        "fps": round(fps, 6),
        "latency_ms": {
            "mean": rounded,
            "p50": rounded,
            "p90": rounded,
            "p95": rounded,
            "p99": rounded,
            "min": rounded,
            "max": rounded,
        },
    }


def _prediction_command(args: Any, *, fmt: str, model_artifact: str, output: Path) -> list[str]:
    base = [sys.executable]
    data = str(getattr(args, "data", ""))
    split = getattr(args, "split", None)
    max_images = getattr(args, "max_images", None)

    if fmt == "torch":
        protocol = _prediction_protocol(args)
        cmd = base + [
            _tool_path("export_predictions_ultralytics.py"),
            "--model",
            str(model_artifact),
            "--dataset",
            data,
            "--output",
            str(output),
            "--image-size",
            str(int(getattr(args, "imgsz", 640))),
            "--batch",
            str(int(getattr(args, "batch", 1))),
            "--device",
            str(getattr(args, "device", "cpu")),
            "--wrap",
            "--strict",
        ]
        if split:
            cmd.extend(["--split", str(split)])
        if max_images is not None:
            cmd.extend(["--max-images", str(int(max_images))])
        if bool(getattr(args, "half", False)):
            cmd.append("--half")
        if bool(getattr(args, "nms", False)):
            cmd.append("--no-end2end")
            if protocol is None:
                protocol = "nms_applied"
        else:
            cmd.append("--end2end")
            if protocol is None:
                protocol = "e2e_nms_free"
        if protocol:
            cmd.extend(["--protocol", protocol])
        return cmd

    if fmt == "onnx":
        cmd = base + [
            _tool_path("export_predictions_onnxrt.py"),
            "--dataset",
            data,
            "--onnx",
            str(model_artifact),
            "--output",
            str(output),
            "--wrap",
            "--strict",
        ]
        if split:
            cmd.extend(["--split", str(split)])
        if max_images is not None:
            cmd.extend(["--max-images", str(int(max_images))])
        return cmd

    if fmt == "engine":
        cmd = base + [
            _tool_path("export_predictions_trt.py"),
            "--dataset",
            data,
            "--engine",
            str(model_artifact),
            "--output",
            str(output),
            "--wrap",
            "--strict",
            "--imgsz",
            str(int(getattr(args, "imgsz", 640))),
        ]
        if split:
            cmd.extend(["--split", str(split)])
        if max_images is not None:
            cmd.extend(["--max-images", str(int(max_images))])
        return cmd

    if fmt == "torchscript":
        cmd = base + [
            _tool_path("export_predictions_torchscript.py"),
            "--dataset",
            data,
            "--model",
            str(model_artifact),
            "--output",
            str(output),
            "--wrap",
            "--strict",
            "--imgsz",
            str(int(getattr(args, "imgsz", 640))),
            "--device",
            str(getattr(args, "device", "cpu")),
        ]
        if split:
            cmd.extend(["--split", str(split)])
        if max_images is not None:
            cmd.extend(["--max-images", str(int(max_images))])
        return cmd

    raise ValueError(f"real backend command not implemented for format: {fmt}")


def _eval_command(args: Any, *, predictions_path: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        _tool_path("eval_suite.py"),
        "--dataset",
        str(getattr(args, "data", "")),
        "--predictions-glob",
        str(predictions_path),
        "--output",
        str(output),
        "--strict",
    ]
    split = getattr(args, "split", None)
    if split:
        cmd.extend(["--split", str(split)])
    protocol = getattr(args, "protocol", None)
    if protocol:
        cmd.extend(["--protocol", str(protocol)])
    max_images = getattr(args, "max_images", None)
    if max_images is not None:
        cmd.extend(["--max-images", str(int(max_images))])
    return cmd


def _depth_eval_command(args: Any, *, pred_depth_path: Path, gt_depth_path: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        _tool_path("eval_depth.py"),
        "--pred-depth",
        str(pred_depth_path),
        "--gt-depth",
        str(gt_depth_path),
        "--align",
        _depth_align(args),
        "--output",
        str(output),
    ]
    mask_path = _depth_mask_path(args)
    if mask_path is not None:
        cmd.extend(["--mask", str(mask_path)])
    return cmd


def _pose_eval_command(args: Any, *, predictions_path: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        _tool_path("eval_pose.py"),
        "--dataset",
        str(getattr(args, "data", "")),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output),
    ]
    split = getattr(args, "split", None)
    if split:
        cmd.extend(["--split", str(split)])
    max_images = getattr(args, "max_images", None)
    if max_images is not None:
        cmd.extend(["--max-images", str(int(max_images))])
    return cmd


def _segmentation_dataset_json(args: Any) -> Path | None:
    data = _resolve_path(getattr(args, "data", None))
    if data is None:
        return None
    if data.is_dir():
        candidate = data / "dataset.json"
        return candidate if candidate.exists() else None
    return data if data.exists() else None


def _segmentation_eval_command(args: Any, *, predictions_path: Path, output: Path) -> list[str]:
    dataset_json = _segmentation_dataset_json(args)
    cmd = [
        sys.executable,
        _tool_path("eval_segmentation.py"),
        "--dataset-json",
        str(dataset_json) if dataset_json is not None else str(getattr(args, "data", "")),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output),
    ]
    max_images = getattr(args, "max_images", None)
    if max_images is not None:
        cmd.extend(["--max-samples", str(int(max_images))])
    return cmd


def _keypoints_eval_command(args: Any, *, predictions_path: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        _tool_path("eval_keypoints.py"),
        "--dataset",
        str(getattr(args, "data", "")),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output),
        "--per-image-limit",
        "0",
    ]
    split = getattr(args, "split", None)
    if split:
        cmd.extend(["--split", str(split)])
    max_images = getattr(args, "max_images", None)
    if max_images is not None:
        cmd.extend(["--max-images", str(int(max_images))])
    return cmd


def _write_depth_predictions_artifact(
    path: Path,
    *,
    fmt: str,
    source_path: Path,
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    arr = load_depth_array(source_path)
    payload = {
        "schema_version": 1,
        "kind": "benchmark_depth_predictions_artifact",
        "format": fmt,
        "status": "reference_artifact",
        "source_path": str(source_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sha256": _sha256_file(source_path),
        "timestamp": now_utc_iso(),
        "run_meta": run_meta,
    }
    write_json(path, payload)
    return payload


def _copy_predictions_artifact(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source_path), str(target_path))


def _write_segmentation_predictions_artifact(
    path: Path,
    *,
    fmt: str,
    source_path: Path,
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    entries, meta = load_segmentation_predictions_entries(source_path)
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        mask = item.get("mask")
        if isinstance(mask, str):
            resolved = Path(mask)
            if not resolved.is_absolute():
                resolved = (source_path.parent / resolved).resolve()
            item["mask"] = str(resolved)
        normalized.append(item)
    payload = {
        "schema_version": 1,
        "kind": "benchmark_segmentation_predictions_artifact",
        "format": fmt,
        "status": "reference_artifact",
        "predictions": normalized,
        "meta": {
            "source_path": str(source_path),
            "source_sha256": _sha256_file(source_path),
            "predictions_meta": meta,
        },
        "timestamp": now_utc_iso(),
        "run_meta": run_meta,
    }
    write_json(path, payload)
    return payload


def _export_settings_payload(
    args: Any,
    *,
    fmt: str,
    supported: bool,
    skip_reason: str | None,
    benchmark_source: str,
    task_label: str,
    task_requested: str,
    task_semantics: dict[str, Any],
    execution_semantics: dict[str, Any],
    model_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "benchmark_export_settings",
        "format": fmt,
        "status": "supported" if supported else "skipped",
        "skip_reason": skip_reason,
        "task": task_label,
        "task_requested": task_requested,
        "task_semantics": task_semantics,
        "execution_semantics": execution_semantics,
        "model": getattr(args, "model", None),
        "model_artifact": model_artifact,
        "data": getattr(args, "data", None),
        "split": getattr(args, "split", None),
        "protocol": getattr(args, "protocol", None),
        "imgsz": int(getattr(args, "imgsz", 640)),
        "device": str(getattr(args, "device", "cpu")),
        "precision": {
            "half": bool(getattr(args, "half", False)),
            "int8": bool(getattr(args, "int8", False)),
        },
        "batch": int(getattr(args, "batch", 1)),
        "dynamic": bool(getattr(args, "dynamic", False)),
        "nms": bool(getattr(args, "nms", False)),
        "simplify": bool(getattr(args, "simplify", False)),
        "opset": int(getattr(args, "opset", 17)),
        "workspace": float(getattr(args, "workspace", 4.0)),
        "fraction": float(getattr(args, "fraction", 1.0)),
        "benchmark_source": benchmark_source,
        "latency_probe": {
            "source": benchmark_source,
            "iterations": int(getattr(args, "iterations", 50)),
            "warmup": int(getattr(args, "warmup", 5)),
            "sleep_s": float(getattr(args, "sleep_s", 0.0)),
        },
    }


def _synthetic_result(args: Any, *, fmt: str) -> tuple[str, Any, Any, Any]:
    metrics = measure_latency(
        iterations=int(getattr(args, "iterations", 50)),
        warmup=int(getattr(args, "warmup", 5)),
        sleep_s=float(getattr(args, "sleep_s", 0.0)),
    )
    latency = metrics.get("latency_ms")
    throughput = {"fps": float(metrics.get("fps", 0.0))}
    return "ok", latency, throughput, {"mode": "synthetic_step"}


def _attach_real_parity(results: list[dict[str, Any]], *, args: Any) -> None:
    if results and str(results[0].get("task")) == "segmentation":
        _attach_segmentation_parity(results, args=args)
        return
    if results and str(results[0].get("task")) == "keypoints":
        _attach_keypoints_parity(results, args=args)
        return
    if results and str(results[0].get("task")) == "depth":
        _attach_depth_parity(results, args=args)
        return
    if results and str(results[0].get("task")) == "pose6d":
        _attach_pose6d_parity(results, args=args)
        return

    def _parity_eligible(item: dict[str, Any]) -> bool:
        if str(item.get("format")) not in REAL_BACKEND_FORMATS:
            return False
        if str(item.get("status")) not in {"ok", "partial"}:
            return False
        predictions = Path(str((item.get("artifacts") or {}).get("predictions") or ""))
        return predictions.exists()

    reference = _select_parity_reference(results, args=args, eligible=_parity_eligible)
    if reference is None:
        return

    ref_backend = str(reference.get("format"))
    ref_predictions = Path(str((reference.get("artifacts") or {}).get("predictions")))
    candidate_backends: list[str] = []

    for item in results:
        if item is reference or not _parity_eligible(item):
            continue
        candidate_backend = str(item.get("format"))
        candidate_predictions = Path(str((item.get("artifacts") or {}).get("predictions")))
        parity_path = Path(str((item.get("artifacts") or {}).get("parity")))
        try:
            report = compare_predictions(
                reference=ref_predictions,
                candidate=candidate_predictions,
                max_images=getattr(args, "max_images", None),
            )
            summary = _parity_summary(report)
            payload = {
                "schema_version": 1,
                "kind": "benchmark_parity_report",
                "format": candidate_backend,
                "status": "ok" if bool(report.get("ok", False)) else "drift",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "summary": summary,
                "report": report,
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = payload["summary"]
            candidate_backends.append(candidate_backend)
            if not bool(report.get("ok", False)) and str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_drift"
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "kind": "benchmark_parity_report",
                "format": candidate_backend,
                "status": "failed",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "error": str(exc),
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = {
                "ok": False,
                "error": str(exc),
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
            }
            if str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_generation_failed"
            if item.get("error"):
                item["error"] = f"{item['error']}\n{exc}"
            else:
                item["error"] = str(exc)

    reference_payload = _write_parity_reference(
        Path(str((reference.get("artifacts") or {}).get("parity"))),
        fmt=ref_backend,
        candidate_backends=candidate_backends,
        run_meta=reference.get("run_meta") or {},
    )
    reference["parity"] = reference_payload["summary"]


def _attach_depth_parity(results: list[dict[str, Any]], *, args: Any) -> None:
    def _eligible(item: dict[str, Any]) -> bool:
        if str(item.get("format")) not in REAL_BACKEND_FORMATS:
            return False
        return str(item.get("status")) in {"ok", "partial"}

    reference = _select_parity_reference(results, args=args, eligible=_eligible)
    if reference is None:
        return

    ref_pred_payload = _load_json_payload(Path(str((reference.get("artifacts") or {}).get("predictions"))))
    if not isinstance(ref_pred_payload, dict) or not ref_pred_payload.get("source_path"):
        return
    ref_arr = load_depth_array(Path(str(ref_pred_payload["source_path"])))
    mask_path = _depth_mask_path(args)
    mask_arr = load_mask_array(mask_path) if mask_path is not None else None
    ref_backend = str(reference.get("format"))
    candidate_backends: list[str] = []
    mae_atol = float(getattr(args, "depth_parity_mae_atol", 0.02))
    rmse_atol = float(getattr(args, "depth_parity_rmse_atol", 0.03))

    for item in results:
        if item is reference or not _eligible(item):
            continue
        candidate_backend = str(item.get("format"))
        candidate_payload = _load_json_payload(Path(str((item.get("artifacts") or {}).get("predictions"))))
        if not isinstance(candidate_payload, dict) or not candidate_payload.get("source_path"):
            continue
        cand_arr = load_depth_array(Path(str(candidate_payload["source_path"])))
        parity_path = Path(str((item.get("artifacts") or {}).get("parity")))
        summary = compare_depth_arrays(
            reference=ref_arr,
            candidate=cand_arr,
            mask=mask_arr,
            align=_depth_align(args),
        )
        metrics = summary.get("metrics") or {}
        ok = float(metrics.get("mae", 1e9)) <= mae_atol and float(metrics.get("rmse", 1e9)) <= rmse_atol
        payload = {
            "schema_version": 1,
            "kind": "benchmark_depth_parity_report",
            "format": candidate_backend,
            "status": "ok" if ok else "drift",
            "reference_backend": ref_backend,
            "candidate_backend": candidate_backend,
            "summary": summary,
            "thresholds": {
                "mae_atol": mae_atol,
                "rmse_atol": rmse_atol,
            },
            "timestamp": now_utc_iso(),
            "run_meta": item.get("run_meta") or {},
        }
        write_json(parity_path, payload)
        item["parity"] = payload["summary"]
        candidate_backends.append(candidate_backend)
        if not ok and str(item.get("status")) == "ok":
            item["status"] = "partial"
            item["skip_reason"] = "parity_drift"

    reference_payload = _write_parity_reference(
        Path(str((reference.get("artifacts") or {}).get("parity"))),
        fmt=ref_backend,
        candidate_backends=candidate_backends,
        run_meta=reference.get("run_meta") or {},
    )
    reference["parity"] = reference_payload["summary"]


def _segmentation_parity_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    results = report.get("results") or []
    failure_images = 0
    total_mismatched = 0
    max_mismatch_rate = 0.0
    compared = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        compared += 1
        if not bool(item.get("ok", False)):
            failure_images += 1
        total_mismatched += int(item.get("pixels_mismatched") or 0)
        max_mismatch_rate = max(max_mismatch_rate, float(item.get("mismatch_rate") or 0.0))
    return {
        "images": int(report.get("images") or compared),
        "ok": bool(report.get("ok", False)),
        "failure_images": int(failure_images),
        "total_mismatched_pixels": int(total_mismatched),
        "max_mismatch_rate": float(max_mismatch_rate),
    }


def _attach_segmentation_parity(results: list[dict[str, Any]], *, args: Any) -> None:
    def _eligible(item: dict[str, Any]) -> bool:
        if str(item.get("format")) not in REAL_BACKEND_FORMATS:
            return False
        if str(item.get("status")) not in {"ok", "partial"}:
            return False
        predictions = Path(str((item.get("artifacts") or {}).get("predictions") or ""))
        return predictions.exists()

    reference = _select_parity_reference(results, args=args, eligible=_eligible)
    if reference is None:
        return

    ref_backend = str(reference.get("format"))
    ref_predictions = Path(str((reference.get("artifacts") or {}).get("predictions")))
    candidate_backends: list[str] = []
    mismatch_atol = float(getattr(args, "segmentation_parity_mismatch_atol", 0.0))

    for item in results:
        if item is reference or not _eligible(item):
            continue
        candidate_backend = str(item.get("format"))
        candidate_predictions = Path(str((item.get("artifacts") or {}).get("predictions")))
        parity_path = Path(str((item.get("artifacts") or {}).get("parity")))
        try:
            report = compare_segmentation_predictions(
                reference=ref_predictions,
                candidate=candidate_predictions,
                mismatch_atol=mismatch_atol,
                max_samples=getattr(args, "max_images", None),
            )
            summary = _segmentation_parity_summary(report)
            payload = {
                "schema_version": 1,
                "kind": "benchmark_segmentation_parity_report",
                "format": candidate_backend,
                "status": "ok" if bool(report.get("ok", False)) else "drift",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "summary": summary,
                "report": report,
                "thresholds": {"mismatch_atol": mismatch_atol},
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = payload["summary"]
            candidate_backends.append(candidate_backend)
            if not bool(report.get("ok", False)) and str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_drift"
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "kind": "benchmark_segmentation_parity_report",
                "format": candidate_backend,
                "status": "failed",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "error": str(exc),
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = {
                "ok": False,
                "error": str(exc),
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
            }
            if str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_generation_failed"

    reference_payload = _write_parity_reference(
        Path(str((reference.get("artifacts") or {}).get("parity"))),
        fmt=ref_backend,
        candidate_backends=candidate_backends,
        run_meta=reference.get("run_meta") or {},
    )
    reference["parity"] = reference_payload["summary"]


def _keypoints_parity_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    results = report.get("results") or []
    failure_images = 0
    total_failures = 0
    matched = 0
    extra_candidate = 0
    kp_abs_max = 0.0
    visibility_mismatch = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok", False)):
            failure_images += 1
        counts = item.get("counts") or {}
        matched += int(counts.get("matched") or 0)
        extra_candidate += int(counts.get("extra_cand") or 0)
        failures = item.get("failures") or []
        total_failures += len(failures)
        for match in item.get("matches") or []:
            if not isinstance(match, dict):
                continue
            kp_abs_max = max(kp_abs_max, float(match.get("keypoints_max_abs_diff") or 0.0))
            if not bool(match.get("visibility_ok", True)):
                visibility_mismatch += 1

    return {
        "images": int(report.get("images") or 0),
        "ok": bool(report.get("ok", False)),
        "failure_images": int(failure_images),
        "total_failures": int(total_failures),
        "matched": int(matched),
        "extra_candidate": int(extra_candidate),
        "kp_abs_max": float(kp_abs_max),
        "visibility_mismatch": int(visibility_mismatch),
    }


def _attach_keypoints_parity(results: list[dict[str, Any]], *, args: Any) -> None:
    def _eligible(item: dict[str, Any]) -> bool:
        if str(item.get("format")) not in REAL_BACKEND_FORMATS:
            return False
        if str(item.get("status")) not in {"ok", "partial"}:
            return False
        predictions = Path(str((item.get("artifacts") or {}).get("predictions") or ""))
        return predictions.exists()

    reference = _select_parity_reference(results, args=args, eligible=_eligible)
    if reference is None:
        return

    ref_backend = str(reference.get("format"))
    ref_predictions = Path(str((reference.get("artifacts") or {}).get("predictions")))
    candidate_backends: list[str] = []
    iou_thresh = float(getattr(args, "keypoints_parity_iou_thresh", 0.99))
    score_atol = float(getattr(args, "keypoints_parity_score_atol", 1e-4))
    bbox_atol = float(getattr(args, "keypoints_parity_bbox_atol", 1e-4))
    kp_atol = float(getattr(args, "keypoints_parity_kp_atol", 1e-4))

    for item in results:
        if item is reference or not _eligible(item):
            continue
        candidate_backend = str(item.get("format"))
        candidate_predictions = Path(str((item.get("artifacts") or {}).get("predictions")))
        parity_path = Path(str((item.get("artifacts") or {}).get("parity")))
        try:
            report = compare_keypoints_predictions(
                reference=ref_predictions,
                candidate=candidate_predictions,
                iou_thresh=iou_thresh,
                score_atol=score_atol,
                bbox_atol=bbox_atol,
                kp_atol=kp_atol,
                max_images=getattr(args, "max_images", None),
            )
            summary = _keypoints_parity_summary(report)
            payload = {
                "schema_version": 1,
                "kind": "benchmark_keypoints_parity_report",
                "format": candidate_backend,
                "status": "ok" if bool(report.get("ok", False)) else "drift",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "summary": summary,
                "report": report,
                "thresholds": {
                    "iou_thresh": iou_thresh,
                    "score_atol": score_atol,
                    "bbox_atol": bbox_atol,
                    "kp_atol": kp_atol,
                },
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = payload["summary"]
            candidate_backends.append(candidate_backend)
            if not bool(report.get("ok", False)) and str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_drift"
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "kind": "benchmark_keypoints_parity_report",
                "format": candidate_backend,
                "status": "failed",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "error": str(exc),
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = {
                "ok": False,
                "error": str(exc),
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
            }
            if str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_generation_failed"

    reference_payload = _write_parity_reference(
        Path(str((reference.get("artifacts") or {}).get("parity"))),
        fmt=ref_backend,
        candidate_backends=candidate_backends,
        run_meta=reference.get("run_meta") or {},
    )
    reference["parity"] = reference_payload["summary"]


def _pose_parity_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    results = report.get("results") or []
    failure_images = 0
    total_failures = 0
    matched = 0
    extra_candidate = 0
    rot_deg_max = 0.0
    trans_l2_max = 0.0
    depth_abs_max = 0.0
    rot_measured = 0
    trans_measured = 0
    depth_measured = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok", False)):
            failure_images += 1
        counts = item.get("counts") or {}
        matched += int(counts.get("matched") or 0)
        extra_candidate += int(counts.get("extra_cand") or 0)
        failures = item.get("failures") or []
        total_failures += len(failures)
        for match in item.get("matches") or []:
            if not isinstance(match, dict):
                continue
            if match.get("rot_deg_diff") is not None:
                rot_measured += 1
                rot_deg_max = max(rot_deg_max, float(match["rot_deg_diff"]))
            if match.get("trans_l2_diff") is not None:
                trans_measured += 1
                trans_l2_max = max(trans_l2_max, float(match["trans_l2_diff"]))
            if match.get("depth_abs_diff") is not None:
                depth_measured += 1
                depth_abs_max = max(depth_abs_max, float(match["depth_abs_diff"]))

    return {
        "images": int(report.get("images") or 0),
        "ok": bool(report.get("ok", False)),
        "failure_images": int(failure_images),
        "total_failures": int(total_failures),
        "matched": int(matched),
        "extra_candidate": int(extra_candidate),
        "rot_measured": int(rot_measured),
        "trans_measured": int(trans_measured),
        "depth_measured": int(depth_measured),
        "rot_deg_max": float(rot_deg_max),
        "trans_l2_max": float(trans_l2_max),
        "depth_abs_max": float(depth_abs_max),
    }


def _attach_pose6d_parity(results: list[dict[str, Any]], *, args: Any) -> None:
    def _eligible(item: dict[str, Any]) -> bool:
        if str(item.get("format")) not in REAL_BACKEND_FORMATS:
            return False
        if str(item.get("status")) not in {"ok", "partial"}:
            return False
        predictions = Path(str((item.get("artifacts") or {}).get("predictions") or ""))
        return predictions.exists()

    reference = _select_parity_reference(results, args=args, eligible=_eligible)
    if reference is None:
        return

    ref_backend = str(reference.get("format"))
    ref_predictions = Path(str((reference.get("artifacts") or {}).get("predictions")))
    candidate_backends: list[str] = []
    rot_deg_atol = float(getattr(args, "pose_parity_rot_deg_atol", 1e-3))
    trans_atol = float(getattr(args, "pose_parity_trans_atol", 1e-4))
    depth_atol = float(getattr(args, "pose_parity_depth_atol", 1e-4))

    for item in results:
        if item is reference or not _eligible(item):
            continue
        candidate_backend = str(item.get("format"))
        candidate_predictions = Path(str((item.get("artifacts") or {}).get("predictions")))
        parity_path = Path(str((item.get("artifacts") or {}).get("parity")))
        try:
            report = compare_pose_predictions(
                reference=ref_predictions,
                candidate=candidate_predictions,
                max_images=getattr(args, "max_images", None),
                rot_deg_atol=rot_deg_atol,
                trans_atol=trans_atol,
                depth_atol=depth_atol,
            )
            summary = _pose_parity_summary(report)
            payload = {
                "schema_version": 1,
                "kind": "benchmark_pose6d_parity_report",
                "format": candidate_backend,
                "status": "ok" if bool(report.get("ok", False)) else "drift",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "summary": summary,
                "report": report,
                "thresholds": {
                    "rot_deg_atol": rot_deg_atol,
                    "trans_atol": trans_atol,
                    "depth_atol": depth_atol,
                },
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = payload["summary"]
            candidate_backends.append(candidate_backend)
            if not bool(report.get("ok", False)) and str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_drift"
        except Exception as exc:
            payload = {
                "schema_version": 1,
                "kind": "benchmark_pose6d_parity_report",
                "format": candidate_backend,
                "status": "failed",
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
                "error": str(exc),
                "timestamp": now_utc_iso(),
                "run_meta": item.get("run_meta") or {},
            }
            write_json(parity_path, payload)
            item["parity"] = {
                "ok": False,
                "error": str(exc),
                "reference_backend": ref_backend,
                "candidate_backend": candidate_backend,
            }
            if str(item.get("status")) == "ok":
                item["status"] = "partial"
                item["skip_reason"] = "parity_generation_failed"

    reference_payload = _write_parity_reference(
        Path(str((reference.get("artifacts") or {}).get("parity"))),
        fmt=ref_backend,
        candidate_backends=candidate_backends,
        run_meta=reference.get("run_meta") or {},
    )
    reference["parity"] = reference_payload["summary"]


def run_benchmark_mode(args: Any) -> tuple[dict[str, Any], int]:
    task_label, task_requested = _normalize_task_label(getattr(args, "task", "detect"))
    requested_formats = _expand_requested_formats(
        getattr(args, "format", None),
        device=str(getattr(args, "device", "cpu")),
        task_label=task_label,
    )
    task_semantics = _task_semantics(task_requested)
    _validate_benchmark_args(args, requested_formats, task_label=task_label)

    report_path = _resolve_path(str(getattr(args, "output", "reports/benchmark_report.json")))
    if report_path is None:
        report_path = (repo_root / "reports" / "benchmark_report.json").resolve()

    history_path = _resolve_path(getattr(args, "history", None))
    run_id = str(getattr(args, "run_id", None) or now_utc_iso().replace(":", "-"))
    model_text = str(getattr(args, "model", "") or "")
    data_text = str(getattr(args, "data", "") or "")

    run_meta_common = {
        "git_sha": _git_head(),
        "python_version": sys.version,
        "device": str(getattr(args, "device", "cpu")),
        "seed": None,
        "repro_policy": str(getattr(args, "repro_policy", "relaxed")),
        "runtime_lock": str(getattr(args, "runtime_lock", "none")),
        "weights_hash": _hash_path_hint(model_text),
        "dataset_hash": _hash_path_hint(data_text),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    results: list[dict[str, Any]] = []
    strict_failure = False

    for fmt in requested_formats:
        supported, skip_reason = _support_status_for_format(
            fmt,
            device=str(getattr(args, "device", "cpu")),
            task_label=task_label,
        )
        runtime_available = bool(supported)
        runtime_reason = skip_reason
        benchmark_source = _selected_benchmark_source(args, fmt=fmt, task_label=task_label)
        execution_semantics = _task_execution_semantics(
            task_label,
            fmt=fmt,
            benchmark_source=benchmark_source,
            dry_run=bool(getattr(args, "dry_run", False)),
        )

        export_settings_path = (report_path.parent / f"export_settings_{fmt}.json").resolve()
        predictions_path = _artifact_path(getattr(args, "predictions_output", None), fmt=fmt, default_name="predictions_{format}.json")
        eval_path = _artifact_path(getattr(args, "eval_output", None), fmt=fmt, default_name="eval_{format}.json")
        parity_path = _artifact_path(getattr(args, "parity_output", None), fmt=fmt, default_name="parity_{format}.json")

        format_run_meta = dict(run_meta_common)
        format_run_meta["backend"] = fmt
        format_run_meta["run_id"] = run_id

        latency = None
        throughput = None
        eval_metrics = None
        parity = None
        error = None
        model_artifact = None
        command_meta: dict[str, Any] | None = None

        if not supported:
            status = "skipped"
            _write_placeholder(
                predictions_path,
                kind="benchmark_predictions_skipped",
                fmt=fmt,
                status=status,
                reason=skip_reason,
                run_meta=format_run_meta,
            )
            _write_placeholder(
                eval_path,
                kind="benchmark_eval_skipped",
                fmt=fmt,
                status=status,
                reason=skip_reason,
                run_meta=format_run_meta,
            )
            _write_placeholder(
                parity_path,
                kind="benchmark_parity_skipped",
                fmt=fmt,
                status=status,
                reason=skip_reason,
                run_meta=format_run_meta,
            )
        elif bool(getattr(args, "dry_run", False)):
            status = "dry_run"
            skip_reason = None
            _write_placeholder(
                predictions_path,
                kind="benchmark_predictions_placeholder",
                fmt=fmt,
                status=status,
                reason=None,
                run_meta=format_run_meta,
            )
            _write_placeholder(
                eval_path,
                kind="benchmark_eval_placeholder",
                fmt=fmt,
                status=status,
                reason=None,
                run_meta=format_run_meta,
            )
            _write_placeholder(
                parity_path,
                kind="benchmark_parity_placeholder",
                fmt=fmt,
                status=status,
                reason=None,
                run_meta=format_run_meta,
            )
        elif benchmark_source == "synthetic_step" or fmt not in REAL_BACKEND_FORMATS:
            status, latency, throughput, command_meta = _synthetic_result(args, fmt=fmt)
            skip_reason = None
            _write_placeholder(
                predictions_path,
                kind="benchmark_predictions_placeholder",
                fmt=fmt,
                status=status,
                reason="phase1_placeholder",
                run_meta=format_run_meta,
            )
            _write_placeholder(
                eval_path,
                kind="benchmark_eval_placeholder",
                fmt=fmt,
                status=status,
                reason="phase1_placeholder",
                run_meta=format_run_meta,
            )
            _write_placeholder(
                parity_path,
                kind="benchmark_parity_placeholder",
                fmt=fmt,
                status=status,
                reason="phase1_placeholder",
                run_meta=format_run_meta,
            )
        else:
            model_artifact, artifact_reason = _resolve_model_artifact(args, fmt=fmt, task_label=task_label)
            if not model_artifact:
                status = "skipped"
                skip_reason = artifact_reason
                _write_placeholder(
                    predictions_path,
                    kind="benchmark_predictions_placeholder",
                    fmt=fmt,
                    status=status,
                    reason=skip_reason,
                    run_meta=format_run_meta,
                )
                _write_placeholder(
                    eval_path,
                    kind="benchmark_eval_placeholder",
                    fmt=fmt,
                    status=status,
                    reason=skip_reason,
                    run_meta=format_run_meta,
                )
                _write_placeholder(
                    parity_path,
                    kind="benchmark_parity_placeholder",
                    fmt=fmt,
                    status=status,
                    reason=skip_reason,
                    run_meta=format_run_meta,
                )
            else:
                if task_label == "segmentation" and benchmark_source == "artifact_eval":
                    pred_source = _resolve_path(model_artifact)
                    dataset_json = _segmentation_dataset_json(args)
                    if pred_source is None or not pred_source.exists():
                        status = "skipped"
                        skip_reason = "model_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    elif dataset_json is None or not dataset_json.exists():
                        status = "skipped"
                        skip_reason = "dataset_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    else:
                        _write_segmentation_predictions_artifact(
                            predictions_path,
                            fmt=fmt,
                            source_path=pred_source,
                            run_meta=format_run_meta,
                        )
                        eval_cmd = _segmentation_eval_command(args, predictions_path=predictions_path, output=eval_path)
                        eval_proc, eval_elapsed = _run_command(eval_cmd)
                        command_meta = {
                            "eval": {
                                "cmd": eval_cmd,
                                "returncode": int(eval_proc.returncode),
                                "elapsed_sec": round(eval_elapsed, 6),
                            }
                        }
                        latency = None
                        throughput = None
                        if int(eval_proc.returncode) != 0:
                            status = "failed"
                            skip_reason = "eval_command_failed"
                            error = _result_tail(eval_proc)
                            if not eval_path.exists():
                                _write_placeholder(
                                    eval_path,
                                    kind="benchmark_eval_placeholder",
                                    fmt=fmt,
                                    status=status,
                                    reason=skip_reason,
                                    run_meta=format_run_meta,
                                )
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason=skip_reason,
                                run_meta=format_run_meta,
                            )
                        else:
                            status = "ok"
                            skip_reason = None
                            eval_metrics = _eval_metrics(eval_path)
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason="artifact_backed_segmentation_parity_pending_attach",
                                run_meta=format_run_meta,
                            )
                elif task_label == "depth" and benchmark_source == "artifact_eval":
                    pred_source = _resolve_path(model_artifact)
                    gt_source = _resolve_path(data_text)
                    if pred_source is None or not pred_source.exists():
                        status = "skipped"
                        skip_reason = "model_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    elif gt_source is None or not gt_source.exists():
                        status = "skipped"
                        skip_reason = "dataset_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    else:
                        _write_depth_predictions_artifact(
                            predictions_path,
                            fmt=fmt,
                            source_path=pred_source,
                            run_meta=format_run_meta,
                        )
                        eval_cmd = _depth_eval_command(args, pred_depth_path=pred_source, gt_depth_path=gt_source, output=eval_path)
                        eval_proc, eval_elapsed = _run_command(eval_cmd)
                        command_meta = {
                            "eval": {
                                "cmd": eval_cmd,
                                "returncode": int(eval_proc.returncode),
                                "elapsed_sec": round(eval_elapsed, 6),
                            }
                        }
                        latency = None
                        throughput = None
                        if int(eval_proc.returncode) != 0:
                            status = "failed"
                            skip_reason = "eval_command_failed"
                            error = _result_tail(eval_proc)
                            if not eval_path.exists():
                                _write_placeholder(
                                    eval_path,
                                    kind="benchmark_eval_placeholder",
                                    fmt=fmt,
                                    status=status,
                                    reason=skip_reason,
                                    run_meta=format_run_meta,
                                )
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason=skip_reason,
                                run_meta=format_run_meta,
                            )
                        else:
                            status = "ok"
                            skip_reason = None
                            eval_metrics = _eval_metrics(eval_path)
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason="artifact_backed_depth_parity_pending_attach",
                                run_meta=format_run_meta,
                            )
                elif task_label == "keypoints" and benchmark_source == "artifact_eval":
                    pred_source = _resolve_path(model_artifact)
                    dataset_source = _resolve_path(data_text)
                    if pred_source is None or not pred_source.exists():
                        status = "skipped"
                        skip_reason = "model_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    elif dataset_source is None or not dataset_source.exists():
                        status = "skipped"
                        skip_reason = "dataset_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    else:
                        _copy_predictions_artifact(pred_source, predictions_path)
                        eval_cmd = _keypoints_eval_command(args, predictions_path=predictions_path, output=eval_path)
                        eval_proc, eval_elapsed = _run_command(eval_cmd)
                        command_meta = {
                            "eval": {
                                "cmd": eval_cmd,
                                "returncode": int(eval_proc.returncode),
                                "elapsed_sec": round(eval_elapsed, 6),
                            }
                        }
                        latency = None
                        throughput = None
                        if int(eval_proc.returncode) != 0:
                            status = "failed"
                            skip_reason = "eval_command_failed"
                            error = _result_tail(eval_proc)
                            if not eval_path.exists():
                                _write_placeholder(
                                    eval_path,
                                    kind="benchmark_eval_placeholder",
                                    fmt=fmt,
                                    status=status,
                                    reason=skip_reason,
                                    run_meta=format_run_meta,
                                )
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason=skip_reason,
                                run_meta=format_run_meta,
                            )
                        else:
                            status = "ok"
                            skip_reason = None
                            eval_metrics = _eval_metrics(eval_path)
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason="artifact_backed_keypoints_parity_pending_attach",
                                run_meta=format_run_meta,
                            )
                elif task_label == "pose6d" and benchmark_source == "artifact_eval":
                    pred_source = _resolve_path(model_artifact)
                    dataset_source = _resolve_path(data_text)
                    if pred_source is None or not pred_source.exists():
                        status = "skipped"
                        skip_reason = "model_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    elif dataset_source is None or not dataset_source.exists():
                        status = "skipped"
                        skip_reason = "dataset_artifact_required"
                        _write_placeholder(
                            predictions_path,
                            kind="benchmark_predictions_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    else:
                        _copy_predictions_artifact(pred_source, predictions_path)
                        eval_cmd = _pose_eval_command(args, predictions_path=predictions_path, output=eval_path)
                        eval_proc, eval_elapsed = _run_command(eval_cmd)
                        command_meta = {
                            "eval": {
                                "cmd": eval_cmd,
                                "returncode": int(eval_proc.returncode),
                                "elapsed_sec": round(eval_elapsed, 6),
                            }
                        }
                        latency = None
                        throughput = None
                        if int(eval_proc.returncode) != 0:
                            status = "failed"
                            skip_reason = "eval_command_failed"
                            error = _result_tail(eval_proc)
                            if not eval_path.exists():
                                _write_placeholder(
                                    eval_path,
                                    kind="benchmark_eval_placeholder",
                                    fmt=fmt,
                                    status=status,
                                    reason=skip_reason,
                                    run_meta=format_run_meta,
                                )
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason=skip_reason,
                                run_meta=format_run_meta,
                            )
                        else:
                            status = "ok"
                            skip_reason = None
                            eval_metrics = _eval_metrics(eval_path)
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason="artifact_backed_pose6d_parity_pending_attach",
                                run_meta=format_run_meta,
                            )
                else:
                    pred_cmd = _prediction_command(args, fmt=fmt, model_artifact=model_artifact, output=predictions_path)
                    pred_proc, pred_elapsed = _run_command(pred_cmd)
                    command_meta = {
                        "predictions": {
                            "cmd": pred_cmd,
                            "returncode": int(pred_proc.returncode),
                            "elapsed_sec": round(pred_elapsed, 6),
                        }
                    }
                    if int(pred_proc.returncode) != 0:
                        status = "failed"
                        skip_reason = "prediction_command_failed"
                        error = _result_tail(pred_proc)
                        if not predictions_path.exists():
                            _write_placeholder(
                                predictions_path,
                                kind="benchmark_predictions_placeholder",
                                fmt=fmt,
                                status=status,
                                reason=skip_reason,
                                run_meta=format_run_meta,
                            )
                        _write_placeholder(
                            eval_path,
                            kind="benchmark_eval_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                        _write_placeholder(
                            parity_path,
                            kind="benchmark_parity_placeholder",
                            fmt=fmt,
                            status=status,
                            reason=skip_reason,
                            run_meta=format_run_meta,
                        )
                    else:
                        image_count = _prediction_entry_count(predictions_path)
                        measured = _single_pass_metrics(pred_elapsed, images=image_count)
                        latency = measured["latency_ms"]
                        throughput = {
                            "fps": measured["fps"],
                            "images": measured["images"],
                            "dataset_pass_sec": measured["total_sec"],
                        }
                        eval_cmd = _eval_command(args, predictions_path=predictions_path, output=eval_path)
                        eval_proc, eval_elapsed = _run_command(eval_cmd)
                        command_meta["eval"] = {
                            "cmd": eval_cmd,
                            "returncode": int(eval_proc.returncode),
                            "elapsed_sec": round(eval_elapsed, 6),
                        }
                        if int(eval_proc.returncode) != 0:
                            status = "partial"
                            skip_reason = "eval_command_failed"
                            error = _result_tail(eval_proc)
                            if not eval_path.exists():
                                _write_placeholder(
                                    eval_path,
                                    kind="benchmark_eval_placeholder",
                                    fmt=fmt,
                                    status=status,
                                    reason=skip_reason,
                                    run_meta=format_run_meta,
                                )
                        else:
                            status = "ok"
                            skip_reason = None
                            eval_metrics = _eval_metrics(eval_path)
                            _write_placeholder(
                                parity_path,
                                kind="benchmark_parity_placeholder",
                                fmt=fmt,
                                status=status,
                                reason="phase2_placeholder",
                                run_meta=format_run_meta,
                            )

        if bool(getattr(args, "strict", False)) and status in {"skipped", "failed"}:
            strict_failure = True

        export_settings = _export_settings_payload(
            args,
            fmt=fmt,
            supported=supported,
            skip_reason=skip_reason,
            benchmark_source=benchmark_source,
            task_label=task_label,
            task_requested=task_requested,
            task_semantics=task_semantics,
            execution_semantics=execution_semantics,
            model_artifact=model_artifact,
        )
        write_json(export_settings_path, export_settings)

        result = {
            "schema_version": 1,
            "kind": "yolozu_benchmark_format_result",
            "task": task_label,
            "task_requested": task_requested,
            "task_semantics": task_semantics,
            "execution_semantics": execution_semantics,
            "model": model_text,
            "data": data_text,
            "split": getattr(args, "split", None),
            "protocol": getattr(args, "protocol", None),
            "imgsz": int(getattr(args, "imgsz", 640)),
            "format": fmt,
            "device": str(getattr(args, "device", "cpu")),
            "precision": {
                "half": bool(getattr(args, "half", False)),
                "int8": bool(getattr(args, "int8", False)),
            },
            "status": status,
            "skip_reason": skip_reason,
            "latency_source": benchmark_source,
            "latency": latency,
            "throughput": throughput,
            "eval_metrics": eval_metrics,
            "parity": parity,
            "model_artifact": model_artifact,
            "artifacts": {
                "predictions": str(predictions_path),
                "eval": str(eval_path),
                "parity": str(parity_path),
                "export_settings": str(export_settings_path),
            },
            "commands": command_meta,
            "error": error,
            "run_meta": format_run_meta,
        }
        _annotate_result_support(result, runtime_available=runtime_available, runtime_reason=runtime_reason)
        results.append(result)

    _attach_real_parity(results, args=args)
    for item in results:
        runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
        _annotate_result_support(
            item,
            runtime_available=bool(runtime.get("available", False)),
            runtime_reason=runtime.get("reason"),
        )

    statuses = {item["status"] for item in results}
    if statuses == {"ok"}:
        aggregate_status = "ok"
    elif statuses == {"dry_run"}:
        aggregate_status = "dry_run"
    elif statuses == {"skipped"}:
        aggregate_status = "skipped"
    elif "failed" in statuses:
        aggregate_status = "partial" if len(statuses) > 1 else "failed"
    else:
        aggregate_status = "partial"

    report = {
        "schema_version": 1,
        "kind": "yolozu_benchmark_report",
        "timestamp": now_utc_iso(),
        "task": task_label,
        "task_requested": task_requested,
        "task_semantics": task_semantics,
        "execution_semantics": {
            "by_format": {
                fmt: _task_execution_semantics(
                    task_label,
                    fmt=fmt,
                    benchmark_source=_selected_benchmark_source(args, fmt=fmt, task_label=task_label),
                    dry_run=bool(getattr(args, "dry_run", False)),
                )
                for fmt in requested_formats
            }
        },
        "model": model_text,
        "data": data_text,
        "split": getattr(args, "split", None),
        "protocol": getattr(args, "protocol", None),
        "imgsz": int(getattr(args, "imgsz", 640)),
        "format": list(requested_formats),
        "device": str(getattr(args, "device", "cpu")),
        "precision": {
            "half": bool(getattr(args, "half", False)),
            "int8": bool(getattr(args, "int8", False)),
        },
        "status": aggregate_status,
        "requested_format": str(getattr(args, "format", "all") or "all"),
        "benchmark_source": str(getattr(args, "latency_source", "auto") or "auto"),
        "support_summary": _support_summary(results, list(requested_formats)),
        "results": results,
        "artifacts": {
            "report": str(report_path),
            "history": str(history_path) if history_path else None,
        },
        "run_meta": {
            **run_meta_common,
            "run_id": run_id,
            "formats": list(requested_formats),
        },
    }
    write_json(report_path, report)
    if history_path:
        append_jsonl(history_path, report)
    return report, 2 if strict_failure else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ultralytics-parity benchmark entrypoint.")
    parser.add_argument("-m", "--model", required=True, help="Primary model/weights path recorded in the benchmark report.")
    parser.add_argument("--torch-model", default=None, help="Optional torch backend model override (typically .pt).")
    parser.add_argument("--onnx-model", default=None, help="Optional ONNX backend model override (typically .onnx).")
    parser.add_argument("--engine-model", default=None, help="Optional TensorRT engine override (typically .engine or .plan).")
    parser.add_argument("--torchscript-model", default=None, help="Optional TorchScript backend model override (typically .torchscript, .ts, or .pt).")
    parser.add_argument("-d", "--data", required=True, help="Dataset root or data.yaml path recorded in the benchmark report.")
    parser.add_argument("--depth-mask", default=None, help="Optional valid-pixel mask used for task=depth artifact evaluation.")
    parser.add_argument(
        "--depth-align",
        choices=("none", "median_scale"),
        default="median_scale",
        help="Depth artifact alignment mode for task=depth benchmark eval/parity (default: median_scale).",
    )
    parser.add_argument("--depth-parity-mae-atol", type=float, default=0.02, help="Depth parity MAE threshold (default: 0.02).")
    parser.add_argument("--depth-parity-rmse-atol", type=float, default=0.03, help="Depth parity RMSE threshold (default: 0.03).")
    parser.add_argument(
        "--segmentation-parity-mismatch-atol",
        type=float,
        default=0.0,
        help="Segmentation parity mismatch-rate tolerance (default: 0.0, exact mask match).",
    )
    parser.add_argument(
        "--parity-reference-backend",
        choices=("auto", "torch", "onnx", "engine", "torchscript"),
        default="auto",
        help="Reference backend used when writing parity artifacts (default: auto prefers torch, then first eligible backend).",
    )
    parser.add_argument("--keypoints-parity-iou-thresh", type=float, default=0.99, help="Keypoints parity IoU threshold (default: 0.99).")
    parser.add_argument("--keypoints-parity-score-atol", type=float, default=1e-4, help="Keypoints parity score tolerance (default: 1e-4).")
    parser.add_argument("--keypoints-parity-bbox-atol", type=float, default=1e-4, help="Keypoints parity bbox tolerance (default: 1e-4).")
    parser.add_argument("--keypoints-parity-kp-atol", type=float, default=1e-4, help="Keypoints parity keypoint tolerance in normalized coords (default: 1e-4).")
    parser.add_argument("--pose-parity-rot-deg-atol", type=float, default=1e-3, help="6DoF parity rotation threshold in degrees (default: 1e-3).")
    parser.add_argument("--pose-parity-trans-atol", type=float, default=1e-4, help="6DoF parity translation L2 threshold in meters (default: 1e-4).")
    parser.add_argument("--pose-parity-depth-atol", type=float, default=1e-4, help="6DoF parity depth threshold in meters (default: 1e-4).")
    parser.add_argument("-i", "--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False, help="Record FP16 intent.")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False, help="Record INT8 intent.")
    parser.add_argument("--device", default="cpu", help="Target device string (default: cpu).")
    parser.add_argument("--verbose", action="store_true", help="Print per-format status lines.")
    parser.add_argument("-f", "--format", default="all", help="Comma-separated Phase-1 formats or all.")
    parser.add_argument(
        "--task",
        default="detect",
        choices=TASK_CHOICES,
        help="Benchmark task label. Canonical tasks: detect, segmentation, classification, obb, keypoints, depth, pose6d. Aliases: detection, seg, classify, cls, pose, 6dof.",
    )
    parser.add_argument("--split", default=None, help="Dataset split label.")
    parser.add_argument(
        "--protocol",
        choices=("yolo26", "nms_applied", "e2e_nms_free"),
        default=None,
        help="Optional eval protocol passed through to eval_suite and torch exporter.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional max image count recorded in the report.")
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring and dry-run artifacts without backend runs.")
    parser.add_argument("--strict", action="store_true", help="Return exit code 2 if any requested format is skipped or fails.")
    parser.add_argument("--repro-policy", choices=("strict", "relaxed", "off"), default="relaxed")
    parser.add_argument("--runtime-lock", default="none", help="Runtime lock label recorded in run_meta.")
    parser.add_argument("--run-id", default=None, help="Optional run id (default: UTC timestamp).")
    parser.add_argument("-o", "--output", default="reports/benchmark_report.json", help="Benchmark report JSON path.")
    parser.add_argument("--history", default=None, help="Optional JSONL history file path.")
    parser.add_argument("--predictions-output", default=None, help="Optional file/dir/template for predictions artifacts.")
    parser.add_argument("--eval-output", default=None, help="Optional file/dir/template for eval artifacts.")
    parser.add_argument("--parity-output", default=None, help="Optional file/dir/template for parity artifacts.")
    parser.add_argument("--batch", type=int, default=1, help="Common batch knob (default: 1).")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False, help="Record dynamic-shape intent.")
    parser.add_argument("--nms", action=argparse.BooleanOptionalAction, default=False, help="Record export-time NMS intent.")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=False, help="Record ONNX simplify intent.")
    parser.add_argument("--opset", type=int, default=17, help="Record ONNX opset (default: 17).")
    parser.add_argument("--workspace", type=float, default=4.0, help="Record TensorRT workspace in GiB (default: 4).")
    parser.add_argument("--fraction", type=float, default=1.0, help="Record dataset fraction knob (default: 1.0).")
    parser.add_argument(
        "--latency-source",
        choices=("auto", "synthetic_step", "dataset_pass_wall_time", "artifact_eval"),
        default="auto",
        help="Benchmark source selection: auto prefers real orchestration for detect and artifact_eval for task=segmentation/keypoints/depth/pose6d.",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Synthetic latency iterations (default: 50).")
    parser.add_argument("--warmup", type=int, default=5, help="Synthetic latency warmup iterations (default: 5).")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Synthetic latency sleep per step (default: 0).")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}
    code = 2
    try:
        report, code = run_benchmark_mode(args)
    except ValueError as exc:
        parser.error(str(exc))
    if bool(args.verbose):
        for item in report.get("results", []):
            detail = item.get("skip_reason") or item.get("latency_source")
            print(f"{item.get('format')}: {item.get('status')} ({detail})")
    print(str(_resolve_path(str(args.output)) or args.output))
    return int(code)


__all__ = [
    "PHASE1_FORMATS",
    "REAL_BACKEND_FORMATS",
    "TASK_CHOICES",
    "TASK_SEMANTICS",
    "run_benchmark_mode",
    "main",
]
