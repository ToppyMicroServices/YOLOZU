"""Ultralytics-parity benchmark mode (Phase 1).

This module provides a thin, honest benchmark orchestration layer for
``yolozu benchmark`` and ``tools/benchmark_model.py``.

Phase 1 intentionally focuses on:

- the CLI surface,
- stable report and artifact wiring,
- explicit skipped statuses for unsupported formats,
- reproducibility metadata,
- a clearly labeled synthetic latency probe.

Real backend export/inference/eval orchestration is deferred to later phases.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from yolozu.eval.benchmark import measure_latency
from yolozu.eval.metrics_report import append_jsonl, now_utc_iso, write_json

repo_root = Path(__file__).resolve().parents[2]

PHASE1_FORMATS = ("torch", "onnx", "engine", "executorch", "opencv_dnn")


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip() or None
    except Exception:
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
    # For directories, keep the hash deterministic but cheap: hash names + sizes.
    h = hashlib.sha256()
    count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = str(child.relative_to(path)).replace("\\", "/")
        try:
            stat = child.stat()
            size = stat.st_size
        except Exception:
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


def _support_status_for_format(fmt: str, *, device: str) -> tuple[bool, str | None]:
    device_l = str(device or "").strip().lower()
    wants_gpu = any(tok in device_l for tok in ("cuda", "gpu", "trt", "tensorrt"))
    system = platform.system().lower()

    if fmt == "torch":
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
        try:
            subprocess.check_output(["trtexec", "--help"], stderr=subprocess.DEVNULL)
            return True, None
        except Exception:
            return False, "missing_runtime_dependency"
    return False, "unsupported_format"


def _expand_requested_formats(value: str | None, *, device: str) -> list[str]:
    requested = _split_csv(value)
    if not requested:
        requested = ["all"]
    if requested == ["all"]:
        expanded = [fmt for fmt in PHASE1_FORMATS if _support_status_for_format(fmt, device=device)[0]]
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


def _artifact_path(base: str | None, *, fmt: str, default_name: str) -> Path:
    if not base:
        return (repo_root / "reports" / default_name.format(format=fmt)).resolve()

    text = str(base)
    if "{format}" in text:
        return _resolve_path(text.format(format=fmt)) or (repo_root / "reports" / default_name.format(format=fmt))

    path = _resolve_path(text)
    assert path is not None
    if path.suffix:
        stem = path.stem
        suffix = path.suffix
        return path.with_name(f"{stem}_{fmt}{suffix}")
    return (path / default_name.format(format=fmt)).resolve()


def _write_placeholder(path: Path, *, kind: str, fmt: str, status: str, reason: str | None, run_meta: dict[str, Any]) -> None:
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


def _export_settings_payload(args: Any, *, fmt: str, supported: bool, skip_reason: str | None, benchmark_source: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "benchmark_export_settings",
        "format": fmt,
        "status": "supported" if supported else "skipped",
        "skip_reason": skip_reason,
        "task": getattr(args, "task", "detect"),
        "model": getattr(args, "model", None),
        "data": getattr(args, "data", None),
        "split": getattr(args, "split", None),
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


def run_benchmark_mode(args: Any) -> tuple[dict[str, Any], int]:
    requested_formats = _expand_requested_formats(getattr(args, "format", None), device=str(getattr(args, "device", "cpu")))

    report_path = _resolve_path(str(getattr(args, "output", "reports/benchmark_report.json")))
    if report_path is None:
        report_path = (repo_root / "reports" / "benchmark_report.json").resolve()

    history_path = _resolve_path(getattr(args, "history", None))
    run_id = str(getattr(args, "run_id", None) or now_utc_iso().replace(":", "-"))
    benchmark_source = str(getattr(args, "latency_source", "synthetic_step") or "synthetic_step")
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
        supported, skip_reason = _support_status_for_format(fmt, device=str(getattr(args, "device", "cpu")))
        export_settings_path = (report_path.parent / f"export_settings_{fmt}.json").resolve()
        predictions_path = _artifact_path(getattr(args, "predictions_output", None), fmt=fmt, default_name="predictions_{format}.json")
        eval_path = _artifact_path(getattr(args, "eval_output", None), fmt=fmt, default_name="eval_{format}.json")
        parity_path = _artifact_path(getattr(args, "parity_output", None), fmt=fmt, default_name="parity_{format}.json")

        format_run_meta = dict(run_meta_common)
        format_run_meta["backend"] = fmt
        format_run_meta["run_id"] = run_id

        if not supported:
            status = "skipped"
            latency = None
            throughput = None
            eval_metrics = None
            parity = None
            _write_placeholder(predictions_path, kind="benchmark_predictions_placeholder", fmt=fmt, status=status, reason=skip_reason, run_meta=format_run_meta)
            _write_placeholder(eval_path, kind="benchmark_eval_placeholder", fmt=fmt, status=status, reason=skip_reason, run_meta=format_run_meta)
            _write_placeholder(parity_path, kind="benchmark_parity_placeholder", fmt=fmt, status=status, reason=skip_reason, run_meta=format_run_meta)
        elif bool(getattr(args, "dry_run", False)):
            status = "dry_run"
            skip_reason = None
            latency = None
            throughput = None
            eval_metrics = None
            parity = None
            _write_placeholder(predictions_path, kind="benchmark_predictions_placeholder", fmt=fmt, status=status, reason=None, run_meta=format_run_meta)
            _write_placeholder(eval_path, kind="benchmark_eval_placeholder", fmt=fmt, status=status, reason=None, run_meta=format_run_meta)
            _write_placeholder(parity_path, kind="benchmark_parity_placeholder", fmt=fmt, status=status, reason=None, run_meta=format_run_meta)
        else:
            metrics = measure_latency(
                iterations=int(getattr(args, "iterations", 50)),
                warmup=int(getattr(args, "warmup", 5)),
                sleep_s=float(getattr(args, "sleep_s", 0.0)),
            )
            latency = metrics.get("latency_ms")
            throughput = {"fps": float(metrics.get("fps", 0.0))}
            eval_metrics = None
            parity = None
            status = "ok"
            _write_placeholder(predictions_path, kind="benchmark_predictions_placeholder", fmt=fmt, status=status, reason="phase1_placeholder", run_meta=format_run_meta)
            _write_placeholder(eval_path, kind="benchmark_eval_placeholder", fmt=fmt, status=status, reason="phase1_placeholder", run_meta=format_run_meta)
            _write_placeholder(parity_path, kind="benchmark_parity_placeholder", fmt=fmt, status=status, reason="phase1_placeholder", run_meta=format_run_meta)

        if bool(getattr(args, "strict", False)) and status == "skipped":
            strict_failure = True

        export_settings = _export_settings_payload(
            args,
            fmt=fmt,
            supported=supported,
            skip_reason=skip_reason,
            benchmark_source=benchmark_source,
        )
        write_json(export_settings_path, export_settings)

        result = {
            "schema_version": 1,
            "kind": "yolozu_benchmark_format_result",
            "task": str(getattr(args, "task", "detect")),
            "model": model_text,
            "data": data_text,
            "split": getattr(args, "split", None),
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
            "artifacts": {
                "predictions": str(predictions_path),
                "eval": str(eval_path),
                "parity": str(parity_path),
                "export_settings": str(export_settings_path),
            },
            "run_meta": format_run_meta,
        }
        results.append(result)

    statuses = {item["status"] for item in results}
    if statuses == {"ok"}:
        aggregate_status = "ok"
    elif statuses == {"dry_run"}:
        aggregate_status = "dry_run"
    elif "ok" in statuses or "dry_run" in statuses:
        aggregate_status = "partial"
    else:
        aggregate_status = "skipped"

    report = {
        "schema_version": 1,
        "kind": "yolozu_benchmark_report",
        "timestamp": now_utc_iso(),
        "task": str(getattr(args, "task", "detect")),
        "model": model_text,
        "data": data_text,
        "split": getattr(args, "split", None),
        "imgsz": int(getattr(args, "imgsz", 640)),
        "format": list(requested_formats),
        "device": str(getattr(args, "device", "cpu")),
        "precision": {
            "half": bool(getattr(args, "half", False)),
            "int8": bool(getattr(args, "int8", False)),
        },
        "status": aggregate_status,
        "requested_format": str(getattr(args, "format", "all") or "all"),
        "benchmark_source": benchmark_source,
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

    parser = argparse.ArgumentParser(description="Ultralytics-parity benchmark entrypoint (Phase 1).")
    parser.add_argument("-m", "--model", required=True, help="Model/weights path recorded in the benchmark report.")
    parser.add_argument("-d", "--data", required=True, help="Dataset root or data.yaml path recorded in the benchmark report.")
    parser.add_argument("-i", "--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False, help="Record FP16 intent.")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False, help="Record INT8 intent.")
    parser.add_argument("--device", default="cpu", help="Target device string (default: cpu).")
    parser.add_argument("--verbose", action="store_true", help="Print per-format status lines.")
    parser.add_argument("-f", "--format", default="all", help="Comma-separated Phase-1 formats or all.")
    parser.add_argument("--task", default="detect", help="Task label recorded in the report (default: detect).")
    parser.add_argument("--split", default=None, help="Dataset split label.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional max image count recorded in the report.")
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring and planned artifacts without timing runs.")
    parser.add_argument("--strict", action="store_true", help="Return exit code 2 if any requested format is skipped.")
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
        choices=("synthetic_step",),
        default="synthetic_step",
        help="Phase-1 latency source (default: synthetic_step).",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Synthetic latency iterations (default: 50).")
    parser.add_argument("--warmup", type=int, default=5, help="Synthetic latency warmup iterations (default: 5).")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Synthetic latency sleep per step (default: 0).")
    args = parser.parse_args(argv)

    report, code = run_benchmark_mode(args)
    if bool(args.verbose):
        for item in report.get("results", []):
            print(f"{item.get('format')}: {item.get('status')} ({item.get('skip_reason') or item.get('latency_source')})")
    print(str(_resolve_path(str(args.output)) or args.output))
    return int(code)


__all__ = ["PHASE1_FORMATS", "run_benchmark_mode", "main"]
