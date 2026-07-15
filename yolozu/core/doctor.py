"""Environment diagnostics (``yolozu doctor``).

Builds a structured JSON report covering Python version, package versions,
GPU capabilities, runtime backend availability, and drift hints that flag
common cross-backend parity pitfalls.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["build_doctor_report", "explain_doctor_report", "run_doctor_proof", "write_doctor_report"]


logger = logging.getLogger(__name__)

_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)
_PROBE_FALLBACK_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, OSError)
_OPTIONAL_RUNTIME_ERRORS = _OPTIONAL_IMPORT_ERRORS + _PROBE_FALLBACK_ERRORS


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_capture(cmd: list[str], *, cwd: Path | None = None, timeout_s: float = 5.0) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except UnicodeDecodeError:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        v = importlib.metadata.version(name)
        return str(v) if v else None
    except importlib.metadata.PackageNotFoundError:
        return None
    except OSError:
        return None


def _torch_mps_state(torch_module: Any) -> tuple[bool, bool]:
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is None:
        return False, False

    built = False
    available = False
    try:
        built = bool(mps_backend.is_built()) if hasattr(mps_backend, "is_built") else False
    except _PROBE_FALLBACK_ERRORS as exc:
        logger.debug("torch MPS is_built probe failed: %s", exc)
    try:
        available = bool(mps_backend.is_available()) if hasattr(mps_backend, "is_available") else False
    except _PROBE_FALLBACK_ERRORS as exc:
        logger.debug("torch MPS is_available probe failed: %s", exc)
    return built, available


def _gather_git_info(*, cwd: Path) -> dict[str, Any]:
    inside_worktree = _run_capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=cwd)
    if inside_worktree != "true":
        return {"head": None, "dirty": None}

    head = _run_capture(["git", "rev-parse", "HEAD"], cwd=cwd)
    dirty = None
    try:
        results = [
            subprocess.run(
                command,
                cwd=str(cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"])
        ]
        dirty = None if any(code not in (0, 1) for code in results) else any(code == 1 for code in results)
    except (OSError, ValueError, subprocess.SubprocessError):
        dirty = None
    return {"head": head, "dirty": dirty}


def _gather_gpu_info() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_list": None,
        "torch": None,
        "onnxruntime": None,
    }

    smi = _run_capture(["nvidia-smi", "-L"])
    if smi:
        gpu["nvidia_smi_list"] = [line.strip() for line in smi.splitlines() if line.strip()]

    try:
        import torch  # type: ignore

        mps_built, mps_available = _torch_mps_state(torch)
        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_built": bool(mps_built),
            "mps_available": bool(mps_available),
        }
        if torch_info["cuda_available"]:
            torch_info["device_count"] = int(torch.cuda.device_count())
        gpu["torch"] = torch_info
    except _OPTIONAL_IMPORT_ERRORS as exc:
        logger.debug("torch GPU probe failed: %s", exc)
        gpu["torch"] = None

    try:
        import onnxruntime as ort  # type: ignore

        gpu["onnxruntime"] = {
            "version": getattr(ort, "__version__", None),
            "providers": list(getattr(ort, "get_available_providers")()),
        }
    except _OPTIONAL_IMPORT_ERRORS as exc:
        logger.debug("onnxruntime GPU probe failed: %s", exc)
        gpu["onnxruntime"] = None

    return gpu


def _gather_runtime_capabilities(*, tools: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "cuda": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi_available": bool(tools.get("nvidia_smi")),
            "gpu_count_from_nvidia_smi": len(gpu.get("nvidia_smi_list") or []),
        },
        "torch": {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "mps_built": False,
            "mps_available": False,
        },
        "onnxruntime": {
            "installed": False,
            "version": None,
            "providers": [],
            "cuda_provider": False,
            "tensorrt_provider": False,
            "coreml_provider": False,
        },
        "tensorrt": {
            "python_package_version": _pkg_version("tensorrt"),
            "python_module_available": False,
            "trtexec_available": bool(tools.get("trtexec")),
            "trtexec_version": _run_capture(["trtexec", "--version"]) if bool(tools.get("trtexec")) else None,
        },
        "opencv": {
            "python_package_version": _pkg_version("opencv-python") or _pkg_version("opencv-python-headless"),
            "module_available": False,
            "cuda_enabled_device_count": None,
        },
    }

    try:
        import torch  # type: ignore

        mps_built, mps_available = _torch_mps_state(torch)
        runtime["torch"] = {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "mps_built": bool(mps_built),
            "mps_available": bool(mps_available),
        }
    except _OPTIONAL_RUNTIME_ERRORS as exc:
        logger.debug("torch runtime probe failed: %s", exc)

    try:
        import onnxruntime as ort  # type: ignore

        providers = list(getattr(ort, "get_available_providers")())
        runtime["onnxruntime"] = {
            "installed": True,
            "version": getattr(ort, "__version__", None),
            "providers": providers,
            "cuda_provider": "CUDAExecutionProvider" in providers,
            "tensorrt_provider": "TensorrtExecutionProvider" in providers,
            "coreml_provider": "CoreMLExecutionProvider" in providers,
        }
    except _OPTIONAL_RUNTIME_ERRORS as exc:
        logger.debug("onnxruntime probe failed: %s", exc)

    try:
        import tensorrt  # type: ignore

        runtime["tensorrt"]["python_module_available"] = True
        if runtime["tensorrt"].get("python_package_version") is None:
            runtime["tensorrt"]["python_package_version"] = getattr(tensorrt, "__version__", None)
    except _OPTIONAL_RUNTIME_ERRORS as exc:
        logger.debug("TensorRT probe failed: %s", exc)

    try:
        import cv2  # type: ignore

        count = None
        try:
            count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except _PROBE_FALLBACK_ERRORS as exc:
            logger.debug("OpenCV CUDA probe failed: %s", exc)
            count = None
        runtime["opencv"] = {
            "python_package_version": runtime["opencv"].get("python_package_version") or getattr(cv2, "__version__", None),
            "module_available": True,
            "cuda_enabled_device_count": count,
        }
    except _OPTIONAL_RUNTIME_ERRORS as exc:
        logger.debug("OpenCV probe failed: %s", exc)

    return runtime


def _build_drift_hints(*, runtime: dict[str, Any], tools: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    def _add_hint(hint_id: str, title: str, detail: str, likely_cause: str, remediation: str) -> None:
        hints.append(
            {
                "id": hint_id,
                "title": title,
                "detail": detail,
                "likely_cause": likely_cause,
                "remediation": remediation,
            }
        )

    torch_cuda = bool((runtime.get("torch") or {}).get("cuda_available"))
    ort_cuda = bool((runtime.get("onnxruntime") or {}).get("cuda_provider"))
    ort_trt = bool((runtime.get("onnxruntime") or {}).get("tensorrt_provider"))
    trtexec = bool((runtime.get("tensorrt") or {}).get("trtexec_available"))
    trt_py = bool((runtime.get("tensorrt") or {}).get("python_module_available"))
    cv_cuda_count = (runtime.get("opencv") or {}).get("cuda_enabled_device_count")
    cuda_visible = (runtime.get("cuda") or {}).get("cuda_visible_devices")

    if torch_cuda and not ort_cuda:
        _add_hint(
            "ort_no_cuda_provider",
            "Torch can use CUDA but ONNXRuntime cannot",
            "`torch.cuda.is_available()` is true, but ONNXRuntime CUDAExecutionProvider is absent.",
            "ONNXRuntime CPU build is installed or CUDA provider dependencies are missing.",
            "docs/backend_parity_matrix.md",
        )

    if ort_trt and not trtexec:
        _add_hint(
            "trt_provider_without_trtexec",
            "ONNXRuntime TensorRT provider found, but trtexec missing",
            "ORT lists TensorrtExecutionProvider, while `trtexec --version` is unavailable.",
            "TensorRT runtime pieces are partially installed on PATH.",
            "docs/tensorrt_pipeline.md",
        )

    if trtexec and not trt_py:
        _add_hint(
            "trtexec_without_py_tensorrt",
            "trtexec is available but Python TensorRT package is missing",
            "CLI tooling can build engines, but Python-level TensorRT checks may fail.",
            "System TensorRT installation exists without matching Python bindings.",
            "docs/tensorrt_pipeline.md",
        )

    if torch_cuda and cv_cuda_count == 0:
        _add_hint(
            "opencv_no_cuda",
            "Torch sees CUDA but OpenCV CUDA path is disabled",
            "OpenCV reports zero CUDA-enabled devices while Torch reports CUDA availability.",
            "Installed OpenCV wheel likely lacks CUDA support.",
            "docs/backend_parity_matrix.md",
        )

    if isinstance(cuda_visible, str) and cuda_visible.strip() in {"", "-1"} and bool(tools.get("nvidia_smi")):
        _add_hint(
            "cuda_visibility_masked",
            "CUDA devices may be masked by environment",
            "CUDA_VISIBLE_DEVICES is empty or -1 while NVIDIA runtime is present.",
            "Environment-level GPU masking can force CPU fallback and parity drift.",
            "docs/yolo26_baseline_repro.md",
        )

    if torch_cuda and (ort_cuda or ort_trt):
        _add_hint(
            "backend_kernel_variance",
            "Cross-backend numeric drift is still possible",
            "Multiple GPU runtimes are available; identical inputs can still produce small differences.",
            "Different kernels/precision paths (Torch/ORT/TRT/OpenCV) are not bit-identical.",
            "docs/onnx_export_parity.md",
        )

    return hints


def _gather_required_runtime() -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    def _check_import(name: str, *, import_name: str | None = None, version_attr: str = "__version__") -> None:
        mod_name = import_name or name
        try:
            mod = __import__(mod_name)
            checks[name] = {"available": True, "version": getattr(mod, version_attr, None)}
        except _OPTIONAL_RUNTIME_ERRORS as exc:
            checks[name] = {"available": False, "version": None, "error": repr(exc)}
            errors.append(f"missing runtime dependency: {name} ({exc})")

    _check_import("numpy")
    _check_import("Pillow", import_name="PIL")
    _check_import("PyYAML", import_name="yaml")
    return checks, errors


def build_doctor_report(*, cwd: Path | None = None) -> tuple[dict[str, Any], int]:
    from yolozu import __version__

    here = Path.cwd() if cwd is None else Path(cwd)

    required, required_errors = _gather_required_runtime()

    tools = {
        "git": bool(_run_capture(["git", "--version"], cwd=here)),
        "nvidia_smi": bool(_run_capture(["nvidia-smi", "-L"])),
        "trtexec": bool(_run_capture(["trtexec", "--version"])),
    }

    gpu_info = _gather_gpu_info()
    runtime_capabilities = _gather_runtime_capabilities(tools=tools, gpu=gpu_info)

    report: dict[str, Any] = {
        "kind": "yolozu_doctor",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "cwd": str(here),
        "yolozu": {"version": str(__version__)},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "packages": {
            "required": required,
            "installed_versions": {
                "yolozu": _pkg_version("yolozu"),
                "numpy": _pkg_version("numpy"),
                "Pillow": _pkg_version("Pillow"),
                "PyYAML": _pkg_version("PyYAML"),
                "torch": _pkg_version("torch"),
                "onnxruntime": _pkg_version("onnxruntime"),
                "tensorrt": _pkg_version("tensorrt"),
            },
        },
        "git": _gather_git_info(cwd=here) if tools["git"] else {"head": None, "dirty": None},
        "gpu": gpu_info,
        "runtime_capabilities": runtime_capabilities,
        "tools": tools,
        "guidance_links": {
            "backend_parity": "docs/backend_parity_matrix.md",
            "onnx_parity": "docs/onnx_export_parity.md",
            "tensorrt": "docs/tensorrt_pipeline.md",
            "baseline_repro": "docs/yolo26_baseline_repro.md",
        },
        "drift_hints": [],
        "warnings": [],
        "errors": list(required_errors),
    }

    warnings: list[str] = []
    # Editable installs or multiple environments can leave stale dist metadata behind.
    dist_version = (report.get("packages") or {}).get("installed_versions", {}).get("yolozu")
    module_version = str(__version__)
    if isinstance(dist_version, str) and dist_version and dist_version != module_version:
        warnings.append(
            "yolozu version mismatch: module __version__="
            f"{module_version} but installed package metadata={dist_version}. "
            "You may have multiple installs or stale editable metadata; consider reinstalling."
        )
    if tools["nvidia_smi"] is False:
        warnings.append("nvidia-smi not found (expected on Linux+NVIDIA; OK on CPU-only/macOS)")
    if tools["trtexec"] is False:
        warnings.append("trtexec not found (TensorRT engine build requires it)")
    torch_runtime = (runtime_capabilities.get("torch") or {})
    if bool(torch_runtime.get("mps_built")) and not bool(torch_runtime.get("mps_available")):
        warnings.append(
            "torch was built with MPS support, but MPS is not available at runtime. "
            "Treat macOS/MPS as a qualified path only when `torch.backends.mps.is_available()` is true. "
            "On newer macOS releases this may be an upstream PyTorch binary/runtime issue; "
            "verify with `sw_vers` and `torch.backends.mps.is_available()`."
        )
    report["warnings"] = warnings
    report["drift_hints"] = _build_drift_hints(runtime=runtime_capabilities, tools=tools)

    exit_code = 0 if not required_errors else 1
    return report, int(exit_code)


_MINIMAL_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c6360f8cf000003010101c9fe92ef"
    "0000000049454e44ae426082"
)


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def run_doctor_proof(*, output_dir: str | Path = "reports/doctor_proof", cwd: Path | None = None) -> tuple[dict[str, Any], int]:
    """Run a tiny artifact-backed validation/evaluation proof.

    The proof intentionally stays CPU-only and dependency-light: it writes a
    one-image YOLO dataset plus matching predictions, validates both contracts,
    runs the simple detection mAP evaluator, writes an eval report, and compares
    the observed metrics with pinned expected values.
    """

    here = Path.cwd() if cwd is None else Path(cwd)
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = here / out_root

    dataset_root = out_root / "toy_dataset"
    images_dir = dataset_root / "images" / "val2017"
    labels_dir = dataset_root / "labels" / "val2017"
    image_path = images_dir / "proof_0001.png"
    label_path = labels_dir / "proof_0001.txt"
    predictions_path = out_root / "known_predictions.json"
    eval_report_path = out_root / "eval_report.json"
    proof_report_path = out_root / "proof_report.json"

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str, **extra: Any) -> None:
        payload = {"name": name, "ok": bool(ok), "detail": detail}
        payload.update(extra)
        checks.append(payload)
        if not ok:
            errors.append(f"{name}: {detail}")

    try:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_MINIMAL_PNG_1X1)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        add_check("toy_dataset", True, "wrote one-image YOLO dataset")
    except OSError as exc:
        add_check("toy_dataset", False, f"failed to write toy dataset ({exc})")

    predictions_payload = {
        "schema_version": 1,
        "predictions": [
            {
                "schema_version": 2,
                "image": str(image_path),
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.99,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5},
                    }
                ],
            }
        ]
    }
    try:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(predictions_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        add_check("known_predictions", True, "wrote known predictions artifact")
    except OSError as exc:
        add_check("known_predictions", False, f"failed to write known predictions ({exc})")

    records: list[dict[str, Any]] = []
    predictions_entries: list[dict[str, Any]] = []
    try:
        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        manifest = build_manifest(dataset_root, split="val2017")
        records = list(manifest.get("images") or [])
        validation = validate_dataset_records(records, strict=True, mode="fail", check_images=True)
        warnings.extend(validation.warnings)
        add_check(
            "dataset_schema_validation",
            not validation.errors and len(records) == 1,
            "validated toy dataset schema",
            errors=validation.errors,
            records=len(records),
        )
    except Exception as exc:
        add_check("dataset_schema_validation", False, f"failed to validate toy dataset ({exc})")

    try:
        from yolozu.predictions import load_predictions_entries, validate_predictions_payload

        validation = validate_predictions_payload(predictions_payload, strict=True)
        warnings.extend(validation.warnings)
        predictions_entries = load_predictions_entries(predictions_path)
        add_check(
            "predictions_schema_validation",
            len(predictions_entries) == 1,
            "validated known predictions schema",
            entries=len(predictions_entries),
        )
    except Exception as exc:
        add_check("predictions_schema_validation", False, f"failed to validate known predictions ({exc})")

    expected_metrics = {"map50": 1.0, "map50_95": 1.0}
    observed_metrics: dict[str, float | None] = {"map50": None, "map50_95": None}
    try:
        from yolozu.simple_map import evaluate_map

        thresholds = [0.5 + 0.05 * idx for idx in range(10)]
        result = evaluate_map(records, predictions_entries, iou_thresholds=thresholds)
        observed_metrics = {
            "map50": _round_metric(result.map50),
            "map50_95": _round_metric(result.map50_95),
        }
        eval_payload = {
            "kind": "yolozu_doctor_proof_eval",
            "schema_version": 1,
            "timestamp": _now_utc(),
            "task": "detect",
            "dataset": str(dataset_root),
            "predictions": str(predictions_path),
            "metrics": observed_metrics,
            "expected_metrics": expected_metrics,
            "counts": {"images": len(records), "prediction_entries": len(predictions_entries)},
            "per_class": result.per_class,
        }
        eval_report_path.write_text(json.dumps(eval_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        add_check("report_generation", eval_report_path.is_file(), "wrote proof eval report", path=str(eval_report_path))
    except Exception as exc:
        add_check("report_generation", False, f"failed to generate proof eval report ({exc})")

    metric_matches = all(observed_metrics.get(key) == expected for key, expected in expected_metrics.items())
    add_check(
        "compare_result",
        metric_matches,
        "compared observed metrics with pinned expected values",
        expected=expected_metrics,
        observed=observed_metrics,
    )

    status = "pass" if not errors else "fail"
    proof_payload: dict[str, Any] = {
        "kind": "yolozu_doctor_proof",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "status": status,
        "artifacts": {
            "dataset": str(dataset_root),
            "predictions": str(predictions_path),
            "eval_report": str(eval_report_path),
            "proof_report": str(proof_report_path),
        },
        "checks": checks,
        "expected_metrics": expected_metrics,
        "observed_metrics": observed_metrics,
        "warnings": warnings,
        "errors": errors,
    }
    try:
        proof_report_path.write_text(json.dumps(proof_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        proof_payload["status"] = "fail"
        proof_payload.setdefault("errors", []).append(f"proof_report: failed to write proof report ({exc})")
        return proof_payload, 1

    return proof_payload, 0 if status == "pass" else 1


def _runtime_available(report: dict[str, Any], name: str) -> bool:
    runtime = (report.get("runtime_capabilities") or {}).get(name) or {}
    if name == "torch":
        return bool(runtime.get("installed"))
    if name == "onnxruntime":
        return bool(runtime.get("installed"))
    if name == "tensorrt":
        return bool(runtime.get("python_module_available") or runtime.get("trtexec_available"))
    return bool(runtime)


def _warning_action(warning: str) -> str:
    lowered = warning.lower()
    if "version mismatch" in lowered:
        return "Reinstall in the active environment, for example: python3 -m pip install -U --force-reinstall yolozu"
    if "nvidia-smi" in lowered:
        return "Ignore on CPU-only/macOS. On Linux GPU machines, install/activate the NVIDIA driver and retry."
    if "trtexec" in lowered:
        return "Ignore unless you need TensorRT. For TensorRT export/benchmarking, install TensorRT and add trtexec to PATH."
    if "mps" in lowered:
        return "Use CPU for stable checks, or verify your PyTorch/macOS MPS build before relying on macOS acceleration."
    return "Review the linked docs in guidance_links or run yolozu guide --goal debug."


def explain_doctor_report(report: dict[str, Any], *, exit_code: int) -> str:
    """Render a beginner-friendly explanation for a doctor JSON report."""

    errors = list(report.get("errors") or [])
    warnings = list(report.get("warnings") or [])
    drift_hints = list(report.get("drift_hints") or [])
    runtime = report.get("runtime_capabilities") or {}
    cuda = runtime.get("cuda") or {}
    gpu_count = int(cuda.get("gpu_count_from_nvidia_smi") or 0)
    torch_ok = _runtime_available(report, "torch")
    ort_ok = _runtime_available(report, "onnxruntime")
    trt_ok = _runtime_available(report, "tensorrt")

    status = "OK" if exit_code == 0 and not errors else "NEEDS ATTENTION"
    lines = [
        "YOLOZU doctor explanation",
        f"Status: {status}",
        "",
        "What this means:",
    ]
    if errors:
        lines.append("  Required Python packages are missing or failed to import.")
    elif warnings:
        lines.append("  Core runtime dependencies are present, but optional environment warnings were found.")
    else:
        lines.append("  Core runtime dependencies are present. You can start with validation/evaluation workflows.")

    lines.extend(
        [
            "",
            "Runtime summary:",
            f"  Python: {str(report.get('python') or '').splitlines()[0]}",
            f"  YOLOZU: {(report.get('yolozu') or {}).get('version')}",
            f"  GPU detected by nvidia-smi: {gpu_count}",
            f"  Torch runtime: {'available' if torch_ok else 'not available'}",
            f"  ONNXRuntime: {'available' if ort_ok else 'not available'}",
            f"  TensorRT: {'available' if trt_ok else 'not available'}",
        ]
    )

    if errors:
        lines.append("")
        lines.append("Must fix:")
        for error in errors:
            lines.append(f"  - {error}")
        lines.append("  Next action: python3 -m pip install -U 'yolozu[demo]'")

    if warnings:
        lines.append("")
        lines.append("Warnings and next actions:")
        for warning in warnings:
            lines.append(f"  - {warning}")
            lines.append(f"    Action: {_warning_action(str(warning))}")

    if drift_hints:
        lines.append("")
        lines.append("Drift hints:")
        for hint in drift_hints[:3]:
            if isinstance(hint, dict):
                title = hint.get("title") or hint.get("id") or "runtime drift hint"
                action = hint.get("action") or hint.get("doc") or "review linked guidance"
                lines.append(f"  - {title}: {action}")

    lines.extend(
        [
            "",
            "Recommended next commands:",
            "  yolozu guide --goal first-run",
            "  yolozu guide --goal evaluate",
            "  yolozu doctor import --dataset-from auto --dataset <dataset_root> --output -",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_doctor_report(
    *,
    output: str | Path,
    cwd: Path | None = None,
    explain: bool = False,
    proof: bool = False,
    proof_dir: str | Path = "reports/doctor_proof",
) -> int:
    report, exit_code = build_doctor_report(cwd=cwd)
    if proof:
        proof_report, proof_exit_code = run_doctor_proof(output_dir=proof_dir, cwd=cwd)
        report["proof"] = proof_report
        exit_code = max(int(exit_code), int(proof_exit_code))

    if explain:
        if str(output) != "-":
            out_path = Path(output)
            if not out_path.is_absolute():
                out_path = Path.cwd() / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Doctor JSON: {out_path}")
        print(explain_doctor_report(report, exit_code=exit_code), end="")
        return int(exit_code)

    if str(output) == "-":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return int(exit_code)

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    if proof:
        proof_artifacts = (report.get("proof") or {}).get("artifacts") or {}
        proof_report_path = proof_artifacts.get("proof_report")
        eval_report_path = proof_artifacts.get("eval_report")
        if proof_report_path:
            print(f"proof_report: {proof_report_path}")
        if eval_report_path:
            print(f"proof_eval_report: {eval_report_path}")
    return int(exit_code)
