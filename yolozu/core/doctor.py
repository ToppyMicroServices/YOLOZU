"""Environment diagnostics (``yolozu doctor``).

Builds a structured JSON report covering Python version, package versions,
GPU capabilities, runtime backend availability, and drift hints that flag
common cross-backend parity pitfalls.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["build_doctor_report", "write_doctor_report"]


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
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version  # py3.8+

        v = version(name)
        return str(v) if v else None
    except Exception:
        return None


def _gather_git_info(*, cwd: Path) -> dict[str, Any]:
    head = _run_capture(["git", "rev-parse", "HEAD"], cwd=cwd)
    dirty = None
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=str(cwd), check=False).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(cwd), check=False).returncode != 0
        dirty = bool(unstaged or staged)
    except Exception:
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

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch_info["cuda_available"]:
            torch_info["device_count"] = int(torch.cuda.device_count())
        gpu["torch"] = torch_info
    except Exception:
        gpu["torch"] = None

    try:
        import onnxruntime as ort  # type: ignore

        gpu["onnxruntime"] = {
            "version": getattr(ort, "__version__", None),
            "providers": list(getattr(ort, "get_available_providers")()),
        }
    except Exception:
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
        },
        "onnxruntime": {
            "installed": False,
            "version": None,
            "providers": [],
            "cuda_provider": False,
            "tensorrt_provider": False,
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

        runtime["torch"] = {
            "installed": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception:
        pass

    try:
        import onnxruntime as ort  # type: ignore

        providers = list(getattr(ort, "get_available_providers")())
        runtime["onnxruntime"] = {
            "installed": True,
            "version": getattr(ort, "__version__", None),
            "providers": providers,
            "cuda_provider": "CUDAExecutionProvider" in providers,
            "tensorrt_provider": "TensorrtExecutionProvider" in providers,
        }
    except Exception:
        pass

    try:
        import tensorrt  # type: ignore

        runtime["tensorrt"]["python_module_available"] = True
        if runtime["tensorrt"].get("python_package_version") is None:
            runtime["tensorrt"]["python_package_version"] = getattr(tensorrt, "__version__", None)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        count = None
        try:
            count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            count = None
        runtime["opencv"] = {
            "python_package_version": runtime["opencv"].get("python_package_version") or getattr(cv2, "__version__", None),
            "module_available": True,
            "cuda_enabled_device_count": count,
        }
    except Exception:
        pass

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
        except Exception as exc:
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
    report["warnings"] = warnings
    report["drift_hints"] = _build_drift_hints(runtime=runtime_capabilities, tools=tools)

    exit_code = 0 if not required_errors else 1
    return report, int(exit_code)


def write_doctor_report(*, output: str | Path, cwd: Path | None = None) -> int:
    report, exit_code = build_doctor_report(cwd=cwd)

    if str(output) == "-":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return int(exit_code)

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out_path))
    return int(exit_code)
