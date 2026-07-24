#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

from yolozu.benchmark import measure_latency
from yolozu.run_record import build_run_record

logger = logging.getLogger(__name__)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _run_capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None

def _parse_cuda_version(nvidia_smi_text: str) -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", str(nvidia_smi_text))
    return None if not m else m.group(1)


def _compute_cap_to_sm(compute_cap: str) -> str | None:
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", str(compute_cap))
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"


def _nvidia_smi_info() -> dict[str, Any] | None:
    raw = _run_capture(["nvidia-smi"])
    if not raw:
        return None

    info: dict[str, Any] = {"raw": raw, "cuda_version": _parse_cuda_version(raw)}
    query = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if query:
        for line in query.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append(
                {
                    "name": parts[0],
                    "uuid": parts[1],
                    "compute_cap": parts[2],
                    "sm": _compute_cap_to_sm(parts[2]),
                    "driver_version": parts[3],
                }
            )
    info["gpus"] = gpus
    return info


def _trtexec_version(trtexec: str) -> str | None:
    return _run_capture([str(trtexec), "--version"])


def _tensorrt_py_version() -> str | None:
    try:
        import tensorrt  # type: ignore
    except Exception:
        return None
    v = getattr(tensorrt, "__version__", None)
    return None if v is None else str(v)


def _onnxruntime_info() -> dict[str, Any] | None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None
    info: dict[str, Any] = {"version": str(getattr(ort, "__version__", "")) or None}
    try:
        info["available_providers"] = list(ort.get_available_providers())
    except Exception:
        info["available_providers"] = None
    return info


def _json_file_record(path: Path | None, *, embed: bool) -> dict[str, Any] | None:
    if path is None:
        return None
    rec: dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return rec
    rec["sha256"] = _sha256(path)
    if embed:
        try:
            rec["data"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rec["embed_error"] = str(exc)
    return rec


def _nvidia_smi_memory() -> dict[str, Any] | None:
    out = _run_capture(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    first = out.splitlines()[0].strip()
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 2:
        return {"raw": out}
    try:
        used = int(float(parts[0]))
        total = int(float(parts[1]))
    except Exception:
        return {"raw": out}
    return {"used_mib": used, "total_mib": total, "raw": out}


def _quantiles(values: np.ndarray, qs: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        for q in qs:
            out[f"p{int(q)}"] = 0.0
        return out
    for q in qs:
        out[f"p{int(q)}"] = float(np.quantile(flat, q / 100.0))
    return out


def _diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"ok": False, "reason": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(diff)
    if not bool(finite.all()):
        return {
            "ok": False,
            "reason": "non_finite_diff",
            "shape": list(diff.shape),
            "non_finite": int((~finite).sum()),
        }
    q = _quantiles(diff, (50, 90, 95, 99))
    return {
        "ok": True,
        "shape": list(diff.shape),
        "max": float(diff.max()) if diff.size else 0.0,
        "mean": float(diff.mean()) if diff.size else 0.0,
        **q,
    }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = ex.sum(axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (ex / denom).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = 1.0 / (1.0 + np.exp(-x))
    return y.astype(np.float32)


def _derive_score_bbox(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if logits is None or bbox is None:
        raise ValueError("outputs must contain logits and bbox")
    probs = _softmax(logits, axis=-1)
    score = probs.max(axis=-1)
    bbox_sig = _sigmoid(bbox)
    return score.astype(np.float32), bbox_sig.astype(np.float32)


def _parse_backends(value: str) -> list[str]:
    if not value:
        return ["torch", "onnxrt", "trt"]
    out = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not out:
        return ["torch", "onnxrt", "trt"]
    for b in out:
        if b not in ("torch", "onnxrt", "trt"):
            raise ValueError(f"unknown backend: {b}")
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTDETRPose backend parity + benchmark suite (torch/onnxrt/trt).")
    p.add_argument("--config", required=True, help="rtdetr_pose JSON config.")
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint path; requires torch in --backends and full "
            "compatibility by default."
        ),
    )
    p.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help=(
            "Explicitly allow transfer/diagnostic loading of only name-and-shape "
            "matches and record status=partial in the suite report."
        ),
    )
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    p.add_argument("--image-size", type=int, default=320, help="Square input size (default: 320).")
    p.add_argument("--batch", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument("--samples", type=int, default=2, help="Number of random samples for parity (default: 2).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    p.add_argument("--onnx", default=None, help="ONNX model path (required for onnxrt).")
    p.add_argument("--engine", default=None, help="TensorRT engine plan path (required for trt).")
    p.add_argument("--input-name", default="images", help="Input tensor name/binding (default: images).")
    p.add_argument("--onnx-meta", default=None, help="Optional ONNX export meta JSON path to record (and optionally embed).")
    p.add_argument("--engine-meta", default=None, help="Optional engine build meta JSON path to record (and optionally embed).")
    p.add_argument("--embed-meta", action="store_true", help="Embed meta JSON files into the report (default: paths only).")
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec for version reporting (default: trtexec).")

    p.add_argument("--backends", default="torch,onnxrt,trt", help="Comma-separated backends (default: torch,onnxrt,trt).")
    p.add_argument("--reference", default="torch", choices=("torch",), help="Reference backend for parity (default: torch).")

    p.add_argument("--score-atol", type=float, default=1e-4, help="Score absolute tolerance (default: 1e-4).")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="BBox absolute tolerance (default: 1e-4).")

    p.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations (default: 20).")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark iterations (default: 200).")

    p.add_argument("--output", default="reports/rtdetr_pose_backend_suite.json", help="Output JSON path.")
    p.add_argument("--dry-run", action="store_true", help="Write a report without running inference.")
    return p.parse_args(argv)


def _load_model(
    *,
    config_path: Path,
    checkpoint_path: Path | None,
    device: str,
    allow_partial_checkpoint: bool = False,
):
    from rtdetr_pose.config import load_config
    from rtdetr_pose.factory import build_model

    cfg = load_config(str(config_path))
    model = build_model(cfg.model).eval()
    checkpoint_report = None
    if checkpoint_path is not None:
        from yolozu.inference.checkpoint_compatibility import (
            load_checkpoint_compatible,
        )

        checkpoint_report = load_checkpoint_compatible(
            model,
            checkpoint_path,
            config_identity=config_path,
            allow_partial=bool(allow_partial_checkpoint),
        )
    model.to(str(device))
    return model, checkpoint_report


def _infer_torch(model, *, x: np.ndarray, device: str) -> dict[str, np.ndarray]:
    import torch

    xt = torch.from_numpy(x).to(device=str(device), dtype=torch.float32)
    with torch.no_grad():
        out = model(xt)
    outputs: dict[str, np.ndarray] = {}
    for name in ("logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"):
        t = out.get(name)
        if t is None:
            continue
        outputs[name] = t.detach().cpu().numpy()
    return outputs


@dataclass
class _OnnxRtRunner:
    session: Any
    input_name: str

    @classmethod
    def create(cls, *, onnx_path: Path, input_name: str):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for onnxrt backend") from exc

        available_providers = list(ort.get_available_providers())
        preferred_order = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in preferred_order if p in available_providers] + [
            p for p in available_providers if p not in preferred_order
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        return cls(session=session, input_name=str(input_name))

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: x.astype(np.float32)})
        names = [o.name for o in self.session.get_outputs()]
        return {str(name): np.asarray(arr) for name, arr in zip(names, outputs)}


class _CudaBackend:
    def __init__(self, name: str, module):
        self.name = name
        self.module = module


def _to_int_handle(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    for attr in ("value", "handle"):
        try:
            attr_value = getattr(value, attr)
            if attr_value is not None:
                return int(attr_value)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("failed to read CUDA handle attr %s from %r: %s", attr, value, exc)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError(f"failed to convert CUDA handle/pointer to int: {value!r}") from exc


def _load_cuda_backend() -> _CudaBackend:
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401

        return _CudaBackend("pycuda", cuda)
    except Exception:
        pass
    try:
        from cuda.bindings import runtime as cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        pass
    try:
        from cuda import cudart  # type: ignore

        return _CudaBackend("cuda", cudart)
    except Exception:
        raise RuntimeError("CUDA bindings not found (install pycuda or cuda-python)")


@dataclass
class _TrtRunner:
    engine: Any
    context: Any
    trt: Any
    backend: _CudaBackend
    input_name: str
    mode: str  # "v2" (bindings) or "v3" (tensor addresses)
    io_names: list[str]
    output_names: list[str]

    bindings: list[int]
    binding_indices: dict[str, int]

    device_buffers: dict[str, object]
    host_buffers: dict[str, np.ndarray]
    last_shapes: dict[str, tuple[int, ...]]
    stream: Any

    @staticmethod
    def _cuda_error_code(result: object) -> int:
        if isinstance(result, tuple):
            if not result:
                return 1
            result = result[0]
        try:
            return int(result)  # enum/int
        except Exception:
            return 1

    def _cuda_check(self, result: object, *, op: str) -> None:
        code = self._cuda_error_code(result)
        if code != 0:
            raise RuntimeError(f"{op} failed (error {code})")

    @classmethod
    def create(cls, *, engine_path: Path, input_name: str):
        backend = _load_cuda_backend()
        if backend.name == "pycuda":
            stream = backend.module.Stream()
        else:
            err, stream_obj = backend.module.cudaStreamCreate()
            if cls._cuda_error_code(err) != 0:
                raise RuntimeError(f"cudaStreamCreate failed (error {int(err)})")
            stream = _to_int_handle(stream_obj)

        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("tensorrt is required for trt backend") from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        uses_v3_api = bool(
            hasattr(context, "execute_async_v3") and hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name")
        )
        api_mode = "v3" if uses_v3_api else "v2"
        io_names: list[str] = []
        output_names: list[str] = []

        bindings: list[int] = []
        binding_indices: dict[str, int] = {}
        device_buffers: dict[str, object] = {}
        host_buffers: dict[str, np.ndarray] = {}
        last_shapes: dict[str, tuple[int, ...]] = {}

        # TensorRT 10+ uses the v3 API (tensor addresses); older versions use bindings/v2.
        if uses_v3_api:
            io_names = [str(engine.get_tensor_name(i)) for i in range(int(engine.num_io_tensors))]
            input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if str(input_name) not in input_names:
                raise ValueError(f"input tensor not found: {input_name} (available: {input_names})")
        else:
            api_mode = "v2"
            bindings = [0] * int(engine.num_bindings)
            io_names = [str(engine.get_binding_name(i)) for i in range(int(engine.num_bindings))]
            output_names = [str(engine.get_binding_name(i)) for i in range(int(engine.num_bindings)) if not engine.binding_is_input(i)]
            binding_indices = {str(engine.get_binding_name(i)): i for i in range(int(engine.num_bindings))}
            if str(input_name) not in binding_indices:
                raise ValueError(f"input binding not found: {input_name}")

        return cls(
            engine=engine,
            context=context,
            trt=trt,
            backend=backend,
            input_name=str(input_name),
            mode=api_mode,
            io_names=io_names,
            output_names=output_names,
            bindings=bindings,
            binding_indices=binding_indices,
            device_buffers=device_buffers,
            host_buffers=host_buffers,
            last_shapes=last_shapes,
            stream=stream,
        )

    def _alloc(self, name: str, shape: tuple[int, ...], dtype):
        size = int(np.prod(shape))
        host = np.empty(size, dtype=dtype)
        nbytes = host.nbytes
        if self.backend.name == "pycuda":
            device = self.backend.module.mem_alloc(nbytes)
        else:
            err, device = self.backend.module.cudaMalloc(nbytes)
            self._cuda_check(err, op=f"cudaMalloc({name})")
            device = _to_int_handle(device)
        self.host_buffers[name] = host
        self.device_buffers[name] = device
        self.last_shapes[name] = shape

    def _ensure_buffers(self, input_shape: tuple[int, ...]):
        if self.mode == "v3":
            self.context.set_input_shape(self.input_name, input_shape)
            for name in self.io_names:
                tensor_mode = self.engine.get_tensor_mode(name)
                if tensor_mode == self.trt.TensorIOMode.INPUT and name == self.input_name:
                    shape = tuple(input_shape)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                if any(dim <= 0 for dim in shape):
                    raise RuntimeError(f"tensor shape unresolved for {name}: {shape}")
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                if name not in self.last_shapes or self.last_shapes[name] != shape:
                    self._alloc(name, shape, dtype)
                self.context.set_tensor_address(name, int(self.device_buffers[name]))
            return

        input_idx = self.binding_indices[self.input_name]
        self.context.set_binding_shape(input_idx, input_shape)
        for i in range(int(self.engine.num_bindings)):
            name = str(self.engine.get_binding_name(i))
            shape = tuple(self.context.get_binding_shape(i))
            if any(dim <= 0 for dim in shape):
                raise RuntimeError(f"binding shape unresolved for {name}: {shape}")
            dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
            if name not in self.last_shapes or self.last_shapes[name] != shape:
                self._alloc(name, shape, dtype)
            self.bindings[i] = int(self.device_buffers[name])

    def infer(self, x: np.ndarray) -> dict[str, np.ndarray]:
        input_shape = tuple(x.shape)
        self._ensure_buffers(input_shape)
        if self.mode == "v3":
            input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        else:
            input_idx = self.binding_indices[self.input_name]
            input_dtype = self.trt.nptype(self.engine.get_binding_dtype(input_idx))
        input_tensor = x.astype(input_dtype, copy=False)

        host_input = self.host_buffers[self.input_name]
        np.copyto(host_input.reshape(input_shape), input_tensor)

        if self.backend.name == "pycuda":
            self.backend.module.memcpy_htod_async(self.device_buffers[self.input_name], host_input, self.stream)
            if self.mode == "v3":
                ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execution failed")
            for name in self.output_names:
                host = self.host_buffers[name]
                self.backend.module.memcpy_dtoh_async(host, self.device_buffers[name], self.stream)
            self.stream.synchronize()
            return {name: self.host_buffers[name].reshape(self.last_shapes[name]).copy() for name in self.output_names}

        err = self.backend.module.cudaMemcpyAsync(
            _to_int_handle(self.device_buffers[self.input_name]),
            int(host_input.ctypes.data),
            host_input.nbytes,
            self.backend.module.cudaMemcpyKind.cudaMemcpyHostToDevice,
            _to_int_handle(self.stream),
        )
        self._cuda_check(err, op="cudaMemcpyAsync H2D")

        if self.mode == "v3":
            ok = self.context.execute_async_v3(stream_handle=_to_int_handle(self.stream))
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=_to_int_handle(self.stream))
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        for name in self.output_names:
            host = self.host_buffers[name]
            err = self.backend.module.cudaMemcpyAsync(
                int(host.ctypes.data),
                _to_int_handle(self.device_buffers[name]),
                host.nbytes,
                self.backend.module.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                _to_int_handle(self.stream),
            )
            self._cuda_check(err, op=f"cudaMemcpyAsync D2H({name})")

        err = self.backend.module.cudaStreamSynchronize(_to_int_handle(self.stream))
        self._cuda_check(err, op="cudaStreamSynchronize")

        return {name: self.host_buffers[name].reshape(self.last_shapes[name]).copy() for name in self.output_names}


def _maybe_torch_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _bench_backend(name: str, step, *, warmup: int, iterations: int) -> dict[str, Any]:
    mem_before = _nvidia_smi_memory()

    def wrapped_step():
        step()
        _maybe_torch_cuda_sync()

    metrics = measure_latency(warmup=int(warmup), iterations=int(iterations), step=wrapped_step)
    mem_after = _nvidia_smi_memory()
    return {"name": name, "metrics": metrics, "memory_before": mem_before, "memory_after": mem_after}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    out_path = _resolve(str(args.output))
    if out_path is None:
        raise SystemExit("--output is required")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    if bool(args.allow_partial_checkpoint) and not args.checkpoint:
        raise SystemExit("--allow-partial-checkpoint requires --checkpoint")

    backends = _parse_backends(str(args.backends))
    config_path = _resolve(str(args.config))
    if config_path is None or not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    checkpoint_path = _resolve(str(args.checkpoint)) if args.checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if checkpoint_path is not None and "torch" not in backends:
        raise SystemExit(
            "--checkpoint requires torch in --backends so compatibility is verified"
        )

    onnx_path = _resolve(str(args.onnx)) if args.onnx else None
    engine_path = _resolve(str(args.engine)) if args.engine else None
    onnx_meta_path = _resolve(str(args.onnx_meta)) if args.onnx_meta else None
    engine_meta_path = _resolve(str(args.engine_meta)) if args.engine_meta else None

    run_record = build_run_record(repo_root=repo_root, argv=sys.argv, args=vars(args))

    report: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "meta": {
            "run_record": run_record,
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "env": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
            "config": str(config_path),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_compatibility": None,
            "onnx": None if onnx_path is None else str(onnx_path),
            "engine": None if engine_path is None else str(engine_path),
            "onnx_meta": _json_file_record(onnx_meta_path, embed=bool(args.embed_meta)),
            "engine_meta": _json_file_record(engine_meta_path, embed=bool(args.embed_meta)),
            "system": {
                "nvidia": _nvidia_smi_info(),
                "tensorrt": {
                    "trtexec": str(args.trtexec),
                    "trtexec_version": _trtexec_version(str(args.trtexec)),
                    "tensorrt_py": _tensorrt_py_version(),
                },
                "onnxruntime": _onnxruntime_info(),
            },
            "input_name": str(args.input_name),
            "image_size": int(args.image_size),
            "batch": int(args.batch),
            "samples": int(args.samples),
            "seed": int(args.seed),
            "backends": backends,
            "dry_run": bool(args.dry_run),
        },
        "parity": {"enabled": True, "reference": "torch", "candidates": {}, "thresholds": {"score_atol": float(args.score_atol), "bbox_atol": float(args.bbox_atol)}},
        "benchmark": {"enabled": True, "results": []},
    }

    if args.dry_run:
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(out_path)
        return 0

    if np is None:  # pragma: no cover
        raise SystemExit("numpy is required for rtdetr_pose_backend_suite (pip install numpy)")

    # Input generator (same for all backends).
    rng = np.random.default_rng(int(args.seed))
    batch = max(1, int(args.batch))
    size = max(1, int(args.image_size))
    samples = max(1, int(args.samples))
    inputs = [rng.random((batch, 3, size, size), dtype=np.float32) for _ in range(samples)]

    model = None
    if "torch" in backends:
        model, checkpoint_report = _load_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=str(args.device),
            allow_partial_checkpoint=bool(args.allow_partial_checkpoint),
        )
        report["meta"]["checkpoint_compatibility"] = checkpoint_report

    onnxrt = None
    if "onnxrt" in backends:
        if onnx_path is None:
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": "missing_onnx"}
        elif not onnx_path.exists():
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"onnx_not_found:{onnx_path}"}
        else:
            try:
                onnxrt = _OnnxRtRunner.create(onnx_path=onnx_path, input_name=str(args.input_name))
            except Exception as exc:
                report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"init_failed:{exc}"}

    trt = None
    if "trt" in backends:
        if engine_path is None:
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": "missing_engine"}
        elif not engine_path.exists():
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"engine_not_found:{engine_path}"}
        else:
            try:
                trt = _TrtRunner.create(engine_path=engine_path, input_name=str(args.input_name))
            except Exception as exc:
                report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"init_failed:{exc}"}

    # Parity (torch reference).
    if model is None:
        raise SystemExit("torch backend is required as reference")

    ref_outputs = [_infer_torch(model, x=x, device=str(args.device)) for x in inputs]
    ref_scores_bbox = [_derive_score_bbox(o) for o in ref_outputs]

    def compare_candidate(name: str, infer_fn):
        cand_outputs = [infer_fn(x) for x in inputs]
        cand_scores_bbox = [_derive_score_bbox(o) for o in cand_outputs]

        score_diffs = []
        bbox_diffs = []
        for (ref_score, ref_bbox), (cand_score, cand_bbox) in zip(ref_scores_bbox, cand_scores_bbox):
            score_diffs.append(_diff_stats(ref_score, cand_score))
            bbox_diffs.append(_diff_stats(ref_bbox, cand_bbox))
        score_max = max((d.get("max", 0.0) for d in score_diffs if d.get("ok")), default=0.0)
        bbox_max = max((d.get("max", 0.0) for d in bbox_diffs if d.get("ok")), default=0.0)
        passed = bool(score_max <= float(args.score_atol) and bbox_max <= float(args.bbox_atol))
        return {
            "available": True,
            "passed": passed,
            "score": {"per_sample": score_diffs, "max_over_samples": float(score_max)},
            "bbox": {"per_sample": bbox_diffs, "max_over_samples": float(bbox_max)},
        }

    if onnxrt is not None:
        try:
            report["parity"]["candidates"]["onnxrt"] = compare_candidate("onnxrt", onnxrt.infer)
        except Exception as exc:
            report["parity"]["candidates"]["onnxrt"] = {"available": False, "reason": f"infer_failed:{exc}"}

    if trt is not None:
        try:
            report["parity"]["candidates"]["trt"] = compare_candidate("trt", trt.infer)
        except Exception as exc:
            report["parity"]["candidates"]["trt"] = {"available": False, "reason": f"infer_failed:{exc}"}

    # Benchmarks.
    bench_results = []

    # Torch.
    if model is not None:
        x0 = inputs[0]

        def torch_step():
            _infer_torch(model, x=x0, device=str(args.device))

        bench_results.append(_bench_backend("torch", torch_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    if onnxrt is not None:
        x0 = inputs[0]

        def onnxrt_step():
            onnxrt.infer(x0)

        bench_results.append(_bench_backend("onnxrt", onnxrt_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    if trt is not None:
        x0 = inputs[0]

        def trt_step():
            trt.infer(x0)

        bench_results.append(_bench_backend("trt", trt_step, warmup=int(args.warmup), iterations=int(args.iterations)))

    report["benchmark"]["results"] = bench_results

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
