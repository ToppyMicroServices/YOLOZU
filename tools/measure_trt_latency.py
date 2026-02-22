import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.benchmark import measure_latency
from yolozu.metrics_report import build_report, write_json


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=None, help="Path to TensorRT engine plan (required unless --dry-run).")
    p.add_argument("--input-name", default="images", help="TensorRT input binding name (default: images).")
    p.add_argument("--shape", default="1x3x640x640", help="Input tensor shape (e.g., 1x3x640x640).")
    p.add_argument("--iterations", type=int, default=200, help="Benchmark iterations.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    p.add_argument("--output", default="reports/latency_trt.json", help="Where to write metrics report JSON.")
    p.add_argument("--notes", default=None, help="Short notes to embed in report meta.")
    p.add_argument("--dry-run", action="store_true", help="Write a report without running TensorRT.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _parse_shape(value: str) -> tuple[int, ...]:
    raw = value.replace(",", "x").lower()
    parts = [p.strip() for p in raw.split("x") if p.strip()]
    if not parts:
        raise ValueError("shape must be non-empty (e.g., 1x3x640x640)")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"invalid shape: {value}")
    return dims


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
        except Exception:
            pass
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


class _TrtRunner:
    def __init__(self, *, engine_path: Path, input_name: str):
        self.backend = _load_cuda_backend()
        self.stream = self._create_stream()

        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("tensorrt is required") from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with engine_path.open("rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        self.input_name = input_name
        self.mode = "v2"
        self.io_names: list[str] = []
        self.output_names: list[str] = []

        self.bindings: list[int] = []
        self.binding_indices: dict[str, int] = {}
        self.device_buffers: dict[str, object] = {}
        self.host_buffers: dict[str, Any] = {}
        self.last_shapes: dict[str, tuple[int, ...]] = {}

        # TensorRT 10+ uses tensor-address v3 API; older uses bindings/v2.
        if hasattr(self.context, "execute_async_v3") and hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name"):
            self.mode = "v3"
            self.io_names = [str(self.engine.get_tensor_name(i)) for i in range(int(self.engine.num_io_tensors))]
            inputs = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if self.input_name not in inputs:
                raise ValueError(f"input tensor not found: {self.input_name} (available: {inputs})")
        else:
            self.mode = "v2"
            self.bindings = [0] * int(self.engine.num_bindings)
            self.io_names = [str(self.engine.get_binding_name(i)) for i in range(int(self.engine.num_bindings))]
            self.output_names = [
                str(self.engine.get_binding_name(i)) for i in range(int(self.engine.num_bindings)) if not self.engine.binding_is_input(i)
            ]
            self.binding_indices = {str(self.engine.get_binding_name(i)): i for i in range(int(self.engine.num_bindings))}
            if self.input_name not in self.binding_indices:
                raise ValueError(f"input binding not found: {self.input_name}")

    @staticmethod
    def _cuda_error_code(result: object) -> int:
        if isinstance(result, tuple):
            if not result:
                return 1
            result = result[0]
        try:
            return int(result)
        except Exception:
            return 1

    def _cuda_check(self, result: object, *, op: str) -> None:
        code = self._cuda_error_code(result)
        if code != 0:
            raise RuntimeError(f"{op} failed (error {code})")

    def _create_stream(self):
        if self.backend.name == "pycuda":
            return self.backend.module.Stream()
        err, stream = self.backend.module.cudaStreamCreate()
        self._cuda_check(err, op="cudaStreamCreate")
        return _to_int_handle(stream)

    def _alloc(self, name: str, shape: tuple[int, ...], dtype, *, np):
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

    def _ensure_buffers(self, input_shape: tuple[int, ...], *, np):
        if self.mode == "v3":
            self.context.set_input_shape(self.input_name, input_shape)
            for name in self.io_names:
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT and name == self.input_name:
                    shape = tuple(input_shape)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                if any(dim <= 0 for dim in shape):
                    raise RuntimeError(f"tensor shape unresolved for {name}: {shape}")
                dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
                if name not in self.last_shapes or self.last_shapes[name] != shape:
                    self._alloc(name, shape, dtype, np=np)
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
                self._alloc(name, shape, dtype, np=np)
            self.bindings[i] = int(self.device_buffers[name])

    def infer(self, input_tensor, *, np) -> None:
        input_shape = tuple(input_tensor.shape)
        self._ensure_buffers(input_shape, np=np)

        input_name = self.input_name
        if self.mode == "v3":
            input_dtype = self.trt.nptype(self.engine.get_tensor_dtype(input_name))
        else:
            input_idx = self.binding_indices[input_name]
            input_dtype = self.trt.nptype(self.engine.get_binding_dtype(input_idx))
        if input_tensor.dtype != input_dtype:
            input_tensor = input_tensor.astype(input_dtype)
        host_input = self.host_buffers[input_name]
        np.copyto(host_input.reshape(input_shape), input_tensor)

        if self.backend.name == "pycuda":
            self.backend.module.memcpy_htod_async(self.device_buffers[input_name], host_input, self.stream)
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
            return

        err = self.backend.module.cudaMemcpyAsync(
            _to_int_handle(self.device_buffers[input_name]),
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    engine_path = None
    if args.engine:
        engine_path = Path(args.engine)
        if not engine_path.is_absolute():
            engine_path = repo_root / engine_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    shape = _parse_shape(str(args.shape))

    meta: dict[str, Any] = {
        "timestamp": _now_utc(),
        "exporter": "tensorrt",
        "dry_run": bool(args.dry_run),
        "git_head": _git_head(),
        "engine": None if engine_path is None else str(engine_path),
        "engine_sha256": None if engine_path is None or not engine_path.exists() else _sha256(engine_path),
        "input_name": str(args.input_name),
        "shape": "x".join(str(d) for d in shape),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "notes": args.notes,
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version,
    }

    if args.dry_run:
        metrics = measure_latency(iterations=int(args.iterations), warmup=int(args.warmup), sleep_s=0.0)
        meta["note"] = "dry-run mode: no TensorRT inference performed"
    else:
        if engine_path is None:
            raise SystemExit("--engine is required unless --dry-run is set")
        if not engine_path.exists():
            raise SystemExit(f"engine not found: {engine_path}")

        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for TensorRT latency measurement") from exc

        runner = _TrtRunner(engine_path=engine_path, input_name=str(args.input_name))
        x = np.zeros(shape, dtype=np.float32)

        def step():
            runner.infer(x, np=np)

        metrics = measure_latency(iterations=int(args.iterations), warmup=int(args.warmup), step=step)

    report = build_report(metrics=metrics, meta=meta)
    write_json(output_path, report)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
