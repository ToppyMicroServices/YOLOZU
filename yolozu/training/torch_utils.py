"""PyTorch utility wrappers for inference, compilation, profiling, and model analysis.

Provides thin, composable helpers that integrate core PyTorch APIs
(``torch.amp``, ``torch.compile``, ``torch.profiler``, ``torch.nn``)
into the YOLOZU workflow.  All functions guard on ``torch`` availability
and raise clear errors when it is missing.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable, Sequence

__all__ = [
    "amp_inference_context",
    "compile_model",
    "profile_callable",
    "model_info",
    "auto_device",
    "seed_everything",
    "configure_matmul_precision",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_torch():
    """Return the ``torch`` module or raise ``RuntimeError``."""
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        raise RuntimeError(
            "PyTorch is required for this feature.  "
            "Install it with: pip install torch"
        ) from None


# ---------------------------------------------------------------------------
# torch.amp – Automatic Mixed Precision
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def amp_inference_context(
    device_type: str = "cpu",
    dtype: Any = None,
    enabled: bool = True,
):
    """Context manager for AMP inference.

    Wraps ``torch.amp.autocast`` for easy mixed-precision inference.

    Parameters
    ----------
    device_type:
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    dtype:
        Override dtype (e.g. ``torch.float16``, ``torch.bfloat16``).
        *None* picks the default for the device.
    enabled:
        Set to *False* to disable AMP without changing call-site code.

    Example
    -------
    >>> with amp_inference_context("cuda", dtype=torch.float16):
    ...     output = model(images)
    """
    torch = _require_torch()
    if dtype is None:
        # sensible defaults
        if device_type == "cpu":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
        yield


# ---------------------------------------------------------------------------
# torch.compile – model compilation
# ---------------------------------------------------------------------------

def compile_model(
    model: Any,
    *,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
) -> Any:
    """Compile a PyTorch model via ``torch.compile``.

    Thin wrapper that makes the call explicit and logs the compilation
    settings.  Falls back to the uncompiled model if ``torch.compile``
    is unavailable (PyTorch < 2.0).

    Parameters
    ----------
    model:
        An ``nn.Module`` or callable.
    backend:
        Compiler backend (``"inductor"``, ``"aot_eager"``, …).
    mode:
        Compilation mode (``"default"``, ``"reduce-overhead"``,
        ``"max-autotune"``).
    fullgraph:
        Require the entire forward to compile as one graph.
    dynamic:
        Enable dynamic shapes.

    Returns
    -------
    The compiled model (or the original model if compile is unavailable).
    """
    torch = _require_torch()
    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile unavailable (PyTorch < 2.0); returning uncompiled model.",
            stacklevel=2,
        )
        return model

    kwargs: dict[str, Any] = {"backend": backend, "fullgraph": fullgraph}
    if mode is not None:
        kwargs["mode"] = mode
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    return torch.compile(model, **kwargs)


# ---------------------------------------------------------------------------
# torch.profiler – profiling helper
# ---------------------------------------------------------------------------

def profile_callable(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 2,
    active: int = 5,
    activities: Sequence[str] | None = None,
    with_stack: bool = False,
    record_shapes: bool = True,
    sort_by: str = "cpu_time_total",
    row_limit: int = 20,
) -> dict[str, Any]:
    """Profile a callable and return a summary dict.

    Uses ``torch.profiler.profile`` with a stepped schedule.

    Parameters
    ----------
    fn:
        The function to profile.
    args / kwargs:
        Positional / keyword arguments forwarded to *fn*.
    warmup / active:
        Number of warmup and active profiling steps.
    activities:
        List of activity strings (``"cpu"``, ``"cuda"``).
        Defaults to CPU only (plus CUDA if available).
    with_stack:
        Record Python call stacks.
    record_shapes:
        Record tensor shapes.
    sort_by:
        Column to sort the key-averages table.
    row_limit:
        Max rows in the summary table.

    Returns
    -------
    dict with keys ``"table"`` (str), ``"events"`` (list[dict]),
    ``"total_calls"`` (int).
    """
    torch = _require_torch()
    if kwargs is None:
        kwargs = {}

    # Resolve activities
    activity_list: list[Any] = []
    if activities is not None:
        mapping = {
            "cpu": torch.profiler.ProfilerActivity.CPU,
        }
        if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
            mapping["cuda"] = torch.profiler.ProfilerActivity.CUDA
        for a in activities:
            if a.lower() in mapping:
                activity_list.append(mapping[a.lower()])
    else:
        activity_list.append(torch.profiler.ProfilerActivity.CPU)
        if torch.cuda.is_available():
            activity_list.append(torch.profiler.ProfilerActivity.CUDA)

    schedule = torch.profiler.schedule(
        wait=0, warmup=warmup, active=active, repeat=1,
    )

    total_steps = warmup + active
    results: dict[str, Any] = {}

    with torch.profiler.profile(
        activities=activity_list,
        schedule=schedule,
        record_shapes=record_shapes,
        with_stack=with_stack,
    ) as prof:
        for _step in range(total_steps):
            fn(*args, **kwargs)
            prof.step()

    averages = prof.key_averages()
    table = averages.table(sort_by=sort_by, row_limit=row_limit)
    events = [
        {
            "name": e.key,
            "cpu_time_ms": e.cpu_time_total / 1000.0,
            "cuda_time_ms": e.cuda_time_total / 1000.0 if hasattr(e, "cuda_time_total") else 0.0,
            "calls": e.count,
        }
        for e in averages
    ]
    results["table"] = table
    results["events"] = events
    results["total_calls"] = sum(e.count for e in averages)
    return results


# ---------------------------------------------------------------------------
# Model info – parameter analysis
# ---------------------------------------------------------------------------

def model_info(model: Any) -> dict[str, Any]:
    """Return summary information about a PyTorch model.

    Parameters
    ----------
    model:
        An ``nn.Module``.

    Returns
    -------
    dict with keys:

    - ``"total_params"`` – total parameter count
    - ``"trainable_params"`` – trainable parameter count
    - ``"frozen_params"`` – non-trainable parameter count
    - ``"dtype_breakdown"`` – dict mapping dtype str → count
    - ``"device"`` – device of the first parameter (or ``"n/a"``)
    - ``"num_buffers"`` – number of registered buffers
    - ``"total_buffer_elements"`` – total buffer elements
    """
    _require_torch()
    total = 0
    trainable = 0
    dtype_counts: dict[str, int] = {}
    device_str = "n/a"

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + n
        if device_str == "n/a":
            device_str = str(p.device)

    num_buffers = 0
    buffer_elements = 0
    for b in model.buffers():
        num_buffers += 1
        buffer_elements += b.numel()

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "dtype_breakdown": dtype_counts,
        "device": device_str,
        "num_buffers": num_buffers,
        "total_buffer_elements": buffer_elements,
    }


# ---------------------------------------------------------------------------
# auto_device – pick best available device
# ---------------------------------------------------------------------------

def auto_device(*, prefer: str | None = None) -> Any:
    """Return the best available ``torch.device``.

    Priority: *prefer* (if available) → CUDA → MPS → CPU.

    Parameters
    ----------
    prefer:
        Preferred device string (e.g. ``"cuda:1"``).

    Returns
    -------
    ``torch.device``
    """
    torch = _require_torch()
    if prefer is not None:
        try:
            dev = torch.device(prefer)
            if dev.type == "cuda" and torch.cuda.is_available():
                return dev
            if dev.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return dev
            if dev.type == "cpu":
                return dev
        except Exception:
            pass  # fall through to auto-detect

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# seed_everything – global reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> int:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        The seed value.

    Returns
    -------
    The seed used (for logging convenience).
    """
    import os
    import random

    torch = _require_torch()

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (best-effort)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # older PyTorch without warn_only
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return seed


# ---------------------------------------------------------------------------
# float32 matmul precision / TF32 controls
# ---------------------------------------------------------------------------

def configure_matmul_precision(
    precision: str = "high",
    *,
    allow_tf32: bool | None = None,
) -> dict[str, Any]:
    """Configure float32 matmul precision and optional TF32 backend flags.

    Wraps stable PyTorch APIs:

    - ``torch.set_float32_matmul_precision(precision)``
    - ``torch.backends.cuda.matmul.allow_tf32``
    - ``torch.backends.cudnn.allow_tf32``

    Parameters
    ----------
    precision:
        One of ``"highest"``, ``"high"``, ``"medium"``.
    allow_tf32:
        If provided, set TF32 flags for CUDA matmul/cuDNN when available.

    Returns
    -------
    dict with keys:

    - ``"matmul_precision"``
    - ``"set_precision_supported"``
    - ``"tf32_cuda_matmul"``
    - ``"tf32_cudnn"``
    """
    torch = _require_torch()

    allowed = {"highest", "high", "medium"}
    if precision not in allowed:
        raise ValueError(f"precision must be one of {sorted(allowed)}, got: {precision!r}")

    set_precision_supported = hasattr(torch, "set_float32_matmul_precision")
    if set_precision_supported:
        torch.set_float32_matmul_precision(precision)
    else:
        warnings.warn(
            "torch.set_float32_matmul_precision is unavailable in this PyTorch version.",
            stacklevel=2,
        )

    tf32_cuda_matmul = None
    tf32_cudnn = None

    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    cudnn_backend = getattr(backends, "cudnn", None) if backends is not None else None

    if allow_tf32 is not None and cuda_backend is not None:
        matmul_backend = getattr(cuda_backend, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = bool(allow_tf32)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = bool(allow_tf32)

    if cuda_backend is not None:
        matmul_backend = getattr(cuda_backend, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            tf32_cuda_matmul = bool(matmul_backend.allow_tf32)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        tf32_cudnn = bool(cudnn_backend.allow_tf32)

    return {
        "matmul_precision": precision,
        "set_precision_supported": bool(set_precision_supported),
        "tf32_cuda_matmul": tf32_cuda_matmul,
        "tf32_cudnn": tf32_cudnn,
    }
