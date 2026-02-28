"""PyTorch profiler integration for inference and training benchmarks.

Wraps ``torch.profiler`` to produce kernel-level latency traces that
complement the existing wall-clock benchmarks in ``yolozu.eval.benchmark``.

Usage::

    from yolozu.inference.profiler import profile_inference, profiler_available

    if profiler_available():
        summary = profile_inference(adapter, records, output_dir="runs/profile")
        print(summary)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = [
    "profiler_available",
    "profile_inference",
    "profile_callable",
]

logger = logging.getLogger(__name__)


def profiler_available() -> bool:
    """Return ``True`` if ``torch.profiler`` is available."""
    try:
        import torch  # type: ignore

        return hasattr(torch, "profiler") and hasattr(torch.profiler, "profile")
    except ImportError:
        return False


def profile_inference(
    adapter: Any,
    records: list[dict],
    *,
    warmup: int = 3,
    active: int = 5,
    output_dir: str | Path = "runs/profile",
    sort_by: str = "cpu_time_total",
) -> str:
    """Profile ``adapter.predict()`` calls with ``torch.profiler``.

    Parameters
    ----------
    adapter : ModelAdapter
        Any adapter implementing ``predict(records)``.
    records : list[dict]
        Input records (each with an ``"image"`` key).
    warmup : int
        Number of warmup iterations before active profiling.
    active : int
        Number of actively profiled iterations.
    output_dir : str | Path
        Directory for Chrome-trace and TensorBoard trace output.
    sort_by : str
        Column to sort the summary table by.

    Returns
    -------
    str
        Text table summarising per-operator CPU/CUDA time.

    Raises
    ------
    RuntimeError
        If ``torch.profiler`` is not available.
    """
    if not profiler_available():
        raise RuntimeError("torch.profiler is not available — install PyTorch 2.x")

    import torch  # type: ignore

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_steps = 1 + warmup + active
    n_records = len(records)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            idx = step % n_records
            adapter.predict([records[idx]])
            prof.step()

    summary = prof.key_averages().table(sort_by=sort_by, row_limit=20)
    logger.info("Profiler trace saved to %s", output_path)
    return summary


def profile_callable(
    fn: Any,
    *args: Any,
    iterations: int = 10,
    warmup: int = 3,
    sort_by: str = "cpu_time_total",
    **kwargs: Any,
) -> str:
    """Profile an arbitrary callable with ``torch.profiler``.

    Parameters
    ----------
    fn : callable
        Function to profile.
    *args, **kwargs
        Arguments forwarded to *fn*.
    iterations : int
        Total number of calls (including warmup).
    warmup : int
        Warmup iterations (not included in active profiling).
    sort_by : str
        Column to sort the summary table by.

    Returns
    -------
    str
        Text table summarising per-operator timings.
    """
    if not profiler_available():
        raise RuntimeError("torch.profiler is not available — install PyTorch 2.x")

    import torch  # type: ignore

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    active = max(1, iterations - warmup - 1)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active),
        record_shapes=True,
    ) as prof:
        for _ in range(1 + warmup + active):
            fn(*args, **kwargs)
            prof.step()

    return prof.key_averages().table(sort_by=sort_by, row_limit=20)
