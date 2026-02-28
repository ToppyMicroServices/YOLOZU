"""Automatic Mixed Precision utilities for YOLOZU training loops.

Provides a thin wrapper around ``torch.amp`` (PyTorch 2.x) that standardizes
AMP setup across SDFT distillation, TTA/TTT loops, and custom training scripts.

Usage::

    from yolozu.training.amp_utils import make_amp_context

    autocast_ctx, scaler = make_amp_context(device_type="cuda", dtype="float16")
    with autocast_ctx():
        loss = model(inputs)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

__all__ = [
    "make_amp_context",
    "amp_available",
]


def amp_available() -> bool:
    """Return ``True`` if ``torch.amp`` is available."""
    try:
        import torch  # type: ignore

        return hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
    except ImportError:
        return False


def make_amp_context(
    device_type: str = "cuda",
    dtype: str = "float16",
    enabled: bool = True,
):
    """Create AMP autocast context manager and optional GradScaler.

    Parameters
    ----------
    device_type : str
        Device type for ``torch.amp.autocast`` (``"cuda"``, ``"cpu"``, ``"mps"``).
    dtype : str
        Precision dtype — ``"float16"`` or ``"bfloat16"``.
    enabled : bool
        When ``False``, returns no-op context and ``None`` scaler.

    Returns
    -------
    tuple[callable, GradScaler | None]
        ``(autocast_ctx_factory, grad_scaler_or_none)``
        Call ``autocast_ctx_factory()`` to get the context manager.
    """
    import contextlib

    if not enabled or not amp_available():
        return contextlib.nullcontext, None

    import torch  # type: ignore

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(str(dtype).lower(), torch.float16)

    def _make_autocast():
        return torch.amp.autocast(device_type=device_type, dtype=torch_dtype)

    # GradScaler is only useful for float16 on CUDA.
    scaler = None
    if torch_dtype == torch.float16 and device_type == "cuda":
        scaler = torch.amp.GradScaler()

    return _make_autocast, scaler
