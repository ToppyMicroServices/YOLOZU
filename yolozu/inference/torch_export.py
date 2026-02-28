"""PyTorch model export utilities.

Provides functions to export a live ``torch.nn.Module`` to ONNX format
using either ``torch.onnx.export`` (TorchScript-based) or the newer
``torch.export``-based dynamo export flow.

Also includes a ``torch.compile`` helper for inference acceleration.

Usage::

    from yolozu.inference.torch_export import (
        export_model_onnx,
        compile_for_inference,
    )

    # JIT-compile model for faster inference
    fast_model = compile_for_inference(model)

    # Export to ONNX
    export_model_onnx(model, sample_input, "model.onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = [
    "compile_for_inference",
    "export_model_onnx",
    "torch_compile_available",
    "torch_export_available",
]

logger = logging.getLogger(__name__)


def torch_compile_available() -> bool:
    """Return ``True`` if ``torch.compile`` is usable."""
    try:
        import torch  # type: ignore

        return hasattr(torch, "compile") and callable(torch.compile)
    except ImportError:
        return False


def torch_export_available() -> bool:
    """Return ``True`` if ``torch.onnx.export`` (dynamo) is usable."""
    try:
        import torch  # type: ignore

        return hasattr(torch, "onnx") and hasattr(torch.onnx, "export")
    except ImportError:
        return False


def compile_for_inference(
    model: Any,
    *,
    backend: str = "inductor",
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool | None = None,
) -> Any:
    """Wrap a ``torch.nn.Module`` with ``torch.compile`` for inference speedup.

    Parameters
    ----------
    model : torch.nn.Module
        The model to compile.
    backend : str
        Compiler backend (default ``"inductor"``).
    mode : str
        Optimization mode: ``"default"``, ``"reduce-overhead"``, ``"max-autotune"``.
    fullgraph : bool
        If ``True``, require the entire model to be captured in one graph.
    dynamic : bool | None
        Enable dynamic shapes (``None`` = auto).

    Returns
    -------
    torch.nn.Module
        The compiled model (or original model if ``torch.compile`` is unavailable).
    """
    if not torch_compile_available():
        logger.warning("torch.compile not available — returning uncompiled model")
        return model

    import torch  # type: ignore

    try:
        compiled = torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info("Model compiled with torch.compile (backend=%s, mode=%s)", backend, mode)
        return compiled
    except Exception as exc:
        logger.warning("torch.compile failed (%s) — returning uncompiled model", exc)
        return model


def export_model_onnx(
    model: Any,
    sample_input: Any,
    output_path: str | Path,
    *,
    opset_version: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export a ``torch.nn.Module`` to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        The model to export (must be in eval mode).
    sample_input : torch.Tensor | tuple
        Sample input(s) for tracing.
    output_path : str | Path
        Destination path for the ``.onnx`` file.
    opset_version : int
        ONNX opset version (default 17).
    input_names : list[str] | None
        Names for input tensors in the ONNX graph.
    output_names : list[str] | None
        Names for output tensors in the ONNX graph.
    dynamic_axes : dict | None
        Dictionary mapping tensor names to dynamic axis indices.

    Returns
    -------
    Path
        Path to the exported ``.onnx`` file.

    Raises
    ------
    RuntimeError
        If torch is not installed or export fails.
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torch is required for ONNX export") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.eval() if hasattr(model, "eval") else model

    with torch.no_grad():
        torch.onnx.export(
            model_eval,
            sample_input,
            str(output_path),
            opset_version=opset_version,
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=dynamic_axes,
        )

    logger.info("Exported ONNX model to %s (opset=%d)", output_path, opset_version)
    return output_path
