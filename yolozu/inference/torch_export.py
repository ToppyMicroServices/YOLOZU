"""PyTorch model export utilities.

Provides functions to export a live ``torch.nn.Module`` to ONNX format
using either ``torch.onnx.export`` (TorchScript-based) or the newer
``torch.export``-based dynamo export flow.

Also includes a ``torch.compile`` helper for inference acceleration.

Usage::

    from yolozu.inference.torch_export import (
        export_model_onnx,
        compile_for_inference,
        get_compile_evidence,
    )

    # Request compile tracking; status becomes compiled after a successful call.
    fast_model = compile_for_inference(model)
    output = fast_model(sample_input)
    compile_evidence = get_compile_evidence(fast_model)

    # Export to ONNX
    export_model_onnx(model, sample_input, "model.onnx")
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

__all__ = [
    "TorchCompileError",
    "compile_for_inference",
    "export_model_onnx",
    "get_compile_evidence",
    "torch_compile_available",
    "torch_export_available",
]

logger = logging.getLogger(__name__)


class TorchCompileError(RuntimeError):
    """Raised when requested ``torch.compile`` execution cannot be established."""

    def __init__(self, message: str, *, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = copy.deepcopy(evidence)


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


def _counter_snapshot(torch: Any) -> dict[str, int] | None:
    try:
        counters = torch._dynamo.utils.counters
    except Exception:
        return None

    snapshot: dict[str, int] = {}
    try:
        groups = counters.items()
    except Exception:
        return None
    for group, values in groups:
        try:
            items = values.items()
        except Exception:
            continue
        for name, value in items:
            if isinstance(value, bool):
                continue
            try:
                snapshot[f"{group}.{name}"] = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
    return snapshot


def _counter_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    keys = set(before) | set(after)
    return {
        key: int(after.get(key, 0) - before.get(key, 0))
        for key in sorted(keys)
        if int(after.get(key, 0) - before.get(key, 0)) != 0
    }


def _update_counter_evidence(
    evidence: dict[str, Any],
    *,
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> None:
    delta = _counter_delta(before, after)
    runtime = evidence["evidence"]
    runtime["counter_delta"] = delta
    if delta is None:
        return
    runtime["counter_source"] = "torch._dynamo.utils.counters"
    runtime["graph_count"] = delta.get("stats.unique_graphs")
    runtime["captured_call_count"] = delta.get("stats.calls_captured")
    graph_break_values = [
        value
        for key, value in delta.items()
        if key == "graph_break" or key.startswith("graph_break.")
    ]
    runtime["graph_break_count"] = (
        int(sum(graph_break_values)) if graph_break_values else None
    )


def _new_compile_evidence(
    *,
    backend: str,
    mode: str,
    fullgraph: bool,
    dynamic: bool | None,
    allow_fallback: bool,
    compile_api_available: bool,
) -> dict[str, Any]:
    return {
        "requested": {
            "enabled": True,
            "backend": str(backend),
            "mode": str(mode),
            "fullgraph": bool(fullgraph),
            "dynamic": dynamic,
            "allow_fallback": bool(allow_fallback),
        },
        "actual": {
            "status": "pending_first_execution",
            "backend": None,
            "mode": None,
            "fullgraph": None,
            "dynamic": None,
        },
        "evidence": {
            "compile_api_available": bool(compile_api_available),
            "setup_completed": False,
            "first_execution_completed": False,
            "fallback_execution_completed": False,
            "counter_source": None,
            "counter_delta": None,
            "graph_count": None,
            "graph_break_count": None,
            "captured_call_count": None,
        },
        "failure": None,
    }


def _set_compile_failure(
    evidence: dict[str, Any],
    *,
    phase: str,
    exc: BaseException,
    fallback: bool,
) -> None:
    evidence["actual"] = {
        "status": "fallback" if fallback else "failed",
        "backend": "eager" if fallback else None,
        "mode": None,
        "fullgraph": False if fallback else None,
        "dynamic": None,
    }
    evidence["failure"] = {
        "phase": str(phase),
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _attach_compile_evidence(model: Any, evidence: dict[str, Any]) -> None:
    try:
        object.__setattr__(model, "_yolozu_compile_evidence", evidence)
    except (AttributeError, TypeError):
        setattr(model, "_yolozu_compile_evidence", evidence)


def get_compile_evidence(model: Any) -> dict[str, Any] | None:
    """Return a JSON-safe snapshot of YOLOZU compile evidence attached to ``model``."""
    evidence = getattr(model, "_yolozu_compile_evidence", None)
    if not isinstance(evidence, dict):
        return None
    return copy.deepcopy(evidence)


class _TrackedCompileCallable:
    def __init__(self, compiled: Any, invoke: Any, evidence: dict[str, Any]):
        self._compiled = compiled
        self._invoke = invoke
        self._yolozu_compile_evidence = evidence

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled, name)


def compile_for_inference(
    model: Any,
    *,
    backend: str = "inductor",
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    allow_fallback: bool = False,
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
    allow_fallback : bool
        If ``True``, an unavailable compiler or a setup/first-execution
        compilation failure may run the original eager model. The attached
        evidence records ``actual.status="fallback"``.

    Returns
    -------
    torch.nn.Module
        A compile-tracked model. The original eager model is returned only when
        explicit fallback is enabled and compilation fails during setup.

    Raises
    ------
    TorchCompileError
        If requested compilation cannot be established and fallback is disabled.
    """
    available = torch_compile_available()
    evidence = _new_compile_evidence(
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        allow_fallback=allow_fallback,
        compile_api_available=available,
    )
    if not available:
        exc = RuntimeError("torch.compile is not available")
        _set_compile_failure(
            evidence,
            phase="setup",
            exc=exc,
            fallback=bool(allow_fallback),
        )
        _attach_compile_evidence(model, evidence)
        if allow_fallback:
            logger.warning("torch.compile not available; using explicit eager fallback")
            return model
        raise TorchCompileError(str(exc), evidence=evidence) from exc

    import torch  # type: ignore

    try:
        suppress_errors = bool(torch._dynamo.config.suppress_errors)
    except Exception:
        suppress_errors = False
    if suppress_errors:
        exc = RuntimeError(
            "torch._dynamo.config.suppress_errors=True can silently run eager "
            "code and cannot establish compiled execution evidence"
        )
        _set_compile_failure(
            evidence,
            phase="setup",
            exc=exc,
            fallback=bool(allow_fallback),
        )
        _attach_compile_evidence(model, evidence)
        if allow_fallback:
            logger.warning("%s; using explicit eager fallback", exc)
            return model
        raise TorchCompileError(str(exc), evidence=evidence) from exc

    counter_before = _counter_snapshot(torch)
    try:
        compiled = torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except Exception as exc:
        _set_compile_failure(
            evidence,
            phase="setup",
            exc=exc,
            fallback=bool(allow_fallback),
        )
        _update_counter_evidence(
            evidence,
            before=counter_before,
            after=_counter_snapshot(torch),
        )
        _attach_compile_evidence(model, evidence)
        if allow_fallback:
            logger.warning("torch.compile setup failed (%s); using explicit eager fallback", exc)
            return model
        raise TorchCompileError(
            f"torch.compile setup failed: {exc}",
            evidence=evidence,
        ) from exc

    evidence["evidence"]["setup_completed"] = True
    state: dict[str, Any] = {"fallback": False, "failed_error": None}

    if hasattr(compiled, "forward") and callable(compiled.forward):
        compiled_call = compiled.forward
    else:
        compiled_call = compiled

    def invoke(*args: Any, **kwargs: Any) -> Any:
        if state["fallback"]:
            result = model(*args, **kwargs)
            evidence["evidence"]["fallback_execution_completed"] = True
            return result
        failed_error = state["failed_error"]
        if isinstance(failed_error, TorchCompileError):
            raise failed_error
        if evidence["evidence"]["first_execution_completed"]:
            return compiled_call(*args, **kwargs)

        try:
            result = compiled_call(*args, **kwargs)
        except Exception as exc:
            _set_compile_failure(
                evidence,
                phase="first_execution",
                exc=exc,
                fallback=bool(allow_fallback),
            )
            _update_counter_evidence(
                evidence,
                before=counter_before,
                after=_counter_snapshot(torch),
            )
            if allow_fallback:
                state["fallback"] = True
                logger.warning(
                    "torch.compile first execution failed (%s); using explicit eager fallback",
                    exc,
                )
                fallback_result = model(*args, **kwargs)
                evidence["evidence"]["fallback_execution_completed"] = True
                return fallback_result
            error = TorchCompileError(
                f"torch.compile first execution failed: {exc}",
                evidence=evidence,
            )
            state["failed_error"] = error
            raise error from exc

        evidence["actual"] = {
            "status": "compiled",
            "backend": str(backend),
            "mode": str(mode),
            "fullgraph": bool(fullgraph),
            "dynamic": dynamic,
        }
        evidence["evidence"]["first_execution_completed"] = True
        _update_counter_evidence(
            evidence,
            before=counter_before,
            after=_counter_snapshot(torch),
        )
        evidence["failure"] = None
        return result

    if hasattr(compiled, "forward") and callable(compiled.forward):
        try:
            object.__setattr__(compiled, "forward", invoke)
            tracked = compiled
        except (AttributeError, TypeError):
            tracked = _TrackedCompileCallable(compiled, invoke, evidence)
    else:
        tracked = _TrackedCompileCallable(compiled, invoke, evidence)

    _attach_compile_evidence(tracked, evidence)
    logger.info(
        "torch.compile setup completed (backend=%s, mode=%s); awaiting first execution",
        backend,
        mode,
    )
    return tracked


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
