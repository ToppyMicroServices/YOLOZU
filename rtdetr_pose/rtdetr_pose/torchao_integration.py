from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TORCHAO_CONFIG_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_TORCHAO_QUANTIZE_ERRORS = (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class TorchAOReport:
    enabled: bool
    recipe: str
    required: bool
    reason: str | None = None
    error: str | None = None
    applied: bool = False
    api: str | None = None


def apply_torchao_quantization(
    model: Any,
    *,
    recipe: str,
    required: bool = False,
) -> tuple[Any, TorchAOReport]:
    """Best-effort torchao quantization hook.

    This is intentionally defensive because torchao APIs can evolve. If torchao
    isn't installed (or a matching API can't be found), this returns the input
    model unchanged unless required=True.
    """

    recipe = str(recipe or "none").strip().lower()
    if recipe in ("none", "off", "false", "0"):
        return model, TorchAOReport(enabled=False, recipe=recipe, required=bool(required), reason="disabled")

    try:
        import importlib

        q = importlib.import_module("torchao.quantization")
    except ImportError as exc:
        if required:
            raise RuntimeError("torchao is required but not installed (pip install torchao)") from exc
        return model, TorchAOReport(
            enabled=False,
            recipe=recipe,
            required=bool(required),
            reason="torchao_not_installed",
            error=str(exc),
        )

    # Heuristic mapping of recipe -> config factory names.
    config_names: list[str]
    if recipe in ("int8wo", "int8_weight_only", "int8"):
        config_names = ["int8_weight_only", "int8_weight_only_config", "Int8WeightOnlyConfig"]
    elif recipe in ("int4wo", "int4_weight_only", "int4", "qlora"):
        config_names = ["int4_weight_only", "int4_weight_only_config", "Int4WeightOnlyConfig"]
    else:
        if required:
            raise ValueError(f"unsupported torchao recipe: {recipe}")
        return model, TorchAOReport(enabled=False, recipe=recipe, required=bool(required), reason="unsupported_recipe")

    config = None
    for name in config_names:
        obj = getattr(q, name, None)
        if obj is None:
            continue
        try:
            config = obj() if callable(obj) else obj
            break
        except _TORCHAO_CONFIG_ERRORS:
            continue

    if config is None:
        if required:
            raise RuntimeError(f"torchao present but could not resolve config for recipe={recipe}")
        return model, TorchAOReport(
            enabled=False,
            recipe=recipe,
            required=bool(required),
            reason="config_not_found",
        )

    # Try common entrypoints: quantize_ (in-place) then quantize (functional).
    try:
        fn = getattr(q, "quantize_", None)
        if callable(fn):
            fn(model, config)  # type: ignore[misc]
            return model, TorchAOReport(
                enabled=True,
                recipe=recipe,
                required=bool(required),
                applied=True,
                api="torchao.quantization.quantize_",
            )
        fn2 = getattr(q, "quantize", None)
        if callable(fn2):
            out = fn2(model, config)  # type: ignore[misc]
            return out, TorchAOReport(
                enabled=True,
                recipe=recipe,
                required=bool(required),
                applied=True,
                api="torchao.quantization.quantize",
            )
    except _TORCHAO_QUANTIZE_ERRORS as exc:
        if required:
            raise RuntimeError(f"torchao quantization failed (recipe={recipe}): {exc}") from exc
        return model, TorchAOReport(
            enabled=True,
            recipe=recipe,
            required=bool(required),
            applied=False,
            reason="quantize_failed",
            error=str(exc),
        )

    if required:
        raise RuntimeError("torchao present but no supported quantize entrypoint found (expected quantize_/quantize)")
    return model, TorchAOReport(
        enabled=True,
        recipe=recipe,
        required=bool(required),
        applied=False,
        reason="api_not_found",
    )
