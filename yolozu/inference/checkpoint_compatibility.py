"""Fail-closed checkpoint compatibility loading for public RT-DETR paths.

The loader in this module is intentionally framework-light at import time. It
accepts a model with the standard ``state_dict``/``load_state_dict`` methods and
imports PyTorch only when it needs to deserialize a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_REPORT_FORMAT = "yolozu_checkpoint_compatibility_v1"
SUPPORTED_LEGACY_PREFIXES = ("module.", "_orig_mod.")


class CheckpointCompatibilityError(RuntimeError):
    """Raised before model mutation when a checkpoint is not compatible."""

    def __init__(self, message: str, *, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _numel(value: Any) -> int:
    method = getattr(value, "numel", None)
    if callable(method):
        try:
            return int(method())
        except (TypeError, ValueError):
            pass
    shape = _shape(value)
    if shape is None:
        return 0
    total = 1
    for size in shape:
        total *= int(size)
    return int(total)


def _ratio(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _load_with_torch(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("torch is required to load checkpoints") from exc
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "checkpoint must be a raw state dict or an object containing a state_dict mapping"
        )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint state_dict must be a mapping")
        return state_dict, "state_dict_wrapper"
    return checkpoint, "raw_state_dict"


def _normalize_legacy_keys(
    state_dict: Mapping[str, Any],
    *,
    model_keys: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = dict(state_dict)
    applied: list[dict[str, Any]] = []
    while normalized:
        current_overlap = len(set(normalized) & model_keys)
        selected: tuple[str, dict[str, Any], int] | None = None
        for prefix in SUPPORTED_LEGACY_PREFIXES:
            if not all(str(key).startswith(prefix) for key in normalized):
                continue
            candidate = {str(key)[len(prefix) :]: value for key, value in normalized.items()}
            candidate_overlap = len(set(candidate) & model_keys)
            if candidate_overlap <= current_overlap:
                continue
            if selected is None or candidate_overlap > selected[2]:
                selected = (prefix, candidate, candidate_overlap)
        if selected is None:
            break
        prefix, normalized, candidate_overlap = selected
        applied.append(
            {
                "rule": "strip_uniform_prefix",
                "prefix": prefix,
                "key_count": len(normalized),
                "model_key_overlap_after": candidate_overlap,
            }
        )
    return normalized, applied


def _config_record(config_identity: str | Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "identity": None if config_identity is None else str(config_identity),
        "path": None,
        "sha256": None,
    }
    if config_identity is None:
        return record
    value = str(config_identity)
    if value.startswith(("builtin:", "pkg:")):
        name = value.split(":", 1)[1].strip()
        if name:
            if not name.endswith(".json"):
                name = f"{name}.json"
            relative = name if "/" in name else f"configs/{name}"
            try:
                import importlib.resources

                content = (
                    importlib.resources.files("rtdetr_pose")
                    .joinpath(relative)
                    .read_bytes()
                )
            except Exception:
                return record
            record["path"] = f"package:rtdetr_pose/{relative}"
            record["sha256"] = hashlib.sha256(content).hexdigest()
        return record
    path = Path(value).expanduser()
    if path.is_file():
        resolved = path.resolve()
        record["path"] = str(resolved)
        record["sha256"] = _sha256_file(resolved)
    return record


def _model_signature(model_state: Mapping[str, Any]) -> str:
    entries = [
        {
            "key": str(key),
            "shape": list(_shape(value) or ()),
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
        for key, value in sorted(model_state.items())
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint_compatible(
    model: Any,
    checkpoint_path: str | Path,
    *,
    config_identity: str | Path | None = None,
    allow_partial: bool = False,
    checkpoint_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint only after producing a complete compatibility report.

    By default, every model-state key must exist with an identical shape and no
    extra checkpoint keys may remain after the documented legacy-prefix
    normalization. ``allow_partial=True`` is intended for transfer learning and
    loads only name-and-shape matches while reporting ``status=partial``.
    """

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    model_state_raw = model.state_dict()
    if not isinstance(model_state_raw, Mapping):
        raise TypeError("model.state_dict() must return a mapping")
    model_state = {str(key): value for key, value in model_state_raw.items()}
    model_keys = set(model_state)
    model_parameters = {
        str(key): value
        for key, value in (
            model.named_parameters() if hasattr(model, "named_parameters") else ()
        )
    }

    checkpoint_sha256 = _sha256_file(path)
    deserialization = {
        "policy": (
            "custom_loader"
            if checkpoint_loader is not None
            else "torch_weights_only"
        ),
        "weights_only": checkpoint_loader is None,
    }
    loader = checkpoint_loader or _load_with_torch
    try:
        loaded = loader(path)
    except Exception as exc:
        report = {
            "format": CHECKPOINT_REPORT_FORMAT,
            "status": "incompatible",
            "allow_partial": bool(allow_partial),
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": checkpoint_sha256,
                "container": "unavailable",
                "deserialization": deserialization,
            },
            "model": {
                "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                "config": _config_record(config_identity),
                "state_signature_sha256": _model_signature(model_state),
            },
            "error": {
                "stage": "deserialization",
                "type": f"{exc.__class__.__module__}.{exc.__class__.__qualname__}",
                "message": str(exc),
            },
        }
        raise CheckpointCompatibilityError(
            "checkpoint could not be safely deserialized as a tensor/state-dict archive",
            report=report,
        ) from exc
    try:
        extracted, container = _extract_state_dict(loaded)
    except TypeError as exc:
        report = {
            "format": CHECKPOINT_REPORT_FORMAT,
            "status": "incompatible",
            "allow_partial": bool(allow_partial),
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": checkpoint_sha256,
                "container": "unsupported",
                "deserialization": deserialization,
            },
            "model": {
                "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                "config": _config_record(config_identity),
                "state_signature_sha256": _model_signature(model_state),
            },
            "error": str(exc),
        }
        raise CheckpointCompatibilityError(str(exc), report=report) from exc

    non_string_keys = [repr(key) for key in extracted if not isinstance(key, str)]
    if non_string_keys:
        report = {
            "format": CHECKPOINT_REPORT_FORMAT,
            "status": "incompatible",
            "allow_partial": bool(allow_partial),
            "checkpoint": {
                "path": str(path.resolve()),
                "sha256": checkpoint_sha256,
                "container": container,
                "deserialization": deserialization,
            },
            "model": {
                "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                "config": _config_record(config_identity),
                "state_signature_sha256": _model_signature(model_state),
            },
            "error": "checkpoint state_dict keys must be strings",
            "non_string_keys": non_string_keys,
        }
        raise CheckpointCompatibilityError(
            "checkpoint state_dict keys must be strings",
            report=report,
        )

    normalized, applied_normalizations = _normalize_legacy_keys(
        extracted,
        model_keys=model_keys,
    )
    checkpoint_tensors = {
        key: value for key, value in normalized.items() if _shape(value) is not None
    }
    non_tensor_keys = sorted(
        str(key) for key, value in normalized.items() if _shape(value) is None
    )

    matched: dict[str, Any] = {}
    shape_mismatches: list[dict[str, Any]] = []
    for key in sorted(model_keys & set(checkpoint_tensors)):
        checkpoint_value = checkpoint_tensors[key]
        model_value = model_state[key]
        checkpoint_shape = _shape(checkpoint_value)
        model_shape = _shape(model_value)
        if checkpoint_shape == model_shape:
            matched[key] = checkpoint_value
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_shape or ()),
                    "model_shape": list(model_shape or ()),
                }
            )

    matched_keys = sorted(matched)
    missing_keys = sorted(model_keys - set(checkpoint_tensors))
    unexpected_keys = sorted(set(checkpoint_tensors) - model_keys)
    shape_mismatch_keys = {item["key"] for item in shape_mismatches}
    unloaded_model_keys = sorted(set(missing_keys) | shape_mismatch_keys)
    skipped_checkpoint_keys = sorted(set(unexpected_keys) | shape_mismatch_keys)

    checkpoint_tensor_numel = sum(_numel(value) for value in checkpoint_tensors.values())
    model_state_numel = sum(_numel(value) for value in model_state.values())
    matched_state_numel = sum(_numel(model_state[key]) for key in matched_keys)
    model_parameter_numel = sum(_numel(value) for value in model_parameters.values())
    matched_parameter_keys = sorted(set(matched_keys) & set(model_parameters))
    matched_parameter_numel = sum(
        _numel(model_parameters[key]) for key in matched_parameter_keys
    )

    full = bool(
        model_keys
        and not missing_keys
        and not unexpected_keys
        and not shape_mismatches
        and not non_tensor_keys
        and len(matched_keys) == len(model_keys)
    )
    partial_allowed = bool(allow_partial and matched_keys)
    status = "full" if full else ("partial" if partial_allowed else "incompatible")

    report: dict[str, Any] = {
        "format": CHECKPOINT_REPORT_FORMAT,
        "status": status,
        "allow_partial": bool(allow_partial),
        "checkpoint": {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256,
            "container": container,
            "deserialization": deserialization,
            "tensor_count": len(checkpoint_tensors),
            "tensor_numel": checkpoint_tensor_numel,
        },
        "model": {
            "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
            "config": _config_record(config_identity),
            "state_signature_sha256": _model_signature(model_state),
            "state_tensor_count": len(model_state),
            "state_tensor_numel": model_state_numel,
            "parameter_tensor_count": len(model_parameters),
            "parameter_numel": model_parameter_numel,
        },
        "legacy_key_normalization": {
            "supported_uniform_prefixes": list(SUPPORTED_LEGACY_PREFIXES),
            "applied": applied_normalizations,
        },
        "compatibility": {
            "matched_keys": matched_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "shape_mismatches": shape_mismatches,
            "shape_mismatch_keys": sorted(shape_mismatch_keys),
            "unloaded_model_keys": unloaded_model_keys,
            "skipped_checkpoint_keys": skipped_checkpoint_keys,
            "non_tensor_keys": non_tensor_keys,
            "tensor_count_coverage": {
                "matched": len(matched_keys),
                "model_total": len(model_state),
                "checkpoint_total": len(checkpoint_tensors),
                "model_ratio": _ratio(len(matched_keys), len(model_state)),
                "checkpoint_ratio": _ratio(
                    len(matched_keys),
                    len(checkpoint_tensors),
                ),
            },
            "state_numel_coverage": {
                "matched": matched_state_numel,
                "model_total": model_state_numel,
                "checkpoint_total": checkpoint_tensor_numel,
                "model_ratio": _ratio(matched_state_numel, model_state_numel),
                "checkpoint_ratio": _ratio(
                    matched_state_numel,
                    checkpoint_tensor_numel,
                ),
            },
            "parameter_numel_coverage": {
                "matched": matched_parameter_numel,
                "model_total": model_parameter_numel,
                "model_ratio": _ratio(
                    matched_parameter_numel,
                    model_parameter_numel,
                ),
                "matched_keys": matched_parameter_keys,
            },
        },
        "load": {
            "mode": "strict_full" if full else "name_and_shape_partial",
            "loaded": False,
            "loaded_key_count": 0,
        },
    }

    if not full and not partial_allowed:
        message = (
            "checkpoint is incompatible with the selected model/config: "
            f"matched={len(matched_keys)}/{len(model_state)}, "
            f"unloaded={len(unloaded_model_keys)}, "
            f"skipped_checkpoint={len(skipped_checkpoint_keys)}, "
            f"missing_name={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
            f"shape_mismatch={len(shape_mismatches)}"
        )
        if allow_partial and not matched_keys:
            message += "; partial loading requires at least one name-and-shape match"
        else:
            message += "; use explicit partial opt-in only for transfer/diagnostic work"
        raise CheckpointCompatibilityError(message, report=report)

    if full:
        state_to_load = extracted if not applied_normalizations else normalized
        model.load_state_dict(state_to_load, strict=True)
        loaded_key_count = len(state_to_load)
    else:
        model.load_state_dict(matched, strict=False)
        loaded_key_count = len(matched)
    report["load"]["loaded"] = True
    report["load"]["loaded_key_count"] = loaded_key_count
    return report
