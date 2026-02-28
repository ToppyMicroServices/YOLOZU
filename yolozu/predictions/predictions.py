"""Core predictions I/O.

Loading, normalising (multiple JSON shapes), validating
schema / structure, and building image-keyed indexes for
detection prediction payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ValidationResult",
    "normalize_predictions_json",
    "normalize_predictions_payload",
    "validate_wrapped_meta",
    "validate_predictions_entries",
    "load_predictions_entries",
    "load_predictions_payload",
    "validate_predictions_payload",
    "load_predictions_index",
]

from yolozu.core.image_keys import add_image_aliases, require_image_key
from yolozu.core.keypoints import normalize_keypoints
from .schema_governance import validate_payload_schema_version


@dataclass(frozen=True)
class ValidationResult:
    warnings: list[str]


def _where(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bbox(bbox: Any, *, strict: bool, where: str) -> list[str]:
    warnings: list[str] = []
    if not isinstance(bbox, dict):
        raise ValueError(f"{where}: bbox must be an object")
    for key in ("cx", "cy", "w", "h"):
        if key not in bbox:
            raise ValueError(f"{where}: bbox missing '{key}'")
        if strict and not _is_number(bbox[key]):
            raise ValueError(f"{where}: bbox.{key} must be a number")
    return warnings


def _validate_detection(det: Any, *, strict: bool, where: str) -> list[str]:
    warnings: list[str] = []
    if not isinstance(det, dict):
        raise ValueError(f"{where}: detection must be an object")

    # Minimal keys needed for most evaluation flows.
    if "score" not in det:
        raise ValueError(f"{where}: detection missing 'score'")
    if strict and not _is_number(det["score"]):
        raise ValueError(f"{where}: detection.score must be a number")

    if "bbox" not in det:
        raise ValueError(f"{where}: detection missing 'bbox'")
    warnings.extend(_validate_bbox(det["bbox"], strict=strict, where=f"{where}.bbox"))

    if "class_id" in det:
        if strict and not isinstance(det["class_id"], int):
            raise ValueError(f"{where}: detection.class_id must be int")
    else:
        warnings.append(f"{where}: detection missing 'class_id' (ok for some flows)")

    # Optional fields (RTDETRPoseAdapter schema)
    if "rot6d" in det:
        rot = det["rot6d"]
        if strict:
            if not isinstance(rot, list) or len(rot) != 6 or not all(_is_number(v) for v in rot):
                raise ValueError(f"{where}: detection.rot6d must be list[6] of numbers")
    if "offsets" in det:
        off = det["offsets"]
        if strict:
            if not isinstance(off, list) or len(off) != 2 or not all(_is_number(v) for v in off):
                raise ValueError(f"{where}: detection.offsets must be list[2] of numbers")
    if "k_delta" in det:
        kd = det["k_delta"]
        if strict:
            if not isinstance(kd, list) or len(kd) != 4 or not all(_is_number(v) for v in kd):
                raise ValueError(f"{where}: detection.k_delta must be list[4] of numbers")

    if "keypoints" in det:
        if strict:
            normalize_keypoints(det["keypoints"], where=f"{where}.keypoints")

    return warnings


def normalize_predictions_json(data: Any) -> list[dict[str, Any]]:
    """Normalize supported prediction JSON shapes into a list of entries.

    Supported:
      1) [{"image": "...", "detections": [...]}, ...]
      2) {"predictions": [ ...same as 1... ], ...}
      3) {"/path.jpg": [...], "0001.jpg": [...]}  (image->detections mapping)
    """

    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for entry in data:
            if isinstance(entry, dict):
                out.append(entry)
        return out

    if isinstance(data, dict):
        out = []
        for image, detections in data.items():
            out.append({"image": str(image), "detections": _as_list(detections)})
        return out

    raise ValueError("Unsupported predictions JSON format")


def normalize_predictions_payload(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Normalize predictions JSON while preserving optional wrapped meta.

    Returns (entries, meta).
    """

    meta: dict[str, Any] | None = None
    if isinstance(data, dict) and "predictions" in data:
        raw_meta = data.get("meta")
        if raw_meta is not None:
            if not isinstance(raw_meta, dict):
                raise ValueError("meta must be an object when present")
            meta = raw_meta
        data = data["predictions"]

    return normalize_predictions_json(data), meta


def _require_type(value: Any, expected: type | tuple[type, ...], *, where: str) -> None:
    if not isinstance(value, expected):
        name = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
        raise ValueError(f"{where} must be {name}")


def _require_bool(value: Any, *, where: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be bool")


def _require_number(value: Any, *, where: str) -> None:
    if not _is_number(value):
        raise ValueError(f"{where} must be a number")


def _require_int_or_none(value: Any, *, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} must be int or null")


def validate_wrapped_meta(meta: dict[str, Any], *, where: str = "meta") -> None:
    """Validate the stable meta contract produced by tools/export_predictions.py --wrap."""

    _require_type(meta, dict, where=where)

    for key in ("timestamp", "adapter", "config", "images", "tta", "ttt"):
        if key not in meta:
            raise ValueError(f"{where}: missing '{key}'")

    _require_type(meta["timestamp"], str, where=_where(where, "timestamp"))
    _require_type(meta["adapter"], str, where=_where(where, "adapter"))
    _require_type(meta["config"], str, where=_where(where, "config"))
    if "checkpoint" in meta and meta["checkpoint"] is not None:
        _require_type(meta["checkpoint"], str, where=_where(where, "checkpoint"))
    if not isinstance(meta["images"], int) or isinstance(meta["images"], bool):
        raise ValueError(f"{where}.images must be int")

    tta = meta["tta"]
    _require_type(tta, dict, where=_where(where, "tta"))
    for key in ("enabled", "seed", "flip_prob", "norm_only", "warnings", "summary"):
        if key not in tta:
            raise ValueError(f"{where}.tta: missing '{key}'")
    _require_bool(tta["enabled"], where=_where(where, "tta.enabled"))
    _require_int_or_none(tta["seed"], where=_where(where, "tta.seed"))
    _require_number(tta["flip_prob"], where=_where(where, "tta.flip_prob"))
    _require_bool(tta["norm_only"], where=_where(where, "tta.norm_only"))
    _require_type(tta["warnings"], list, where=_where(where, "tta.warnings"))
    if tta["summary"] is not None:
        _require_type(tta["summary"], dict, where=_where(where, "tta.summary"))

    ttt = meta["ttt"]
    _require_type(ttt, dict, where=_where(where, "ttt"))
    for key in (
        "enabled",
        "method",
        "steps",
        "batch_size",
        "lr",
        "update_filter",
        "include",
        "exclude",
        "max_batches",
        "seed",
        "mim",
        "report",
    ):
        if key not in ttt:
            raise ValueError(f"{where}.ttt: missing '{key}'")

    _require_bool(ttt["enabled"], where=_where(where, "ttt.enabled"))
    _require_type(ttt["method"], str, where=_where(where, "ttt.method"))
    if not isinstance(ttt["steps"], int) or isinstance(ttt["steps"], bool):
        raise ValueError(f"{where}.ttt.steps must be int")
    if not isinstance(ttt["batch_size"], int) or isinstance(ttt["batch_size"], bool):
        raise ValueError(f"{where}.ttt.batch_size must be int")
    _require_number(ttt["lr"], where=_where(where, "ttt.lr"))
    _require_type(ttt["update_filter"], str, where=_where(where, "ttt.update_filter"))
    if ttt["include"] is not None:
        _require_type(ttt["include"], list, where=_where(where, "ttt.include"))
    if ttt["exclude"] is not None:
        _require_type(ttt["exclude"], list, where=_where(where, "ttt.exclude"))
    if not isinstance(ttt["max_batches"], int) or isinstance(ttt["max_batches"], bool):
        raise ValueError(f"{where}.ttt.max_batches must be int")
    _require_int_or_none(ttt["seed"], where=_where(where, "ttt.seed"))

    mim = ttt["mim"]
    _require_type(mim, dict, where=_where(where, "ttt.mim"))
    for key in ("mask_prob", "patch_size", "mask_value"):
        if key not in mim:
            raise ValueError(f"{where}.ttt.mim: missing '{key}'")
    _require_number(mim["mask_prob"], where=_where(where, "ttt.mim.mask_prob"))
    if not isinstance(mim["patch_size"], int) or isinstance(mim["patch_size"], bool):
        raise ValueError(f"{where}.ttt.mim.patch_size must be int")
    _require_number(mim["mask_value"], where=_where(where, "ttt.mim.mask_value"))

    if ttt["report"] is not None:
        _require_type(ttt["report"], dict, where=_where(where, "ttt.report"))


def validate_predictions_entries(entries: Iterable[dict[str, Any]], *, strict: bool = False) -> ValidationResult:
    warnings: list[str] = []
    for idx, entry in enumerate(entries):
        where = f"predictions[{idx}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: entry must be an object")
        image = entry.get("image")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"{where}: image must be a non-empty string")
        dets = entry.get("detections", [])
        if dets is None:
            dets = []
        if not isinstance(dets, list):
            raise ValueError(f"{where}: 'detections' must be a list")
        for j, det in enumerate(dets):
            warnings.extend(_validate_detection(det, strict=strict, where=f"{where}.detections[{j}]"))
    return ValidationResult(warnings=warnings)


def load_predictions_entries(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text())
    entries, _ = normalize_predictions_payload(data)
    return entries


def load_predictions_payload(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = Path(path)
    data = json.loads(path.read_text())
    return normalize_predictions_payload(data)


def validate_predictions_payload(payload: Any, *, strict: bool = False) -> ValidationResult:
    """Validate any supported predictions JSON payload shape (wrapper/list/mapping)."""

    warnings = validate_payload_schema_version(payload, artifact="predictions")
    entries, meta = normalize_predictions_payload(payload)
    if meta is not None:
        validate_wrapped_meta(meta)
    res = validate_predictions_entries(entries, strict=strict)
    return ValidationResult(warnings=[*warnings, *res.warnings])


def load_predictions_index(path: str | Path, *, add_basename_aliases: bool = True) -> dict[str, list[Any]]:
    """Load predictions into an index mapping image key -> detections list."""

    entries = load_predictions_entries(path)

    index: dict[str, list[Any]] = {}
    for idx, entry in enumerate(entries):
        image = entry.get("image")
        if image is None:
            continue
        try:
            image_key = require_image_key(image, where=f"predictions[{idx}].image")
        except ValueError:
            continue
        dets = entry.get("detections", [])
        if dets is None:
            dets = []
        dets_list = dets if isinstance(dets, list) else _as_list(dets)
        if add_basename_aliases:
            add_image_aliases(index, image_key, dets_list)
        else:
            index[image_key] = dets_list

    return index
