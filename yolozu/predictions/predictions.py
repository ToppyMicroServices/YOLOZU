"""Core predictions I/O.

Loading, normalising (multiple JSON shapes), validating
schema / structure, and building image-keyed indexes for
detection prediction payloads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ValidationResult",
    "CanonicalizationResult",
    "canonicalize_predictions",
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


@dataclass(frozen=True)
class CanonicalizationResult:
    entries: list[dict[str, Any]]
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
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return True


def _validate_bbox(bbox: Any, *, strict: bool, where: str) -> list[str]:
    warnings: list[str] = []
    if not isinstance(bbox, dict):
        raise ValueError(f"{where}: bbox must be an object")
    for key in ("cx", "cy", "w", "h"):
        if key not in bbox:
            raise ValueError(f"{where}: bbox missing '{key}'")
        if strict and not _is_number(bbox[key]):
            raise ValueError(f"{where}: bbox.{key} must be a number")
    if strict:
        cx = float(bbox["cx"])
        cy = float(bbox["cy"])
        w = float(bbox["w"])
        h = float(bbox["h"])
        if not (0.0 <= cx <= 1.0):
            raise ValueError(f"{where}: bbox.cx must be in [0,1]")
        if not (0.0 <= cy <= 1.0):
            raise ValueError(f"{where}: bbox.cy must be in [0,1]")
        if not (0.0 < w <= 1.0):
            raise ValueError(f"{where}: bbox.w must be in (0,1]")
        if not (0.0 < h <= 1.0):
            raise ValueError(f"{where}: bbox.h must be in (0,1]")
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
        if strict:
            raise ValueError(f"{where}: detection missing 'class_id'")
        warnings.append(f"{where}: detection missing 'class_id' (ok for non-strict flows)")

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


_ENTRY_ALLOWED_KEYS = frozenset(
    {
        "image",
        "detections",
        "image_size",
        "preprocess",
        "intrinsics",
        "task",
        "meta",
    }
)
_DETECTION_ALLOWED_KEYS = frozenset(
    {
        "class_id",
        "score",
        "bbox",
        "rot6d",
        "offsets",
        "k_delta",
        "keypoints",
        "log_z",
        "log_sigma_z",
        "log_sigma_rot",
        "sigma_z",
        "sigma_rot",
        "category_id",
        "bbox_abs",
        "mask",
        "meta",
    }
)


def _validate_unknown_keys(
    keys: set[str],
    *,
    allowed: frozenset[str],
    where: str,
    policy: str,
    warnings: list[str],
) -> None:
    unknown = sorted(key for key in keys if key not in allowed)
    if not unknown:
        return
    if policy == "error":
        raise ValueError(f"{where}: unknown keys {unknown}")
    if policy == "warn":
        warnings.append(f"{where}: unknown keys {unknown}")


def _stable_detection_sort_key(det: dict[str, Any]) -> tuple[Any, ...]:
    bbox = det.get("bbox") or {}
    return (
        -float(det.get("score", 0.0)),
        int(det.get("class_id", -1)),
        float(bbox.get("cx", 0.0)),
        float(bbox.get("cy", 0.0)),
        float(bbox.get("w", 0.0)),
        float(bbox.get("h", 0.0)),
    )


def _range_fix(
    value: float,
    *,
    minimum: float,
    maximum: float,
    clamp: bool,
    where: str,
    warnings: list[str],
) -> float:
    out = float(value)
    if minimum <= out <= maximum:
        return out
    if not clamp:
        raise ValueError(f"{where}: must be in [{minimum},{maximum}]")
    clamped = min(max(out, minimum), maximum)
    warnings.append(f"{where}: out of range ({out}); clamped to {clamped}")
    return clamped


def canonicalize_predictions(
    entries: Iterable[dict[str, Any]],
    *,
    policy: str = "clamp",
    strict: bool = False,
    unknown_keys: str | None = None,
) -> CanonicalizationResult:
    """Canonicalize prediction entries for stable evaluation and regression.

    Rules:
    - image key is normalized via ``require_image_key``.
    - detections list is normalized to ``[]`` when null.
    - score/bbox values must be finite numbers.
    - score and bbox range checks use either clamp policy or strict errors.
    - detections are stable-sorted by ``(-score, class_id, bbox)``.
    - duplicate image entries are rejected.
    """

    mode = str(policy).strip().lower()
    if mode not in ("clamp", "error"):
        raise ValueError("policy must be one of: clamp, error")

    unknown_mode = str(unknown_keys or ("warn" if strict else "allow")).strip().lower()
    if unknown_mode not in ("allow", "warn", "error"):
        raise ValueError("unknown_keys must be one of: allow, warn, error")

    warnings: list[str] = []
    out: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    clamp = bool(mode == "clamp" and not strict)

    for idx, entry in enumerate(entries):
        where = f"predictions[{idx}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: entry must be an object")

        _validate_unknown_keys(
            set(entry.keys()),
            allowed=_ENTRY_ALLOWED_KEYS,
            where=where,
            policy=unknown_mode,
            warnings=warnings,
        )

        image = require_image_key(entry.get("image"), where=f"{where}.image")
        if image in seen_images:
            raise ValueError(f"{where}.image: duplicate image entry '{image}'")
        seen_images.add(image)

        dets = entry.get("detections")
        if dets is None:
            dets = []
        if not isinstance(dets, list):
            raise ValueError(f"{where}.detections: must be a list")

        dets_out: list[dict[str, Any]] = []
        for det_idx, det in enumerate(dets):
            det_where = f"{where}.detections[{det_idx}]"
            if not isinstance(det, dict):
                raise ValueError(f"{det_where}: detection must be an object")

            _validate_unknown_keys(
                set(det.keys()),
                allowed=_DETECTION_ALLOWED_KEYS,
                where=det_where,
                policy=unknown_mode,
                warnings=warnings,
            )

            if "class_id" not in det:
                if strict:
                    raise ValueError(f"{det_where}: detection missing 'class_id'")
                warnings.append(f"{det_where}: detection missing 'class_id' (ok for non-strict flows)")

            score = det.get("score")
            if not _is_number(score):
                raise ValueError(f"{det_where}.score: must be finite number")
            score_f = _range_fix(
                float(score),
                minimum=0.0,
                maximum=1.0,
                clamp=clamp,
                where=f"{det_where}.score",
                warnings=warnings,
            )

            bbox = det.get("bbox")
            if not isinstance(bbox, dict):
                raise ValueError(f"{det_where}.bbox: must be an object")
            for key in ("cx", "cy", "w", "h"):
                if key not in bbox:
                    raise ValueError(f"{det_where}.bbox: missing '{key}'")
                if not _is_number(bbox[key]):
                    raise ValueError(f"{det_where}.bbox.{key}: must be finite number")

            cx = _range_fix(
                float(bbox["cx"]),
                minimum=0.0,
                maximum=1.0,
                clamp=clamp,
                where=f"{det_where}.bbox.cx",
                warnings=warnings,
            )
            cy = _range_fix(
                float(bbox["cy"]),
                minimum=0.0,
                maximum=1.0,
                clamp=clamp,
                where=f"{det_where}.bbox.cy",
                warnings=warnings,
            )
            w = _range_fix(
                float(bbox["w"]),
                minimum=0.0,
                maximum=1.0,
                clamp=clamp,
                where=f"{det_where}.bbox.w",
                warnings=warnings,
            )
            h = _range_fix(
                float(bbox["h"]),
                minimum=0.0,
                maximum=1.0,
                clamp=clamp,
                where=f"{det_where}.bbox.h",
                warnings=warnings,
            )
            if w <= 0.0:
                raise ValueError(f"{det_where}.bbox.w: must be > 0")
            if h <= 0.0:
                raise ValueError(f"{det_where}.bbox.h: must be > 0")

            det_out = dict(det)
            det_out["score"] = float(score_f)
            det_out["bbox"] = {
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
            if "class_id" in det_out:
                if strict and not isinstance(det_out["class_id"], int):
                    raise ValueError(f"{det_where}.class_id: must be int")
            dets_out.append(det_out)

        dets_out.sort(key=_stable_detection_sort_key)
        entry_out = dict(entry)
        entry_out["image"] = image
        entry_out["detections"] = dets_out
        out.append(entry_out)

    return CanonicalizationResult(entries=out, warnings=warnings)


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
    canonical = canonicalize_predictions(entries, strict=bool(strict), policy="clamp")
    warnings: list[str] = []
    warnings.extend(canonical.warnings)
    for idx, entry in enumerate(canonical.entries):
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
    return canonicalize_predictions(entries, strict=False, policy="clamp").entries


def load_predictions_payload(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = Path(path)
    data = json.loads(path.read_text())
    entries, meta = normalize_predictions_payload(data)
    canonical = canonicalize_predictions(entries, strict=False, policy="clamp")
    return canonical.entries, meta


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
