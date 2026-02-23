from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired  # type: ignore


SYNTHGEN_REQUIRED_FIELDS: tuple[str, ...] = (
    "image",
    "depth_ndc",
    "inst_id",
    "sem_id",
    "kpts2d",
    "prompt",
    "scene_spec",
    "schema_id",
    "schema_version",
    "asset_ids",
    "inst_map",
)

SYNTHGEN_SCHEMA_IDS: tuple[str, ...] = (
    "animal_v1",
    "mechanical_v1",
)


class SynthGenSample(TypedDict):
    image: Any
    depth_ndc: Any
    inst_id: Any
    sem_id: Any
    kpts2d: Any
    prompt: str
    scene_spec: str | dict[str, Any]
    schema_id: str
    schema_version: str
    asset_ids: list[str]
    inst_map: str | dict[str, Any]
    kpts3d_object: NotRequired[Any]
    pose_obj2cam: NotRequired[Any]


@dataclass(frozen=True)
class SynthGenValidationResult:
    ok: bool
    errors: list[str]


def _as_ndarray(value: Any, *, field: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        try:
            return np.asarray(value)
        except Exception as exc:
            raise ValueError(f"{field}: unable to convert list to ndarray: {exc}") from exc
    raise ValueError(f"{field}: expected numpy array or list")


def _json_like_ok(value: Any, *, field: str) -> bool:
    if isinstance(value, dict):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        decoded = json.loads(value)
    except Exception:
        return False
    return isinstance(decoded, dict)


def _validate_required_keys(sample: dict[str, Any], errors: list[str]) -> None:
    for key in SYNTHGEN_REQUIRED_FIELDS:
        if key not in sample:
            errors.append(f"missing required field: {key}")


def _validate_core_tensors(sample: dict[str, Any], errors: list[str]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    image = depth = inst = sem = None
    if "image" in sample:
        try:
            image = _as_ndarray(sample["image"], field="image")
            if image.ndim != 3 or int(image.shape[-1]) != 3:
                errors.append(f"image: expected [H,W,3], got shape={tuple(image.shape)}")
            if image.dtype != np.uint8:
                errors.append(f"image: expected uint8, got {image.dtype}")
        except Exception as exc:
            errors.append(str(exc))

    if "depth_ndc" in sample:
        try:
            depth = _as_ndarray(sample["depth_ndc"], field="depth_ndc")
            if depth.ndim != 2:
                errors.append(f"depth_ndc: expected [H,W], got shape={tuple(depth.shape)}")
            if depth.dtype != np.float32:
                errors.append(f"depth_ndc: expected float32, got {depth.dtype}")
            if depth.size > 0:
                dmin = float(depth.min())
                dmax = float(depth.max())
                if dmin < 0.0 or dmax > 1.0:
                    errors.append(f"depth_ndc: expected range [0,1], got min={dmin} max={dmax}")
        except Exception as exc:
            errors.append(str(exc))

    if "inst_id" in sample:
        try:
            inst = _as_ndarray(sample["inst_id"], field="inst_id")
            if inst.ndim != 2:
                errors.append(f"inst_id: expected [H,W], got shape={tuple(inst.shape)}")
            if inst.dtype != np.uint32:
                errors.append(f"inst_id: expected uint32, got {inst.dtype}")
        except Exception as exc:
            errors.append(str(exc))

    if "sem_id" in sample:
        try:
            sem = _as_ndarray(sample["sem_id"], field="sem_id")
            if sem.ndim != 2:
                errors.append(f"sem_id: expected [H,W], got shape={tuple(sem.shape)}")
            if sem.dtype != np.uint16:
                errors.append(f"sem_id: expected uint16, got {sem.dtype}")
        except Exception as exc:
            errors.append(str(exc))

    return image, depth, inst, sem


def _validate_shape_consistency(
    *,
    image: np.ndarray | None,
    depth: np.ndarray | None,
    inst: np.ndarray | None,
    sem: np.ndarray | None,
    errors: list[str],
) -> None:
    if image is None or depth is None or inst is None or sem is None:
        return
    h, w = int(image.shape[0]), int(image.shape[1])
    if tuple(depth.shape) != (h, w):
        errors.append(f"depth_ndc: shape mismatch with image ({h},{w}) vs {tuple(depth.shape)}")
    if tuple(inst.shape) != (h, w):
        errors.append(f"inst_id: shape mismatch with image ({h},{w}) vs {tuple(inst.shape)}")
    if tuple(sem.shape) != (h, w):
        errors.append(f"sem_id: shape mismatch with image ({h},{w}) vs {tuple(sem.shape)}")


def _validate_keypoints(sample: dict[str, Any], errors: list[str]) -> None:
    if "kpts2d" not in sample:
        return
    try:
        kpts = _as_ndarray(sample["kpts2d"], field="kpts2d")
    except Exception as exc:
        errors.append(str(exc))
        return
    if kpts.ndim != 3 or int(kpts.shape[-1]) != 3:
        errors.append(f"kpts2d: expected [N_inst,K,3], got shape={tuple(kpts.shape)}")
        return
    if kpts.dtype != np.float32:
        errors.append(f"kpts2d: expected float32, got {kpts.dtype}")
    if kpts.size > 0:
        vis = np.rint(kpts[..., 2]).astype(np.int64)
        bad = np.logical_not(np.isin(vis, (0, 1, 2)))
        if bool(np.any(bad)):
            errors.append("kpts2d: visibility channel must be in {0,1,2}")


def _validate_optional_fields(sample: dict[str, Any], errors: list[str]) -> None:
    if "kpts3d_object" in sample:
        try:
            arr = _as_ndarray(sample["kpts3d_object"], field="kpts3d_object")
            if arr.ndim != 3 or int(arr.shape[-1]) != 3:
                errors.append(f"kpts3d_object: expected [N_inst,K,3], got shape={tuple(arr.shape)}")
        except Exception as exc:
            errors.append(str(exc))
    if "pose_obj2cam" in sample:
        try:
            arr = _as_ndarray(sample["pose_obj2cam"], field="pose_obj2cam")
            if arr.ndim not in (2, 3):
                errors.append(f"pose_obj2cam: expected [4,4] or [N_inst,4,4], got shape={tuple(arr.shape)}")
            elif arr.ndim == 2 and tuple(arr.shape) != (4, 4):
                errors.append(f"pose_obj2cam: expected [4,4], got shape={tuple(arr.shape)}")
            elif arr.ndim == 3 and tuple(arr.shape[-2:]) != (4, 4):
                errors.append(f"pose_obj2cam: expected [N_inst,4,4], got shape={tuple(arr.shape)}")
        except Exception as exc:
            errors.append(str(exc))


def validate_synthgen_sample(sample: dict[str, Any]) -> SynthGenValidationResult:
    errors: list[str] = []
    if not isinstance(sample, dict):
        return SynthGenValidationResult(ok=False, errors=["sample must be a dict"])

    _validate_required_keys(sample, errors)
    image, depth, inst, sem = _validate_core_tensors(sample, errors)
    _validate_shape_consistency(image=image, depth=depth, inst=inst, sem=sem, errors=errors)
    _validate_keypoints(sample, errors)
    _validate_optional_fields(sample, errors)

    prompt = sample.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt: expected non-empty string")

    schema_id = sample.get("schema_id")
    if not isinstance(schema_id, str) or not schema_id.strip():
        errors.append("schema_id: expected non-empty string")

    schema_version = sample.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        errors.append("schema_version: expected non-empty string")

    asset_ids = sample.get("asset_ids")
    if not isinstance(asset_ids, list) or any(not isinstance(x, str) or not x for x in asset_ids):
        errors.append("asset_ids: expected list[str]")

    if not _json_like_ok(sample.get("scene_spec"), field="scene_spec"):
        errors.append("scene_spec: expected JSON object string or dict")
    if not _json_like_ok(sample.get("inst_map"), field="inst_map"):
        errors.append("inst_map: expected JSON object string or dict")

    return SynthGenValidationResult(ok=not errors, errors=errors)


def _decode_json_object(value: str | dict[str, Any], *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(value)
    except Exception as exc:
        raise ValueError(f"{field}: invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field}: expected JSON object")
    return decoded


def normalize_synthgen_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Coerce SynthGen sample fields into canonical runtime dtypes."""

    if not isinstance(sample, dict):
        raise ValueError("sample must be a dict")

    out: dict[str, Any] = dict(sample)
    out["image"] = _as_ndarray(out["image"], field="image").astype(np.uint8, copy=False)
    out["depth_ndc"] = _as_ndarray(out["depth_ndc"], field="depth_ndc").astype(np.float32, copy=False)
    out["inst_id"] = _as_ndarray(out["inst_id"], field="inst_id").astype(np.uint32, copy=False)
    out["sem_id"] = _as_ndarray(out["sem_id"], field="sem_id").astype(np.uint16, copy=False)
    out["kpts2d"] = _as_ndarray(out["kpts2d"], field="kpts2d").astype(np.float32, copy=False)
    out["scene_spec"] = _decode_json_object(out["scene_spec"], field="scene_spec")
    out["inst_map"] = _decode_json_object(out["inst_map"], field="inst_map")
    out["asset_ids"] = [str(x) for x in (out.get("asset_ids") or [])]

    if "kpts3d_object" in out:
        out["kpts3d_object"] = _as_ndarray(out["kpts3d_object"], field="kpts3d_object").astype(np.float32, copy=False)
    if "pose_obj2cam" in out:
        out["pose_obj2cam"] = _as_ndarray(out["pose_obj2cam"], field="pose_obj2cam").astype(np.float32, copy=False)

    result = validate_synthgen_sample(out)
    if not result.ok:
        msg = "; ".join(result.errors)
        raise ValueError(f"invalid SynthGen sample: {msg}")
    return out
