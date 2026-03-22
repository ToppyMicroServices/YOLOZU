"""3-D pose inference from 2-D detections.

Reconstructs translation, rotation, and depth from bounding-box
detections using camera intrinsics and geometric constraints.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

__all__ = ["infer_constraints"]

from yolozu.geometry.constraints import apply_constraints
from yolozu.geometry.geometry import corrected_intrinsics, recover_translation


def _image_size(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    if "image_hw" in entry:
        value = entry.get("image_hw")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[1]), float(value[0])
    value = entry.get("image_size")
    if isinstance(value, dict):
        try:
            return float(value.get("width")), float(value.get("height"))
        except (TypeError, ValueError):
            return (None, None)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return (None, None)
    return (None, None)


def _intrinsics_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ("fx", "fy", "cx", "cy")):
            return (float(value["fx"]), float(value["fy"]), float(value["cx"]), float(value["cy"]))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            fx = float(value[0][0])
            fy = float(value[1][1])
            cx = float(value[0][2])
            cy = float(value[1][2])
            return (fx, fy, cx, cy)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _intrinsics(entry: dict[str, Any], det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("intrinsics", "K_gt", "K"):
        value = det.get(key)
        got = _intrinsics_from_value(value)
        if got is not None:
            return got
    for key in ("intrinsics", "K_gt", "K"):
        value = entry.get(key)
        got = _intrinsics_from_value(value)
        if got is not None:
            return got
    return None


def _rot6d_to_matrix(rot6d: Iterable[float]) -> list[list[float]] | None:
    vals = list(rot6d)
    if len(vals) != 6:
        return None
    a1 = vals[0:3]
    a2 = vals[3:6]

    def _norm(v):
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if n <= 0:
            return [0.0, 0.0, 0.0]
        return [v[0] / n, v[1] / n, v[2] / n]

    b1 = _norm(a1)
    dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]
    a2o = [a2[0] - dot * b1[0], a2[1] - dot * b1[1], a2[2] - dot * b1[2]]
    b2 = _norm(a2o)
    b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
    return [b1, b2, b3]


def _bbox_center_wh(
    bbox: dict[str, Any],
    *,
    image_wh: tuple[float | None, float | None],
    bbox_format: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not all(k in bbox for k in ("cx", "cy", "w", "h")):
        return None
    cx = float(bbox["cx"])
    cy = float(bbox["cy"])
    bw = float(bbox["w"])
    bh = float(bbox["h"])
    if bbox_format == "cxcywh_norm":
        width, height = image_wh
        if width is None or height is None:
            return None
        return (cx * width, cy * height), (bw * width, bh * height)
    if bbox_format == "cxcywh_abs":
        return (cx, cy), (bw, bh)
    raise ValueError(f"unsupported bbox_format: {bbox_format}")


def _gate_ok(cfg: dict[str, Any], constraints: dict[str, Any]) -> bool:
    enabled = cfg.get("enabled", {}) if isinstance(cfg, dict) else {}
    if enabled.get("depth_prior", False) and constraints.get("depth_range_violation", 0.0) > 0:
        return False
    if enabled.get("table_plane", False) and not bool(constraints.get("plane_ok", True)):
        return False
    if enabled.get("upright", False) and constraints.get("upright_violation", 0.0) > 0:
        return False
    return True


def infer_constraints(
    entries: Iterable[dict[str, Any]],
    *,
    constraints_cfg: dict[str, Any],
    bbox_format: str = "cxcywh_norm",
    default_size_wh: tuple[float, float] = (1.0, 1.0),
) -> list[dict[str, Any]]:
    out_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image_wh = _image_size(entry)
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []
        new_dets = []
        for det in dets:
            if not isinstance(det, dict):
                continue
            new_det = dict(det)
            bbox = new_det.get("bbox") if isinstance(new_det.get("bbox"), dict) else None
            z_pred = None
            if "log_z" in new_det:
                try:
                    z_pred = float(math.exp(float(new_det["log_z"])))
                except (OverflowError, TypeError, ValueError):
                    z_pred = None
            if z_pred is None and "z" in new_det:
                try:
                    z_pred = float(new_det["z"])
                except (TypeError, ValueError):
                    z_pred = None

            offsets = new_det.get("offsets") or [0.0, 0.0]
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                offsets = [0.0, 0.0]
            offsets = (float(offsets[0]), float(offsets[1]))

            k_delta = new_det.get("k_delta")
            if not isinstance(k_delta, (list, tuple)) or len(k_delta) != 4:
                k_delta = None

            intrinsics = _intrinsics(entry, new_det)
            bbox_info = _bbox_center_wh(bbox, image_wh=image_wh, bbox_format=bbox_format) if bbox else None
            size_wh = new_det.get("size_wh") or entry.get("size_wh") or default_size_wh
            if not isinstance(size_wh, (list, tuple)) or len(size_wh) != 2:
                size_wh = default_size_wh
            size_wh = (float(size_wh[0]), float(size_wh[1]))

            r_mat = new_det.get("R") or new_det.get("r_mat") or new_det.get("rot_mat")
            if r_mat is None and "rot6d" in new_det:
                r_mat = _rot6d_to_matrix(new_det.get("rot6d"))

            if intrinsics and bbox_info and z_pred is not None:
                k_prime = corrected_intrinsics(intrinsics, k_delta) if k_delta is not None else intrinsics
                t_xyz = recover_translation(bbox_info[0], offsets, z_pred, k_prime)
                new_det["k_prime"] = list(k_prime)
                new_det["t_xyz"] = list(t_xyz)
                class_key = new_det.get("class_id", new_det.get("class_name"))
                constraints = apply_constraints(
                    constraints_cfg,
                    class_key=class_key,
                    bbox_wh=bbox_info[1],
                    size_wh=size_wh,
                    intrinsics_fx_fy=(k_prime[0], k_prime[1]),
                    t_xyz=t_xyz,
                    r_mat=r_mat or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    z_pred=z_pred,
                )
                new_det["constraints"] = constraints
                new_det["gate_ok"] = _gate_ok(constraints_cfg, constraints)
            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        out_entries.append(new_entry)
    return out_entries
