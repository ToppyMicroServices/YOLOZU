from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from yolozu.boxes import iou_xyxy_abs
from yolozu.image_keys import add_image_aliases, lookup_image_alias
from yolozu.intrinsics import parse_intrinsics


if TYPE_CHECKING:  # pragma: no cover
    import torch


_FLOAT_PARSE_ERRORS = (TypeError, ValueError, OverflowError)
_OPTIONAL_TORCH_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except _FLOAT_PARSE_ERRORS:
            return None
    return None


def _parse_intrinsics(value: Any) -> tuple[float, float, float, float] | None:
    intr = parse_intrinsics(value)
    if intr is None:
        return None
    return (float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]))


def _get_intrinsics(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("intrinsics", "K_gt", "K"):
        got = _parse_intrinsics(record.get(key))
        if got is not None:
            return got
    return None


def _get_image_hw(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("image_hw") or record.get("hw")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except _FLOAT_PARSE_ERRORS:
            return None
    value = record.get("image_size")
    if isinstance(value, dict):
        try:
            return (float(value.get("height")), float(value.get("width")))
        except _FLOAT_PARSE_ERRORS:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            # image_size is (w,h)
            return (float(value[1]), float(value[0]))
        except _FLOAT_PARSE_ERRORS:
            return None
    return None


def _expand_to_instances(value: Any, n: int) -> list[Any | None]:
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        # Per-instance list.
        if len(value) == n and (n == 0 or not isinstance(value[0], (int, float, str))):
            return list(value)
        # Broadcast a single vector/matrix-ish thing.
        return [value] * n
    return [value] * n


def _extract_t_gt(record: dict[str, Any], n: int) -> list[list[float] | None]:
    pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
    t_raw = record.get("t_gt")
    if t_raw is None and isinstance(pose, dict):
        t_raw = pose.get("t")
    t_list = _expand_to_instances(t_raw, n)
    out: list[list[float] | None] = []
    for item in t_list:
        vals = _as_float_list(item)
        if vals is None or len(vals) != 3:
            out.append(None)
        else:
            out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return out


def _bbox_xyxy_from_cxcywh_norm(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1 = float(cx) - float(w) / 2.0
    y1 = float(cy) - float(h) / 2.0
    x2 = float(cx) + float(w) / 2.0
    y2 = float(cy) + float(h) / 2.0
    return (x1, y1, x2, y2)


def _extract_det_bbox(det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = det.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        return _bbox_xyxy_from_cxcywh_norm(float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"]))
    except (KeyError, *_FLOAT_PARSE_ERRORS):
        return None


def _extract_gt_bbox(gt: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        return _bbox_xyxy_from_cxcywh_norm(float(gt["cx"]), float(gt["cy"]), float(gt["w"]), float(gt["h"]))
    except (KeyError, *_FLOAT_PARSE_ERRORS):
        return None


def _match_dets_to_gts(
    dets: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    if not dets or not gts:
        return []

    pairs: list[tuple[float, int, int]] = []
    for det_idx, det in enumerate(dets):
        det_bbox = _extract_det_bbox(det)
        if det_bbox is None:
            continue
        det_cls = det.get("class_id")
        for gt_idx, gt in enumerate(gts):
            if det_cls is not None and det_cls != gt.get("class_id"):
                continue
            gt_bbox = _extract_gt_bbox(gt)
            if gt_bbox is None:
                continue
            iou = iou_xyxy_abs(det_bbox, gt_bbox)
            if iou >= float(iou_threshold):
                pairs.append((float(iou), int(det_idx), int(gt_idx)))

    pairs.sort(reverse=True, key=lambda x: x[0])
    used_det: set[int] = set()
    used_gt: set[int] = set()
    out: list[tuple[int, int]] = []
    for _, det_idx, gt_idx in pairs:
        if det_idx in used_det or gt_idx in used_gt:
            continue
        used_det.add(det_idx)
        used_gt.add(gt_idx)
        out.append((det_idx, gt_idx))
    return out


def _compose_k_delta(per_det: list[float] | None, shared: torch.Tensor) -> torch.Tensor:
    # Compose multiplicative dfx/dfy, additive dcx/dcy.
    # per_det may be None.
    import torch

    if per_det is None or len(per_det) != 4:
        return shared

    base = torch.tensor([float(v) for v in per_det], dtype=shared.dtype, device=shared.device)
    dfx = (1.0 + base[0]) * (1.0 + shared[0]) - 1.0
    dfy = (1.0 + base[1]) * (1.0 + shared[1]) - 1.0
    dcx = base[2] + shared[2]
    dcy = base[3] + shared[3]
    return torch.stack([dfx, dfy, dcx, dcy], dim=0)


@dataclass(frozen=True)
class CalibConfig:
    iou_threshold: float = 0.5
    optimize_k_delta: bool = False
    image_hw_override: tuple[float, float] | None = None
    lbfgs_max_iter: int = 30
    lbfgs_lr: float = 1.0
    w_z: float = 1.0
    w_t: float = 0.2
    reg_k: float = 1e-2
    reg_log_s: float = 1e-4


@dataclass(frozen=True)
class CalibResult:
    scale_s: float
    shared_k_delta: list[float] | None
    matches: int
    used_for_t: int
    loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_s": float(self.scale_s),
            "shared_k_delta": self.shared_k_delta,
            "matches": int(self.matches),
            "used_for_t": int(self.used_for_t),
            "loss": float(self.loss),
        }


def calibrate_predictions_lbfgs(
    records: Iterable[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    config: CalibConfig,
) -> tuple[list[dict[str, Any]], CalibResult]:
    """Calibrate depth scale and optional shared k_delta using L-BFGS.

    Requires torch.

    Assumptions:
    - detections contain `log_z` (preferred) or `z`.
    - records contain YOLO labels with keys {class_id,cx,cy,w,h}.
    - If optimize_k_delta is enabled, records should provide intrinsics and either
      image_hw/image_size or config.image_hw_override.
    - If translation supervision is available, records should include t_gt (per-image
      or per-instance list); we use matched GT index.
    """

    try:
        import torch
        import torch.nn.functional as F
    except _OPTIONAL_TORCH_ERRORS as exc:  # pragma: no cover
        raise RuntimeError("torch is required for L-BFGS calibration") from exc

    pred_index: dict[str, dict[str, Any]] = {}
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not image:
            continue
        add_image_aliases(pred_index, str(image), entry)

    z_pred_list: list[float] = []
    z_gt_list: list[float] = []

    # Optional translation supervision.
    u_list: list[float] = []
    v_list: list[float] = []
    du_list: list[float] = []
    dv_list: list[float] = []
    fx_list: list[float] = []
    fy_list: list[float] = []
    cx_list: list[float] = []
    cy_list: list[float] = []
    t_gt_list: list[list[float]] = []
    det_k_delta_list: list[list[float] | None] = []

    matches = 0
    used_for_t = 0

    for record in records:
        image = record.get("image")
        if not image:
            continue
        entry = lookup_image_alias(pred_index, str(image))
        if entry is None:
            continue

        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            continue
        gts = record.get("labels") or []
        if not isinstance(gts, list) or not gts:
            continue

        pairs = _match_dets_to_gts(
            [d for d in dets if isinstance(d, dict)],
            [g for g in gts if isinstance(g, dict)],
            iou_threshold=float(config.iou_threshold),
        )
        if not pairs:
            continue

        t_gt_per_inst = _extract_t_gt(record, len(gts))
        intr = _get_intrinsics(record)
        image_hw = config.image_hw_override or _get_image_hw(record)

        for det_idx, gt_idx in pairs:
            det = dets[det_idx]
            gt_t = t_gt_per_inst[gt_idx] if 0 <= gt_idx < len(t_gt_per_inst) else None

            # z prediction
            z_pred = None
            if "log_z" in det:
                try:
                    z_pred = float(math.exp(float(det["log_z"])))
                except _FLOAT_PARSE_ERRORS:
                    z_pred = None
            if z_pred is None and "z" in det:
                try:
                    z_pred = float(det["z"])
                except _FLOAT_PARSE_ERRORS:
                    z_pred = None
            if z_pred is None:
                continue

            # z supervision: prefer t_gt.z
            z_gt = None
            if gt_t is not None:
                z_gt = float(gt_t[2])
            if z_gt is None:
                continue

            z_pred_list.append(float(z_pred))
            z_gt_list.append(float(z_gt))
            matches += 1

            if not config.optimize_k_delta:
                continue
            if gt_t is None or intr is None or image_hw is None:
                continue

            bbox = det.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                cx_n = float(bbox["cx"])
                cy_n = float(bbox["cy"])
            except (KeyError, *_FLOAT_PARSE_ERRORS):
                continue

            offsets = det.get("offsets") or [0.0, 0.0]
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                offsets = [0.0, 0.0]

            h, w = float(image_hw[0]), float(image_hw[1])
            u = cx_n * w
            v = cy_n * h
            du = float(offsets[0])
            dv = float(offsets[1])

            fx, fy, cx0, cy0 = intr
            u_list.append(float(u))
            v_list.append(float(v))
            du_list.append(float(du))
            dv_list.append(float(dv))
            fx_list.append(float(fx))
            fy_list.append(float(fy))
            cx_list.append(float(cx0))
            cy_list.append(float(cy0))
            t_gt_list.append([float(gt_t[0]), float(gt_t[1]), float(gt_t[2])])

            kd = det.get("k_delta")
            if isinstance(kd, (list, tuple)) and len(kd) == 4:
                try:
                    det_k_delta_list.append([float(v) for v in kd])
                except _FLOAT_PARSE_ERRORS:
                    det_k_delta_list.append(None)
            else:
                det_k_delta_list.append(None)

            used_for_t += 1

    if matches == 0:
        raise ValueError("No matched detections with usable z supervision (need t_gt).")

    device = torch.device("cpu")
    dtype = torch.float32

    z_pred_t = torch.tensor(z_pred_list, dtype=dtype, device=device)
    z_gt_t = torch.tensor(z_gt_list, dtype=dtype, device=device)

    log_s = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
    shared_k = torch.zeros((4,), dtype=dtype, device=device, requires_grad=True)

    params = [log_s] + ([shared_k] if config.optimize_k_delta else [])

    opt = torch.optim.LBFGS(
        params,
        lr=float(config.lbfgs_lr),
        max_iter=int(config.lbfgs_max_iter),
        line_search_fn="strong_wolfe",
    )

    # Pre-build tensors for translation loss if enabled.
    if config.optimize_k_delta and used_for_t > 0:
        u_t = torch.tensor(u_list, dtype=dtype, device=device)
        v_t = torch.tensor(v_list, dtype=dtype, device=device)
        du_t = torch.tensor(du_list, dtype=dtype, device=device)
        dv_t = torch.tensor(dv_list, dtype=dtype, device=device)
        fx_t = torch.tensor(fx_list, dtype=dtype, device=device)
        fy_t = torch.tensor(fy_list, dtype=dtype, device=device)
        cx_t = torch.tensor(cx_list, dtype=dtype, device=device)
        cy_t = torch.tensor(cy_list, dtype=dtype, device=device)
        t_gt_t = torch.tensor(t_gt_list, dtype=dtype, device=device)
    else:
        u_t = v_t = du_t = dv_t = fx_t = fy_t = cx_t = cy_t = t_gt_t = None

    def closure():
        opt.zero_grad(set_to_none=True)
        s = torch.exp(log_s)
        z = (s * z_pred_t).clamp(min=1e-6)

        loss_z = F.smooth_l1_loss(z, z_gt_t)
        loss = float(config.w_z) * loss_z

        loss_t = None
        if config.optimize_k_delta and used_for_t > 0 and u_t is not None:
            # Apply shared k_delta only (per-det k_delta is handled later when writing outputs).
            dfx = shared_k[0]
            dfy = shared_k[1]
            dcx = shared_k[2]
            dcy = shared_k[3]
            fx_p = fx_t * (1.0 + dfx)
            fy_p = fy_t * (1.0 + dfy)
            cx_p = cx_t + dcx
            cy_p = cy_t + dcy

            u_p = u_t + du_t
            v_p = v_t + dv_t
            x = (u_p - cx_p) / fx_p.clamp(min=1e-6) * z
            y = (v_p - cy_p) / fy_p.clamp(min=1e-6) * z
            t_pred = torch.stack([x, y, z], dim=-1)
            loss_t = F.smooth_l1_loss(t_pred, t_gt_t)
            loss = loss + float(config.w_t) * loss_t

        reg = float(config.reg_log_s) * (log_s * log_s)
        if config.optimize_k_delta:
            reg = reg + float(config.reg_k) * (shared_k * shared_k).sum()
        loss = loss + reg

        loss.backward()
        return loss

    final_loss = opt.step(closure)

    s_value = float(torch.exp(log_s).detach().cpu().item())
    shared_k_value = None
    if config.optimize_k_delta:
        shared_k_value = [float(v) for v in shared_k.detach().cpu().tolist()]

    # Apply calibration to predictions.
    out_predictions: list[dict[str, Any]] = []
    log_s_value = math.log(max(s_value, 1e-12))

    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []
        new_dets: list[Any] = []
        for det in dets:
            if not isinstance(det, dict):
                new_dets.append(det)
                continue
            new_det = dict(det)
            if "log_z" in new_det:
                try:
                    new_det["log_z"] = float(new_det["log_z"]) + float(log_s_value)
                except (TypeError, ValueError):
                    new_det["log_z"] = new_det.get("log_z")
            elif "z" in new_det:
                try:
                    new_det["z"] = float(new_det["z"]) * float(s_value)
                except (TypeError, ValueError):
                    new_det["z"] = new_det.get("z")

            if shared_k_value is not None:
                kd = new_det.get("k_delta")
                kd_list = None
                if isinstance(kd, (list, tuple)) and len(kd) == 4:
                    try:
                        kd_list = [float(v) for v in kd]
                    except _FLOAT_PARSE_ERRORS:
                        kd_list = None
                # Compose in python (same formula as _compose_k_delta).
                dfx = (1.0 + (kd_list[0] if kd_list else 0.0)) * (1.0 + shared_k_value[0]) - 1.0
                dfy = (1.0 + (kd_list[1] if kd_list else 0.0)) * (1.0 + shared_k_value[1]) - 1.0
                dcx = (kd_list[2] if kd_list else 0.0) + shared_k_value[2]
                dcy = (kd_list[3] if kd_list else 0.0) + shared_k_value[3]
                new_det["k_delta"] = [float(dfx), float(dfy), float(dcx), float(dcy)]

            new_dets.append(new_det)
        new_entry["detections"] = new_dets
        out_predictions.append(new_entry)

    result = CalibResult(
        scale_s=s_value,
        shared_k_delta=shared_k_value,
        matches=matches,
        used_for_t=used_for_t,
        loss=float(final_loss.detach().cpu().item()) if hasattr(final_loss, "detach") else float(final_loss),
    )

    return out_predictions, result
