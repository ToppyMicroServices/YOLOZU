import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "wrap": False,
    "refine_offsets": True,
    "device": "cpu",
    "steps": 5,
    "damping": 1e-2,
    "fd_eps": 1e-2,
    "line_search": 3,
    "line_search_decay": 0.5,
    "w_reg": 1e-2,
    "w_depth": 1.0,
    "w_mask": 0.1,
    "max_step_px": 5.0,
    "max_total_update_px": 50.0,
    "tol_delta": 1e-3,
    "tol_loss": 1e-8,
    "log_steps": False,
    "dry_run": False,
}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Input predictions JSON.")
    p.add_argument("--output", required=True, help="Output predictions JSON.")
    p.add_argument("--config", default=None, help="Optional YAML/JSON config for refinement settings.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--enable", action="store_true", help="Enable Hessian refinement (opt-in).")
    mode.add_argument("--disable", action="store_true", help="Disable Hessian refinement (pass-through).")
    p.add_argument("--wrap", action="store_true", help="Wrap output as {predictions:[...], meta:{...}}.")
    p.add_argument("--refine-offsets", action="store_true", help="Refine per-detection offsets (experimental).")
    p.add_argument("--dataset", default=None, help="Optional YOLO dataset root for aux metadata (labels/*.json).")
    p.add_argument("--split", default=None, help="Optional dataset split for --dataset (e.g. val2017).")
    p.add_argument("--device", default=None, help="Torch device for refinement (default: cpu).")

    p.add_argument("--steps", type=int, default=None, help="Max Newton steps per detection (default: 5).")
    p.add_argument("--damping", type=float, default=None, help="Levenberg-Marquardt damping (default: 1e-2).")
    p.add_argument(
        "--fd-eps",
        type=float,
        default=None,
        help="Finite-difference epsilon for Hessian approximation (default: 1e-2).",
    )
    p.add_argument("--line-search", type=int, default=None, help="Line-search attempts per step (default: 3).")
    p.add_argument("--line-search-decay", type=float, default=None, help="Line-search decay (default: 0.5).")

    p.add_argument("--w-reg", type=float, default=None, help="L2 regularization weight for offset delta (default: 1e-2).")
    p.add_argument("--w-depth", type=float, default=None, help="Depth consistency weight (default: 1.0).")
    p.add_argument("--w-mask", type=float, default=None, help="Mask penalty weight (default: 0.1).")

    p.add_argument("--max-step-px", type=float, default=None, help="Max per-iteration update norm in pixels (default: 5).")
    p.add_argument(
        "--max-total-update-px",
        type=float,
        default=None,
        help="Max total |offset - offset0| norm in pixels (default: 50).",
    )
    p.add_argument("--tol-delta", type=float, default=None, help="Stop if update norm < tol (default: 1e-3).")
    p.add_argument("--tol-loss", type=float, default=None, help="Stop if loss improvement < tol (default: 1e-8).")

    p.add_argument("--log-output", default=None, help="Optional JSON log output path.")
    p.add_argument("--log-steps", action="store_true", help="Include per-iteration logs (can be large).")
    p.add_argument("--dry-run", action="store_true", help="Write schema-only output without changing values.")
    p.set_defaults(enable=None, disable=None, wrap=None, refine_offsets=None, log_steps=None, dry_run=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for --config YAML files") from exc
        obj = yaml.safe_load(text)
    else:
        obj = json.loads(text)

    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("config must be a JSON/YAML object")
    section = obj.get("hessian_refinement")
    if isinstance(section, dict):
        return section
    return obj


def _resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    config: dict[str, Any] = {}
    config_path = None
    if args.config:
        config_path = str(_resolve(str(args.config)))
        config = _load_config(Path(config_path))

    def pick(name: str) -> Any:
        cli = getattr(args, name)
        if cli is not None:
            return cli
        if name in config:
            return config[name]
        return _DEFAULTS[name]

    if args.enable is True:
        enabled = True
    elif args.disable is True:
        enabled = False
    elif "enabled" in config:
        enabled = bool(config.get("enabled"))
    else:
        enabled = bool(_DEFAULTS["enabled"])

    args.enabled = bool(enabled)
    args.config_path = config_path
    args.wrap = bool(pick("wrap"))
    args.refine_offsets = bool(pick("refine_offsets"))
    args.device = str(pick("device"))
    args.steps = int(pick("steps"))
    args.damping = float(pick("damping"))
    args.fd_eps = float(pick("fd_eps"))
    args.line_search = int(pick("line_search"))
    args.line_search_decay = float(pick("line_search_decay"))
    args.w_reg = float(pick("w_reg"))
    args.w_depth = float(pick("w_depth"))
    args.w_mask = float(pick("w_mask"))
    args.max_step_px = float(pick("max_step_px"))
    args.max_total_update_px = float(pick("max_total_update_px"))
    args.tol_delta = float(pick("tol_delta"))
    args.tol_loss = float(pick("tol_loss"))
    args.log_steps = bool(pick("log_steps"))
    args.dry_run = bool(pick("dry_run"))
    return args


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), list):
        meta = obj.get("meta")
        return list(obj["predictions"]), meta if isinstance(meta, dict) else None
    raise ValueError("unsupported predictions format")

def _try_import_torch():  # pragma: no cover - exercised in CI
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for refine_predictions_hessian when refinement is enabled") from exc
    return torch, F


def _load_2d(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value
    if hasattr(value, "shape"):
        return value
    if isinstance(value, str):
        path = _resolve(value)
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        if suffix in (".npy", ".npz"):
            try:
                import numpy as np
            except Exception:
                return None
            try:
                loaded = np.load(path)
            except Exception:
                return None
            if hasattr(loaded, "files"):
                if not loaded.files:
                    return None
                return loaded[loaded.files[0]]
            return loaded
    return None


def _shape_hw(array2d: Any) -> tuple[int, int] | None:
    if array2d is None:
        return None
    if hasattr(array2d, "shape"):
        try:
            h = int(array2d.shape[0])
            w = int(array2d.shape[1])
            return h, w
        except Exception:
            return None
    if isinstance(array2d, (list, tuple)) and array2d:
        if isinstance(array2d[0], (list, tuple)) and array2d[0]:
            return len(array2d), len(array2d[0])
    return None


def _as_torch_map(torch, array2d: Any, *, device: str) -> Any:
    if array2d is None:
        return None
    if isinstance(array2d, torch.Tensor):
        t = array2d
    else:
        t = torch.tensor(array2d, dtype=torch.float32)
    if t.ndim != 2:
        return None
    return t.to(device=device, dtype=torch.float32)


def _sample_map(torch, F, map_hw: Any, *, u_px: Any, v_px: Any) -> Any:
    # map_hw: (H,W) float32 tensor
    h = int(map_hw.shape[0])
    w = int(map_hw.shape[1])
    if h <= 0 or w <= 0:
        return map_hw.mean() * 0.0
    if h == 1 and w == 1:
        return map_hw[0, 0]

    u = u_px
    v = v_px
    if w > 1:
        x = (u / float(w - 1)) * 2.0 - 1.0
    else:
        x = u * 0.0
    if h > 1:
        y = (v / float(h - 1)) * 2.0 - 1.0
    else:
        y = v * 0.0
    grid = torch.stack((x, y), dim=-1).reshape(1, 1, 1, 2)
    inp = map_hw.reshape(1, 1, h, w)
    sampled = F.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.reshape(())


def _float2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _parse_bbox_center(det: dict[str, Any]) -> tuple[float, float] | None:
    bb = det.get("bbox")
    if not isinstance(bb, dict):
        return None
    try:
        cx = float(bb.get("cx", 0.0))
        cy = float(bb.get("cy", 0.0))
    except Exception:
        return None
    return cx, cy


def _to_pixel_center(cxcy_norm: tuple[float, float], *, hw: tuple[int, int]) -> tuple[float, float]:
    cx, cy = cxcy_norm
    h, w = hw
    u = cx * float(max(1, w - 1))
    v = cy * float(max(1, h - 1))
    return u, v


def _invert_2x2(torch, h: Any, *, eps: float = 1e-12) -> Any | None:
    a = h[0, 0]
    b = h[0, 1]
    c = h[1, 0]
    d = h[1, 1]
    det = a * d - b * c
    if not torch.isfinite(det) or float(det.abs()) < float(eps):
        return None
    inv_det = 1.0 / det
    return torch.stack((torch.stack((d, -b)), torch.stack((-c, a)))) * inv_det


def _project_offsets(
    torch,
    offsets: Any,
    *,
    offsets0: Any,
    bbox_center_px: tuple[float, float] | None,
    hw: tuple[int, int] | None,
    max_total_update_px: float,
) -> Any:
    out = offsets

    delta = out - offsets0
    delta_norm = torch.linalg.norm(delta)
    if torch.isfinite(delta_norm) and float(delta_norm) > float(max_total_update_px) > 0.0:
        out = offsets0 + delta * (float(max_total_update_px) / float(delta_norm))

    if bbox_center_px is not None and hw is not None:
        u0, v0 = bbox_center_px
        h, w = hw
        min_du = -float(u0)
        max_du = float(max(0, w - 1)) - float(u0)
        min_dv = -float(v0)
        max_dv = float(max(0, h - 1)) - float(v0)
        out = torch.stack(
            (
                out[0].clamp(min=min_du, max=max_du),
                out[1].clamp(min=min_dv, max=max_dv),
            )
        )

    return out


def _refine_offsets(
    *,
    offsets0_xy: tuple[float, float],
    bbox_center_px: tuple[float, float] | None,
    z_target: float | None,
    depth_map: Any,
    mask_map: Any,
    device: str,
    steps: int,
    damping: float,
    fd_eps: float,
    line_search: int,
    line_search_decay: float,
    w_reg: float,
    w_depth: float,
    w_mask: float,
    max_step_px: float,
    max_total_update_px: float,
    tol_delta: float,
    tol_loss: float,
    log_steps: bool,
) -> tuple[tuple[float, float], dict[str, Any]]:
    torch, F = _try_import_torch()

    offsets0 = torch.tensor(offsets0_xy, dtype=torch.float32, device=device)
    offsets = offsets0.clone().detach().requires_grad_(True)

    depth_t = _as_torch_map(torch, depth_map, device=device)
    mask_t = _as_torch_map(torch, mask_map, device=device)
    hw = _shape_hw(depth_map) or _shape_hw(mask_map)

    warnings: list[str] = []
    if steps <= 0:
        return offsets0_xy, {"enabled": True, "steps_run": 0, "stop_reason": "steps_le_0", "warnings": warnings}

    if bbox_center_px is None:
        warnings.append("missing_bbox_center")
    if z_target is None:
        warnings.append("missing_log_z")
    if depth_t is None:
        warnings.append("missing_depth_map")

    if bbox_center_px is None or z_target is None or depth_t is None or hw is None:
        return offsets0_xy, {
            "enabled": True,
            "steps_run": 0,
            "stop_reason": "no_signal",
            "warnings": warnings,
        }

    u0, v0 = bbox_center_px
    z_t = torch.tensor(float(z_target), dtype=torch.float32, device=device)

    def loss_only(curr_offsets: Any) -> Any:
        return loss_terms(curr_offsets)["loss_total"]

    def grad_at(curr_offsets: Any) -> Any:
        curr_offsets = curr_offsets.clone().detach().requires_grad_(True)
        loss = loss_only(curr_offsets)
        grad = torch.autograd.grad(loss, curr_offsets, create_graph=False, retain_graph=False)[0]
        return loss.detach(), grad.detach()

    def loss_terms(curr_offsets: Any) -> dict[str, Any]:
        du = curr_offsets[0]
        dv = curr_offsets[1]
        u = torch.tensor(float(u0), dtype=torch.float32, device=device) + du
        v = torch.tensor(float(v0), dtype=torch.float32, device=device) + dv

        depth_sample = _sample_map(torch, F, depth_t, u_px=u, v_px=v)
        mask_sample = None if mask_t is None else _sample_map(torch, F, mask_t, u_px=u, v_px=v).clamp(0.0, 1.0)

        delta = curr_offsets - offsets0
        loss_reg = float(w_reg) * (delta * delta).sum()

        loss_depth = depth_sample * 0.0
        if float(w_depth) != 0.0:
            loss_depth = float(w_depth) * (depth_sample - z_t) ** 2

        loss_mask = depth_sample * 0.0
        if mask_sample is not None and float(w_mask) != 0.0:
            loss_mask = float(w_mask) * (1.0 - mask_sample) ** 2

        loss_total = loss_reg + loss_depth + loss_mask
        return {
            "loss_total": loss_total,
            "loss_reg": loss_reg,
            "loss_depth": loss_depth,
            "loss_mask": loss_mask,
            "depth_sample": depth_sample,
            "mask_sample": mask_sample,
        }

    history: list[dict[str, Any]] = []

    with torch.no_grad():
        terms0 = loss_terms(offsets0)
        loss_prev = float(terms0["loss_total"].detach().cpu())

    steps_run = 0
    stop_reason = "max_steps"

    for it in range(int(steps)):
        terms = loss_terms(offsets)
        loss = terms["loss_total"]
        if not torch.isfinite(loss):
            stop_reason = "nan_loss"
            break

        grad = torch.autograd.grad(loss, offsets, create_graph=False, retain_graph=False)[0]
        if grad is None or grad.numel() != 2 or not torch.isfinite(grad).all():
            stop_reason = "nan_grad"
            break

        eps = float(fd_eps)
        if not (eps > 0.0):
            eps = 1e-2
        e0 = torch.tensor((eps, 0.0), dtype=torch.float32, device=device)
        e1 = torch.tensor((0.0, eps), dtype=torch.float32, device=device)
        _loss_p0, g_p0 = grad_at(offsets.detach() + e0)
        _loss_m0, g_m0 = grad_at(offsets.detach() - e0)
        _loss_p1, g_p1 = grad_at(offsets.detach() + e1)
        _loss_m1, g_m1 = grad_at(offsets.detach() - e1)

        col0 = (g_p0 - g_m0) * (0.5 / eps)
        col1 = (g_p1 - g_m1) * (0.5 / eps)
        hess = torch.stack((col0, col1), dim=1)
        hess = 0.5 * (hess + hess.T)
        if not torch.isfinite(hess).all():
            stop_reason = "nan_hess"
            break

        h_damped = hess + torch.eye(2, dtype=hess.dtype, device=hess.device) * float(damping)
        inv = _invert_2x2(torch, h_damped)
        if inv is None:
            step = -grad.detach()
        else:
            step = -(inv @ grad.detach())

        step_norm = torch.linalg.norm(step)
        if not torch.isfinite(step_norm):
            stop_reason = "nan_step"
            break
        if float(step_norm) < float(tol_delta):
            stop_reason = "delta_tol"
            break
        if float(step_norm) > float(max_step_px) > 0.0:
            step = step * (float(max_step_px) / float(step_norm))
            step_norm = torch.linalg.norm(step)

        best_offsets = None
        best_loss = loss_prev
        best_terms = None
        scale = 1.0
        with torch.no_grad():
            for _ in range(max(0, int(line_search)) + 1):
                candidate = offsets.detach() + step * float(scale)
                candidate = _project_offsets(
                    torch,
                    candidate,
                    offsets0=offsets0,
                    bbox_center_px=bbox_center_px,
                    hw=hw,
                    max_total_update_px=float(max_total_update_px),
                )
                cand_terms = loss_terms(candidate)
                cand_loss = float(cand_terms["loss_total"].detach().cpu())
                if not math.isfinite(cand_loss):
                    scale *= float(line_search_decay)
                    continue
                if cand_loss <= best_loss - float(tol_loss):
                    best_offsets = candidate
                    best_loss = cand_loss
                    best_terms = cand_terms
                    break
                scale *= float(line_search_decay)

        if best_offsets is None:
            stop_reason = "no_improve"
            break

        steps_run += 1
        offsets = best_offsets.detach().requires_grad_(True)

        if log_steps:
            entry = {
                "iter": int(it),
                "loss_before": float(loss_prev),
                "loss_after": float(best_loss),
                "step_norm": float(step_norm.detach().cpu()),
                "line_search_scale": float(scale),
                "hessian": "finite_diff",
                "fd_eps": float(eps),
                "grad": [float(v) for v in grad.detach().cpu().tolist()],
                "step": [float(v) for v in step.detach().cpu().tolist()],
            }
            if best_terms is not None:
                entry["depth_sample"] = float(best_terms["depth_sample"].detach().cpu())
                ms = best_terms.get("mask_sample")
                entry["mask_sample"] = None if ms is None else float(ms.detach().cpu())
            history.append(entry)

        if loss_prev - best_loss < float(tol_loss):
            stop_reason = "loss_tol"
            break
        loss_prev = best_loss

    with torch.no_grad():
        final_terms = loss_terms(offsets.detach())
        loss_final = float(final_terms["loss_total"].detach().cpu())
        depth_final = float(final_terms["depth_sample"].detach().cpu())
        mask_final_t = final_terms.get("mask_sample")
        mask_final = None if mask_final_t is None else float(mask_final_t.detach().cpu())
        offsets_final_xy = (float(offsets.detach()[0].cpu()), float(offsets.detach()[1].cpu()))

    report: dict[str, Any] = {
        "enabled": True,
        "hessian": "finite_diff",
        "fd_eps": float(fd_eps),
        "steps_requested": int(steps),
        "steps_run": int(steps_run),
        "stop_reason": str(stop_reason),
        "warnings": warnings,
        "loss_before": float(terms0["loss_total"].detach().cpu()),
        "loss_after": float(loss_final),
        "depth_before": float(terms0["depth_sample"].detach().cpu()),
        "depth_after": float(depth_final),
        "mask_after": mask_final,
        "offsets_before": [float(offsets0_xy[0]), float(offsets0_xy[1])],
        "offsets_after": [float(offsets_final_xy[0]), float(offsets_final_xy[1])],
        "delta": [float(offsets_final_xy[0] - offsets0_xy[0]), float(offsets_final_xy[1] - offsets0_xy[1])],
    }
    if log_steps:
        report["steps"] = history
    return offsets_final_xy, report


def _load_aux_from_entry(entry: dict[str, Any]) -> tuple[Any, Any]:
    depth = _load_2d(entry.get("D_obj") or entry.get("depth") or entry.get("depth_path"))
    mask = _load_2d(entry.get("M") or entry.get("mask") or entry.get("mask_path"))
    return depth, mask


def _build_dataset_index(dataset_root: str, *, split: str | None) -> dict[str, dict[str, Any]]:
    # Lightweight: reuse rtdetr_pose dataset metadata reader (labels/*.json).
    repo_root_local = repo_root
    sys.path.insert(0, str(repo_root_local / "rtdetr_pose"))
    from rtdetr_pose.dataset import load_yolo_dataset  # type: ignore

    records = load_yolo_dataset(dataset_root, split=(split or "train2017"))
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        path = str(rec.get("image_path") or "")
        if path:
            index[path] = rec
            try:
                index[str(Path(path).resolve())] = rec
            except OSError as exc:
                logger.debug("failed to resolve dataset path %s: %s", path, exc)
            index[Path(path).name] = rec
    return index


def _load_aux_from_dataset(index: dict[str, dict[str, Any]], image: str) -> tuple[Any, Any]:
    rec = index.get(image) or index.get(str(Path(image).resolve())) or index.get(Path(image).name)
    if not rec:
        return None, None
    depth = _load_2d(rec.get("D_obj") or rec.get("depth") or rec.get("depth_path"))
    mask = _load_2d(rec.get("M") or rec.get("mask") or rec.get("mask_path"))
    return depth, mask


def _refine_entry(
    entry: dict[str, Any],
    *,
    refine_offsets: bool,
    dry_run: bool,
    dataset_index: dict[str, dict[str, Any]] | None,
    args: argparse.Namespace,
    log_images: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    dets = entry.get("detections")
    if not isinstance(dets, list):
        return dict(entry)

    depth_map, mask_map = _load_aux_from_entry(entry)
    if (depth_map is None and mask_map is None) and dataset_index is not None:
        depth_map, mask_map = _load_aux_from_dataset(dataset_index, str(entry.get("image", "")))

    hw = _shape_hw(depth_map) or _shape_hw(mask_map)

    new_entry = dict(entry)
    new_dets = []
    log_entry = None
    if log_images is not None:
        log_entry = {"image": entry.get("image"), "hw": None if hw is None else [int(hw[0]), int(hw[1])], "detections": []}
    for det in dets:
        if not isinstance(det, dict):
            new_dets.append(det)
            continue

        new_det = dict(det)
        refined = []
        warnings: list[str] = []

        if refine_offsets:
            offsets = new_det.get("offsets")
            offsets_xy = _float2(offsets)
            if offsets_xy is not None:
                refined.append("offsets")
                if dry_run:
                    report = {
                        "enabled": True,
                        "steps_requested": int(args.steps),
                        "steps_run": 0,
                        "stop_reason": "dry_run",
                        "warnings": [],
                        "offsets_before": [float(offsets_xy[0]), float(offsets_xy[1])],
                        "offsets_after": [float(offsets_xy[0]), float(offsets_xy[1])],
                        "delta": [0.0, 0.0],
                    }
                else:
                    bbox_center_norm = _parse_bbox_center(new_det)
                    bbox_center_px = None
                    if bbox_center_norm is None:
                        warnings.append("missing_bbox")
                    elif hw is not None:
                        bbox_center_px = _to_pixel_center(bbox_center_norm, hw=hw)
                    else:
                        warnings.append("missing_hw")

                    z_target = None
                    if "log_z" in new_det:
                        try:
                            z_target = float(math.exp(float(new_det["log_z"])))
                        except Exception:
                            warnings.append("bad_log_z")

                    offsets_final, report = _refine_offsets(
                        offsets0_xy=offsets_xy,
                        bbox_center_px=bbox_center_px,
                        z_target=z_target,
                        depth_map=depth_map,
                        mask_map=mask_map,
                        device=str(args.device),
                        steps=int(args.steps),
                        damping=float(args.damping),
                        fd_eps=float(args.fd_eps),
                        line_search=int(args.line_search),
                        line_search_decay=float(args.line_search_decay),
                        w_reg=float(args.w_reg),
                        w_depth=float(args.w_depth),
                        w_mask=float(args.w_mask),
                        max_step_px=float(args.max_step_px),
                        max_total_update_px=float(args.max_total_update_px),
                        tol_delta=float(args.tol_delta),
                        tol_loss=float(args.tol_loss),
                        log_steps=bool(args.log_steps),
                    )
                    new_det["offsets"] = [float(offsets_final[0]), float(offsets_final[1])]
            else:
                warnings.append("missing_offsets")

        new_det["hessian_refinement"] = {
            "enabled": bool(refine_offsets),
            "refined": refined,
            "warnings": warnings,
        }
        if refine_offsets and offsets_xy is not None:
            new_det["hessian_refinement"]["offsets"] = report

        if log_entry is not None:
            det_log = {
                "class_id": new_det.get("class_id"),
                "score": new_det.get("score"),
                "refined": refined,
                "warnings": warnings,
            }
            if refine_offsets and offsets_xy is not None:
                det_log["offsets"] = report
            log_entry["detections"].append(det_log)
        new_dets.append(new_det)

    new_entry["detections"] = new_dets
    if log_entry is not None:
        log_images.append(log_entry)
    return new_entry


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args = _resolve_runtime_args(args)

    in_path = _resolve(args.predictions)
    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predictions, meta_in = _load_predictions(in_path)
    dataset_index = None
    if args.dataset:
        dataset_index = _build_dataset_index(str(args.dataset), split=args.split)

    log_images: list[dict[str, Any]] | None = [] if args.log_output else None

    refined = []
    for p in predictions:
        if not isinstance(p, dict):
            refined.append(p)
            continue
        refined.append(
            _refine_entry(
                p,
                refine_offsets=bool(args.enabled and args.refine_offsets),
                dry_run=bool(args.dry_run),
                dataset_index=dataset_index,
                args=args,
                log_images=log_images,
            )
        )

    if args.wrap:
        payload: dict[str, Any] = {
            "predictions": refined,
            "meta": {
                "timestamp": _now_utc(),
                "tool": "refine_predictions_hessian",
                "enabled": bool(args.enabled),
                "refine_offsets": bool(args.enabled and args.refine_offsets),
                "config": {
                    "config_path": args.config_path,
                    "dataset": args.dataset,
                    "split": args.split,
                    "device": args.device,
                    "steps": int(args.steps),
                    "damping": float(args.damping),
                    "fd_eps": float(args.fd_eps),
                    "line_search": int(args.line_search),
                    "line_search_decay": float(args.line_search_decay),
                    "w_reg": float(args.w_reg),
                    "w_depth": float(args.w_depth),
                    "w_mask": float(args.w_mask),
                    "max_step_px": float(args.max_step_px),
                    "max_total_update_px": float(args.max_total_update_px),
                    "tol_delta": float(args.tol_delta),
                    "tol_loss": float(args.tol_loss),
                    "log_output": args.log_output,
                },
                "dry_run": bool(args.dry_run),
                "input_meta": meta_in,
            },
        }
    else:
        payload = refined

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.log_output and log_images is not None:
        log_path = _resolve(str(args.log_output))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "timestamp": _now_utc(),
            "tool": "refine_predictions_hessian",
            "predictions": str(in_path),
            "output": str(out_path),
            "enabled": bool(args.enabled),
            "config": {
                "config_path": args.config_path,
                "dataset": args.dataset,
                "split": args.split,
                "device": args.device,
                "steps": int(args.steps),
                "damping": float(args.damping),
                "fd_eps": float(args.fd_eps),
                "line_search": int(args.line_search),
                "line_search_decay": float(args.line_search_decay),
                "w_reg": float(args.w_reg),
                "w_depth": float(args.w_depth),
                "w_mask": float(args.w_mask),
                "max_step_px": float(args.max_step_px),
                "max_total_update_px": float(args.max_total_update_px),
                "tol_delta": float(args.tol_delta),
                "tol_loss": float(args.tol_loss),
                "log_steps": bool(args.log_steps),
            },
            "images": log_images,
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
