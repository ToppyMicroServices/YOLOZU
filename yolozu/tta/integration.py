from __future__ import annotations

import copy
import math
import platform
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .config import SUPPORTED_TTT_METHODS, TTTConfig
from .method_profiles import get_ttt_method_profile
from .tent import TentConfig, TentRunner

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TTTReport:
    enabled: bool
    method: str
    reset: str
    steps_requested: int
    steps_run: int
    batches_used: int
    seconds: float
    losses: list[float]
    mask_ratio: float | None
    updated_param_count: int | None
    warnings: list[str]
    stopped_early: bool = False
    stop_reason: str | None = None
    step_metrics: list[dict[str, Any]] | None = None
    method_profile: dict[str, Any] | None = None
    forward_calls: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    update_ratio: float = 0.0
    memory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        losses = list(self.losses or [])
        if losses:
            data["loss_summary"] = {
                "count": int(len(losses)),
                "min": float(min(losses)),
                "mean": float(sum(losses) / float(len(losses))),
                "max": float(max(losses)),
                "last": float(losses[-1]),
            }
        else:
            data["loss_summary"] = {"count": 0, "min": None, "mean": None, "max": None, "last": None}

        max_keep = 50
        if len(losses) > max_keep:
            data["losses_total"] = int(len(losses))
            data["losses_truncated"] = True
            data["losses"] = losses[:max_keep]
        else:
            data["losses_total"] = int(len(losses))
            data["losses_truncated"] = False

        step_metrics = list(self.step_metrics or [])
        if len(step_metrics) > max_keep:
            data["step_metrics_total"] = int(len(step_metrics))
            data["step_metrics_truncated"] = True
            data["step_metrics"] = step_metrics[:max_keep]
        else:
            data["step_metrics_total"] = int(len(step_metrics))
            data["step_metrics_truncated"] = False
        return data


def _ensure_torch():
    if torch is None:
        raise RuntimeError("torch is required for TTT")


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _memory_start(model: Any) -> dict[str, Any]:
    device_type = "cpu"
    try:
        first_param = next(model.parameters())
        device_type = str(first_param.device.type)
    except (AttributeError, StopIteration):
        pass

    if device_type == "cuda" and torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return {
            "backend": "torch_cuda",
            "metric": "peak_allocated_bytes",
            "start_allocated_bytes": int(torch.cuda.memory_allocated()),
        }
    if device_type == "mps" and torch is not None and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return {
                "backend": "torch_mps",
                "metric": "current_allocated_bytes",
                "start_allocated_bytes": int(current()),
            }
    return {
        "backend": "process",
        "metric": "peak_rss_bytes",
        "start_peak_rss_bytes": _process_peak_rss_bytes(),
    }


def _memory_finish(start: dict[str, Any]) -> dict[str, Any]:
    result = dict(start)
    backend = str(start.get("backend") or "")
    if backend == "torch_cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["end_allocated_bytes"] = int(torch.cuda.memory_allocated())
    elif backend == "torch_mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            result["end_allocated_bytes"] = int(current())
    else:
        result["end_peak_rss_bytes"] = _process_peak_rss_bytes()
    return result


def _count_unique_params(params: Iterable["torch.Tensor"]) -> int:
    seen: set[int] = set()
    total = 0
    for param in params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.numel())
    return total


def _take_batches(loader: Iterable[Any], *, max_batches: int) -> list[Any]:
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")
    batches = []
    it = iter(loader)
    for _ in range(int(max_batches)):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    return batches


def _snapshot_params(params: Iterable["torch.Tensor"]) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
    _ensure_torch()
    snap: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            snap.append((p, p.detach().clone()))
    return snap


def _restore_params(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> None:
    _ensure_torch()
    with torch.no_grad():
        for p, value in snapshot:
            p.copy_(value)


def _snapshot_norm_buffers(model: Any) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
    _ensure_torch()
    snap: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        named_buffers = model.named_buffers()
    except Exception:
        return snap

    with torch.no_grad():
        for name, buffer in named_buffers:
            if buffer is None:
                continue
            name = str(name)
            if not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                continue
            try:
                snap.append((buffer, buffer.detach().clone()))
            except Exception:  # pragma: no cover
                continue
    return snap


def _restore_buffers(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> None:
    _ensure_torch()
    with torch.no_grad():
        for buffer, value in snapshot:
            buffer.copy_(value)


def _global_l2_update_norm(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> float:
    _ensure_torch()
    total = None
    for p, base in snapshot:
        delta = (p.detach() - base).to(dtype=torch.float32)
        if total is None:
            total = torch.zeros((), device=delta.device, dtype=torch.float32)
        total = total + (delta * delta).sum()
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def _global_grad_norm(params: Iterable["torch.Tensor"]) -> float:
    _ensure_torch()
    total = None
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        g2 = grad.detach().to(dtype=torch.float32)
        if total is None:
            total = torch.zeros((), device=g2.device, dtype=torch.float32)
        total = total + (g2 * g2).sum()
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def _extract_logits(output: Any) -> "torch.Tensor":
    _ensure_torch()
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "pred_logits" in output:
            return output["pred_logits"]
        if "recon" in output:
            return output["recon"]
    if isinstance(output, (list, tuple)) and output:
        out0 = output[0]
        if torch.is_tensor(out0):
            return out0
    if torch.is_tensor(output):
        return output
    raise RuntimeError("failed to extract logits tensor from model output")


def _entropy_loss_from_logits(logits: "torch.Tensor") -> "torch.Tensor":
    _ensure_torch()
    if logits.shape[-1] <= 1:
        return logits.sum() * 0.0
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return entropy.mean()


def _eata_sample_signals(logits: "torch.Tensor", *, conf_min: float) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    _ensure_torch()
    if logits.shape[-1] <= 1:
        batch = int(logits.shape[0]) if logits.ndim >= 1 else 1
        conf = torch.zeros((batch,), dtype=torch.float32, device=logits.device)
        entropy = torch.zeros((batch,), dtype=torch.float32, device=logits.device)
        valid = torch.zeros((batch,), dtype=torch.float32, device=logits.device)
        return conf, entropy, valid

    probs = torch.softmax(logits, dim=-1)
    conf_map = probs.amax(dim=-1)
    entropy_map = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    if conf_map.ndim == 1:
        conf = conf_map
        entropy = entropy_map
        valid = torch.ones_like(conf_map)
    else:
        batch = int(conf_map.shape[0])
        conf_flat = conf_map.reshape(batch, -1)
        entropy_flat = entropy_map.reshape(batch, -1)
        conf = conf_flat.mean(dim=1)
        entropy = entropy_flat.mean(dim=1)
        valid = (conf_flat >= float(conf_min)).to(dtype=torch.float32).sum(dim=1)
    return conf, entropy, valid


def _eata_anchor_loss(snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> "torch.Tensor":
    _ensure_torch()
    total = None
    for param, base in snapshot:
        diff = (param - base).to(dtype=torch.float32)
        part = (diff * diff).mean()
        total = part if total is None else (total + part)
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total


def _sar_perturb_and_backup(
    params: Iterable["torch.Tensor"],
    *,
    rho: float,
    adaptive: bool,
    first_step_scale: float,
) -> tuple[list[tuple["torch.Tensor", "torch.Tensor"]], float]:
    _ensure_torch()
    rho_value = max(0.0, float(rho)) * max(0.0, float(first_step_scale))
    if rho_value <= 0.0:
        return [], 0.0

    grad_norm_sq = None
    params_list = list(params)
    for param in params_list:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        direction = grad.detach().to(dtype=torch.float32)
        if adaptive:
            direction = direction * param.detach().abs().to(dtype=torch.float32)
        term = (direction * direction).sum()
        grad_norm_sq = term if grad_norm_sq is None else (grad_norm_sq + term)
    if grad_norm_sq is None:
        return [], 0.0

    grad_norm = float(torch.sqrt(grad_norm_sq).detach().cpu().item())
    if grad_norm <= 0.0:
        return [], 0.0

    scale = rho_value / (grad_norm + 1e-12)
    backup: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for param in params_list:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            backup.append((param, param.detach().clone()))
            perturb = grad
            if adaptive:
                perturb = perturb * param.detach().abs()
            param.add_(perturb, alpha=float(scale))
    return backup, float(rho_value)


def _sar_restore(backup: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> None:
    _ensure_torch()
    with torch.no_grad():
        for param, value in backup:
            param.copy_(value)


def _forward_cotta_augs(
    model: Any,
    x: "torch.Tensor",
    *,
    augmentations: Iterable[str],
    aggregation: str,
) -> tuple["torch.Tensor", dict[str, Any]]:
    _ensure_torch()
    aug_names = [str(name).lower() for name in augmentations if str(name).strip()]
    if not aug_names:
        aug_names = ["identity"]

    logits_list: list[torch.Tensor] = []
    used: list[str] = []
    for aug in aug_names:
        if aug == "identity":
            x_aug = x
            invert = False
        elif aug == "hflip":
            if x.ndim < 4:
                continue
            x_aug = torch.flip(x, dims=(-1,))
            invert = True
        else:
            continue

        logits = _extract_logits(model(x_aug))
        if invert and logits.ndim >= 4:
            logits = torch.flip(logits, dims=(-1,))
        logits_list.append(logits)
        used.append(aug)

    if not logits_list:
        raise RuntimeError("no valid CoTTA augmentations for current batch")

    if len(logits_list) == 1:
        agg_logits = logits_list[0]
    elif str(aggregation).lower() == "confidence_weighted_mean":
        weighted = None
        denom = None
        for logits in logits_list:
            if logits.shape[-1] <= 1:
                weight = torch.ones_like(logits)
            else:
                conf = torch.softmax(logits, dim=-1).amax(dim=-1, keepdim=True)
                weight = conf.expand_as(logits)
            contrib = logits * weight
            weighted = contrib if weighted is None else (weighted + contrib)
            denom = weight if denom is None else (denom + weight)
        agg_logits = weighted / denom.clamp_min(1e-8)
    else:
        agg_logits = torch.stack(logits_list, dim=0).mean(dim=0)

    return agg_logits, {"augmentations": used, "aggregation": str(aggregation), "branches": int(len(logits_list))}


def _ema_update(teacher: Any, student: Any, *, momentum: float) -> None:
    _ensure_torch()
    m = float(momentum)
    m = 0.0 if m < 0.0 else (1.0 if m > 1.0 else m)
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.mul_(m).add_(student_param.detach(), alpha=(1.0 - m))


def _stochastic_restore(
    snapshot: Iterable[tuple["torch.Tensor", "torch.Tensor"]],
    *,
    prob: float,
    generator: "torch.Generator | None",
) -> int:
    _ensure_torch()
    p = float(prob)
    if p <= 0.0:
        return 0
    if p >= 1.0:
        with torch.no_grad():
            total = 0
            for param, base in snapshot:
                param.copy_(base)
                total += int(param.numel())
        return total

    total = 0
    with torch.no_grad():
        for param, base in snapshot:
            mask = torch.rand(
                tuple(param.shape),
                device=param.device,
                generator=generator,
            ) < p
            if not bool(mask.any()):
                continue
            param.copy_(torch.where(mask, base, param))
            total += int(mask.sum().detach().cpu().item())
    return total


def run_ttt(adapter: Any, records: list[dict[str, Any]], *, config: TTTConfig) -> TTTReport:
    if not config.enabled:
        return TTTReport(
            enabled=False,
            method=str(config.method),
            reset=str(config.reset),
            steps_requested=int(config.steps),
            steps_run=0,
            batches_used=0,
            seconds=0.0,
            losses=[],
            mask_ratio=None,
            updated_param_count=None,
            warnings=[],
            stopped_early=False,
            stop_reason=None,
            step_metrics=[],
            method_profile=get_ttt_method_profile(str(config.method)),
            forward_calls=0,
            backward_calls=0,
            optimizer_steps=0,
            update_ratio=0.0,
            memory=None,
        )

    _ensure_torch()

    if not hasattr(adapter, "get_model") or not hasattr(adapter, "build_loader"):
        raise RuntimeError("TTT requires an adapter with get_model() and build_loader()")

    model = adapter.get_model()
    if model is None:
        raise RuntimeError("TTT requires a non-null model from adapter.get_model()")

    loader = adapter.build_loader(records, batch_size=int(config.batch_size))
    batches = _take_batches(loader, max_batches=int(config.max_batches))
    if not batches:
        raise RuntimeError("TTT requires at least one batch from adapter.build_loader()")

    method = (config.method or "tent").lower()
    if method not in SUPPORTED_TTT_METHODS:
        raise ValueError(f"TTT method must be one of: {', '.join(SUPPORTED_TTT_METHODS)}")

    steps = int(config.steps)
    if steps <= 0:
        raise ValueError("steps must be > 0")

    warnings: list[str] = []
    losses: list[float] = []
    mask_ratios: list[float] = []
    updated_param_count: int | None = None
    step_metrics: list[dict[str, Any]] = []
    stopped_early = False
    stop_reason: str | None = None
    forward_calls = 0
    backward_calls = 0
    optimizer_steps = 0

    was_training = bool(getattr(model, "training", False))

    generator = None
    if config.seed is not None:
        try:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(config.seed))
        except Exception:  # pragma: no cover
            warnings.append("failed_to_seed_generator")
            generator = None

    memory_start = _memory_start(model)
    start = time.time()
    try:
        initial_loss: float | None = None
        if method == "tent":
            runner = TentRunner(
                model,
                config=TentConfig(
                    lr=float(config.lr),
                    include=config.include,
                    exclude=config.exclude,
                    update_filter=str(config.update_filter),
                    max_grad_norm=config.max_grad_norm,
                    aux_pose_weight=float(getattr(config, "aux_pose_weight", 0.0)),
                    aux_keypoints_weight=float(getattr(config, "aux_keypoints_weight", 0.0)),
                    aux_depth_weight=float(getattr(config, "aux_depth_weight", 0.0)),
                    aux_seg_weight=float(getattr(config, "aux_seg_weight", 0.0)),
                ),
            )
            params = list(getattr(runner, "params", []))
            if not params:
                raise RuntimeError("no parameters selected for TTT")
            updated_param_count = int(getattr(runner, "updated_param_count", _count_unique_params(params)))
            base_snapshot = _snapshot_params(params)
            for step_idx in range(steps):
                batch = batches[step_idx % len(batches)]
                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []
                metrics = runner.adapt_step(batch)
                forward_calls += int(metrics.get("forward_calls", 1))
                backward_calls += int(metrics.get("backward_calls", 1))
                optimizer_steps += int(metrics.get("optimizer_steps", 1))
                loss_value = float(metrics.get("loss_total", metrics.get("loss_entropy", 0.0)))
                if initial_loss is None:
                    initial_loss = loss_value

                step_grad_norm = metrics.get("grad_norm")
                step_grad_norm_clipped = metrics.get("grad_norm_clipped")
                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_entry: dict[str, Any] = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "grad_norm": (float(step_grad_norm) if step_grad_norm is not None else None),
                    "grad_norm_clipped": (float(step_grad_norm_clipped) if step_grad_norm_clipped is not None else None),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                }
                for metric_name in (
                    "loss_entropy",
                    "loss_aux_consistency",
                    "aux_target_current_batch",
                    "aux_student_view_mean_abs_delta",
                ):
                    if metric_name in metrics:
                        step_entry[metric_name] = float(metrics[metric_name])

                non_finite_fields: list[str] = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))
                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                step_metrics.append(step_entry)
        elif method == "mim":
            from .ttt_mim import select_parameters, supports_structured_mim, ttt_mim_step, ttt_structured_mim_step

            params = select_parameters(
                model,
                update_filter=str(config.update_filter),
                include=config.include,
                exclude=config.exclude,
            )
            if not params:
                raise RuntimeError("no parameters selected for TTT")
            optimizer = torch.optim.Adam(params, lr=float(config.lr))
            updated_param_count = _count_unique_params(params)
            base_snapshot = _snapshot_params(params)
            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []
                if supports_structured_mim(model):
                    loss, _mask_ratio, extra = ttt_structured_mim_step(
                        model,
                        optimizer,
                        x,
                        mask_prob=float(config.mim_mask_prob),
                        patch_size=int(config.mim_patch_size),
                        mask_value=float(config.mim_mask_value),
                        generator=generator,
                        max_grad_norm=config.max_grad_norm,
                    )
                else:
                    loss, _mask_ratio, extra = ttt_mim_step(
                        model,
                        optimizer,
                        x,
                        mask_prob=float(config.mim_mask_prob),
                        patch_size=int(config.mim_patch_size),
                        mask_value=float(config.mim_mask_value),
                        generator=generator,
                        max_grad_norm=config.max_grad_norm,
                    )
                forward_calls += 1
                backward_calls += 1
                optimizer_steps += 1
                loss_value = float(loss.detach().cpu().item())
                if initial_loss is None:
                    initial_loss = loss_value

                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_grad_norm = extra.get("grad_norm") if isinstance(extra, dict) else None
                step_grad_norm_clipped = extra.get("grad_norm_clipped") if isinstance(extra, dict) else None
                step_entry = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "mask_ratio": float(_mask_ratio),
                    "grad_norm": (float(step_grad_norm) if step_grad_norm is not None else None),
                    "grad_norm_clipped": (float(step_grad_norm_clipped) if step_grad_norm_clipped is not None else None),
                    "mim_loss": (
                        float(extra.get("loss_mim"))
                        if isinstance(extra, dict) and extra.get("loss_mim") is not None
                        else None
                    ),
                    "entropy_loss": (
                        float(extra.get("loss_entropy"))
                        if isinstance(extra, dict) and extra.get("loss_entropy") is not None
                        else None
                    ),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                }

                non_finite_fields = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))
                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                mask_ratios.append(float(_mask_ratio))
                step_metrics.append(step_entry)
        elif method == "cotta":
            from .ttt_mim import select_parameters

            params = select_parameters(
                model,
                update_filter=str(config.update_filter),
                include=config.include,
                exclude=config.exclude,
            )
            if not params:
                raise RuntimeError("no parameters selected for TTT")

            optimizer = torch.optim.Adam(params, lr=float(config.lr))
            updated_param_count = _count_unique_params(params)
            base_snapshot = _snapshot_params(params)

            teacher = copy.deepcopy(model)
            teacher.eval()
            for teacher_param in teacher.parameters():
                teacher_param.requires_grad_(False)

            aug_set = tuple(config.cotta_augmentations or ("identity", "hflip"))
            restore_interval = max(1, int(config.cotta_restore_interval))
            restore_prob = float(config.cotta_restore_prob)
            ema_momentum = float(config.cotta_ema_momentum)

            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                if not torch.is_tensor(x):
                    raise RuntimeError("CoTTA requires tensor batches from adapter.build_loader()")

                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []

                optimizer.zero_grad(set_to_none=True)
                student_logits, aug_meta = _forward_cotta_augs(
                    model,
                    x,
                    augmentations=aug_set,
                    aggregation=str(config.cotta_aggregation),
                )
                forward_calls += int(aug_meta.get("branches") or 0)
                loss = _entropy_loss_from_logits(student_logits)
                loss.backward()
                backward_calls += 1
                grad_norm = _global_grad_norm(params)
                grad_norm_clipped = None
                if config.max_grad_norm is not None:
                    clipped = torch.nn.utils.clip_grad_norm_(params, float(config.max_grad_norm))
                    grad_norm_clipped = float(clipped.detach().cpu().item()) if torch.is_tensor(clipped) else float(clipped)
                optimizer.step()
                optimizer_steps += 1

                with torch.no_grad():
                    teacher_logits, teacher_aug_meta = _forward_cotta_augs(
                        teacher,
                        x,
                        augmentations=aug_set,
                        aggregation=str(config.cotta_aggregation),
                    )
                    forward_calls += int(teacher_aug_meta.get("branches") or 0)
                    consistency = float(((student_logits.detach() - teacher_logits.detach()) ** 2).mean().cpu().item())

                _ema_update(teacher, model, momentum=ema_momentum)

                restored_count = 0
                if (step_idx + 1) % restore_interval == 0 and restore_prob > 0.0:
                    restored_count = _stochastic_restore(base_snapshot, prob=restore_prob, generator=generator)

                loss_value = float(loss.detach().cpu().item())
                if initial_loss is None:
                    initial_loss = loss_value

                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_entry = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "grad_norm": float(grad_norm),
                    "grad_norm_clipped": (float(grad_norm_clipped) if grad_norm_clipped is not None else None),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                    "ema_momentum": float(ema_momentum),
                    "restore_prob": float(restore_prob),
                    "restore_interval": int(restore_interval),
                    "restored_count": int(restored_count),
                    "consistency_mse": float(consistency),
                    "aug": aug_meta,
                }

                non_finite_fields = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))
                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                step_metrics.append(step_entry)
        elif method == "eata":
            from .ttt_mim import select_parameters

            params = select_parameters(
                model,
                update_filter=str(config.update_filter),
                include=config.include,
                exclude=config.exclude,
            )
            if not params:
                raise RuntimeError("no parameters selected for TTT")

            optimizer = torch.optim.Adam(params, lr=float(config.lr))
            updated_param_count = _count_unique_params(params)
            base_snapshot = _snapshot_params(params)

            conf_min = float(config.eata_conf_min)
            entropy_min = float(config.eata_entropy_min)
            entropy_max = float(config.eata_entropy_max)
            min_valid_dets = int(config.eata_min_valid_dets)
            anchor_lambda = float(config.eata_anchor_lambda)
            selected_ratio_min = float(config.eata_selected_ratio_min)
            max_skip_streak = max(0, int(config.eata_max_skip_streak))
            skip_streak = 0

            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                if not torch.is_tensor(x):
                    raise RuntimeError("EATA requires tensor batches from adapter.build_loader()")
                if x.ndim == 0:
                    raise RuntimeError("EATA requires batched tensor input")

                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []

                optimizer.zero_grad(set_to_none=True)
                logits = _extract_logits(model(x))
                forward_calls += 1
                if logits.ndim == 0:
                    raise RuntimeError("EATA logits must have batch dimension")

                conf_vec, entropy_vec, valid_vec = _eata_sample_signals(logits, conf_min=conf_min)
                conf_sel = conf_vec >= conf_min
                ent_sel = (entropy_vec >= entropy_min) & (entropy_vec <= entropy_max)
                valid_sel = valid_vec >= float(min_valid_dets)
                selected_mask = conf_sel & ent_sel & valid_sel

                batch_count = int(conf_vec.shape[0]) if conf_vec.ndim >= 1 else 1
                selected_count = int(selected_mask.to(dtype=torch.int32).sum().detach().cpu().item())
                selected_ratio = float(selected_count) / float(max(1, batch_count))

                step_entry: dict[str, Any] = {
                    "step": int(step_idx),
                    "selected_count": int(selected_count),
                    "batch_count": int(batch_count),
                    "selected_ratio": float(selected_ratio),
                    "mean_conf": float(conf_vec.mean().detach().cpu().item()) if conf_vec.numel() else 0.0,
                    "mean_entropy": float(entropy_vec.mean().detach().cpu().item()) if entropy_vec.numel() else 0.0,
                    "mean_valid_dets": float(valid_vec.mean().detach().cpu().item()) if valid_vec.numel() else 0.0,
                    "rolled_back": False,
                }

                if selected_count <= 0:
                    skip_streak += 1
                    step_entry.update(
                        {
                            "skipped": True,
                            "skip_reason": "empty_selected_set",
                            "skip_streak": int(skip_streak),
                            "loss": None,
                            "anchor_loss": 0.0,
                            "adapt_loss": 0.0,
                            "grad_norm": None,
                            "grad_norm_clipped": None,
                            "update_norm": 0.0,
                            "total_update_norm": float(_global_l2_update_norm(base_snapshot)),
                        }
                    )
                    step_metrics.append(step_entry)
                    warnings.append("eata_empty_selected_set")
                    if skip_streak > max_skip_streak:
                        stopped_early = True
                        stop_reason = "eata_max_skip_streak_exceeded"
                        warnings.append(str(stop_reason))
                        step_entry["stop_reason"] = str(stop_reason)
                        break
                    continue

                skip_streak = 0
                if selected_ratio < selected_ratio_min:
                    stopped_early = True
                    stop_reason = "eata_selected_ratio_below_min"

                selected = selected_mask
                while selected.ndim < logits.ndim - 1:
                    selected = selected.unsqueeze(-1)
                selected = selected.expand_as(logits[..., 0])

                probs = torch.softmax(logits, dim=-1)
                entropy_map = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                selected_entropy = entropy_map.masked_select(selected)
                if selected_entropy.numel() == 0:
                    adapt_loss = logits.sum() * 0.0
                else:
                    adapt_loss = selected_entropy.mean()

                anchor_loss = _eata_anchor_loss(base_snapshot)
                loss = adapt_loss + (anchor_lambda * anchor_loss)
                loss.backward()
                backward_calls += 1

                grad_norm = _global_grad_norm(params)
                grad_norm_clipped = None
                if config.max_grad_norm is not None:
                    clipped = torch.nn.utils.clip_grad_norm_(params, float(config.max_grad_norm))
                    grad_norm_clipped = float(clipped.detach().cpu().item()) if torch.is_tensor(clipped) else float(clipped)

                optimizer.step()
                optimizer_steps += 1

                loss_value = float(loss.detach().cpu().item())
                adapt_loss_value = float(adapt_loss.detach().cpu().item())
                anchor_loss_value = float(anchor_loss.detach().cpu().item())
                if initial_loss is None:
                    initial_loss = loss_value

                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)

                step_entry.update(
                    {
                        "skipped": False,
                        "loss": float(loss_value),
                        "adapt_loss": float(adapt_loss_value),
                        "anchor_loss": float(anchor_loss_value),
                        "anchor_lambda": float(anchor_lambda),
                        "grad_norm": float(grad_norm),
                        "grad_norm_clipped": (float(grad_norm_clipped) if grad_norm_clipped is not None else None),
                        "update_norm": float(update_norm),
                        "total_update_norm": float(total_update_norm),
                    }
                )

                non_finite_fields = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))

                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                step_metrics.append(step_entry)
        else:
            from .ttt_mim import select_parameters

            params = select_parameters(
                model,
                update_filter=str(config.update_filter),
                include=config.include,
                exclude=config.exclude,
            )
            if not params:
                raise RuntimeError("no parameters selected for TTT")

            optimizer = torch.optim.Adam(params, lr=float(config.lr))
            updated_param_count = _count_unique_params(params)
            base_snapshot = _snapshot_params(params)
            rho = float(config.sar_rho)
            adaptive = bool(config.sar_adaptive)
            first_step_scale = float(config.sar_first_step_scale)

            for step_idx in range(steps):
                x = batches[step_idx % len(batches)]
                if not torch.is_tensor(x):
                    raise RuntimeError("SAR requires tensor batches from adapter.build_loader()")

                pre_snapshot = _snapshot_params(params)
                pre_buffer_snapshot = _snapshot_norm_buffers(model) if bool(config.rollback_on_stop) else []

                optimizer.zero_grad(set_to_none=True)
                logits_first = _extract_logits(model(x))
                forward_calls += 1
                loss_first = _entropy_loss_from_logits(logits_first)
                loss_first.backward()
                backward_calls += 1

                grad_norm_first = _global_grad_norm(params)
                grad_norm_clipped = None
                if config.max_grad_norm is not None:
                    clipped = torch.nn.utils.clip_grad_norm_(params, float(config.max_grad_norm))
                    grad_norm_clipped = float(clipped.detach().cpu().item()) if torch.is_tensor(clipped) else float(clipped)

                backup, rho_applied = _sar_perturb_and_backup(
                    params,
                    rho=rho,
                    adaptive=adaptive,
                    first_step_scale=first_step_scale,
                )

                optimizer.zero_grad(set_to_none=True)
                logits_second = _extract_logits(model(x))
                forward_calls += 1
                loss_second = _entropy_loss_from_logits(logits_second)
                loss_second.backward()
                backward_calls += 1

                _sar_restore(backup)
                optimizer.step()
                optimizer_steps += 1

                loss_value = float(loss_second.detach().cpu().item())
                if initial_loss is None:
                    initial_loss = loss_value

                update_norm = _global_l2_update_norm(pre_snapshot)
                total_update_norm = _global_l2_update_norm(base_snapshot)
                step_entry = {
                    "step": int(step_idx),
                    "loss": float(loss_value),
                    "loss_first": float(loss_first.detach().cpu().item()),
                    "loss_second": float(loss_second.detach().cpu().item()),
                    "grad_norm": float(grad_norm_first),
                    "grad_norm_clipped": (float(grad_norm_clipped) if grad_norm_clipped is not None else None),
                    "update_norm": float(update_norm),
                    "total_update_norm": float(total_update_norm),
                    "rolled_back": False,
                    "sar_rho": float(rho),
                    "sar_rho_applied": float(rho_applied),
                    "sar_adaptive": bool(adaptive),
                    "sar_first_step_scale": float(first_step_scale),
                }

                non_finite_fields = []
                if bool(config.stop_on_non_finite):
                    for key, value in (
                        ("loss", loss_value),
                        ("grad_norm", step_entry["grad_norm"]),
                        ("grad_norm_clipped", step_entry["grad_norm_clipped"]),
                        ("update_norm", update_norm),
                        ("total_update_norm", total_update_norm),
                    ):
                        if value is None:
                            continue
                        if not math.isfinite(float(value)):
                            non_finite_fields.append(str(key))

                if non_finite_fields:
                    stopped_early = True
                    stop_reason = "non_finite_metrics"
                elif config.max_update_norm is not None and update_norm > float(config.max_update_norm):
                    stopped_early = True
                    stop_reason = "max_update_norm_exceeded"
                elif config.max_total_update_norm is not None and total_update_norm > float(config.max_total_update_norm):
                    stopped_early = True
                    stop_reason = "max_total_update_norm_exceeded"
                elif (
                    config.max_loss_ratio is not None
                    and initial_loss is not None
                    and initial_loss > 0.0
                    and loss_value > (initial_loss * float(config.max_loss_ratio))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_ratio_exceeded"
                elif (
                    config.max_loss_increase is not None
                    and initial_loss is not None
                    and loss_value > (initial_loss + float(config.max_loss_increase))
                ):
                    stopped_early = True
                    stop_reason = "max_loss_increase_exceeded"

                if stopped_early:
                    step_entry["rolled_back"] = bool(config.rollback_on_stop)
                    step_entry["stop_reason"] = str(stop_reason)
                    step_entry["non_finite_fields"] = non_finite_fields or None
                    warnings.append(str(stop_reason))
                    if non_finite_fields:
                        warnings.append(f"non_finite_fields:{','.join(non_finite_fields)}")
                    if bool(config.rollback_on_stop):
                        _restore_params(pre_snapshot)
                        _restore_buffers(pre_buffer_snapshot)
                    step_metrics.append(step_entry)
                    break

                losses.append(float(loss_value))
                step_metrics.append(step_entry)
    finally:
        if not was_training:
            try:
                model.eval()
            except Exception:  # pragma: no cover
                warnings.append("failed_to_restore_eval")

    seconds = float(time.time() - start)

    mask_ratio = None
    if mask_ratios:
        mask_ratio = float(sum(mask_ratios) / float(len(mask_ratios)))

    return TTTReport(
        enabled=True,
        method=method,
        reset=str(config.reset),
        steps_requested=steps,
        steps_run=len(losses),
        batches_used=len(batches),
        seconds=seconds,
        losses=losses,
        mask_ratio=mask_ratio,
        updated_param_count=updated_param_count,
        warnings=warnings,
        stopped_early=bool(stopped_early),
        stop_reason=stop_reason,
        step_metrics=step_metrics,
        method_profile=get_ttt_method_profile(method),
        forward_calls=int(forward_calls),
        backward_calls=int(backward_calls),
        optimizer_steps=int(optimizer_steps),
        update_ratio=float(optimizer_steps) / float(max(1, steps)),
        memory=_memory_finish(memory_start),
    )
