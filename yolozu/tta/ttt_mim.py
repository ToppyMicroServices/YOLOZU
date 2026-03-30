from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class TTTMIMResult:
    losses: list[float]
    mask_ratio: float
    updated_param_count: int


def _ensure_torch():
    if torch is None or F is None:
        raise RuntimeError("torch is required for TTT MIM")


def generate_block_mask(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    mask_prob: float = 0.6,
    generator: "torch.Generator | None" = None,
    device: "torch.device | None" = None,
) -> "torch.Tensor":
    _ensure_torch()
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    grid_h = max(1, int((height + patch_size - 1) / patch_size))
    grid_w = max(1, int((width + patch_size - 1) / patch_size))
    mask = torch.rand((grid_h, grid_w), generator=generator, device=device) < float(mask_prob)
    mask_full = mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return mask_full[:height, :width]


def _expand_mask(mask: "torch.Tensor", ref: "torch.Tensor") -> "torch.Tensor":
    mask_full = mask.to(dtype=torch.bool)
    if ref.ndim == 4:
        if mask_full.ndim == 2:
            mask_full = mask_full.unsqueeze(0).unsqueeze(0)
        elif mask_full.ndim == 3:
            mask_full = mask_full.unsqueeze(1)
        elif mask_full.ndim != 4:
            raise ValueError("mask must be 2D, 3D, or 4D for 4D input")
        if mask_full.shape[-2:] != ref.shape[-2:]:
            raise ValueError("mask spatial dims must match input")
        if mask_full.shape[0] == 1 and ref.shape[0] > 1:
            mask_full = mask_full.expand(ref.shape[0], -1, -1, -1)
        if mask_full.shape[1] == 1 and ref.shape[1] > 1:
            mask_full = mask_full.expand(-1, ref.shape[1], -1, -1)
    elif ref.ndim == 3:
        if mask_full.ndim == 2:
            mask_full = mask_full.unsqueeze(0)
        elif mask_full.ndim != 3:
            raise ValueError("mask must be 2D or 3D for 3D input")
        if mask_full.shape[-2:] != ref.shape[-2:]:
            raise ValueError("mask spatial dims must match input")
        if mask_full.shape[0] == 1 and ref.shape[0] > 1:
            mask_full = mask_full.expand(ref.shape[0], -1, -1)
    else:
        raise ValueError("input must be 3D or 4D")
    if mask_full.shape != ref.shape:
        raise ValueError("mask must broadcast to input shape")
    return mask_full


def apply_mask(x: "torch.Tensor", mask: "torch.Tensor", *, mask_value: float = 0.0) -> "torch.Tensor":
    _ensure_torch()
    mask_full = _expand_mask(mask, x)
    return x.masked_fill(mask_full, float(mask_value))


def reconstruction_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    *,
    mask: "torch.Tensor | None" = None,
    reduction: str = "mean",
) -> "torch.Tensor":
    _ensure_torch()
    if mask is None:
        return F.mse_loss(pred, target, reduction=reduction)
    mask_full = _expand_mask(mask, pred)
    diff = (pred - target) ** 2
    masked = diff.masked_select(mask_full)
    if masked.numel() == 0:
        return diff.sum() * 0.0
    if reduction == "sum":
        return masked.sum()
    if reduction == "none":
        return masked
    return masked.mean()


def filter_parameters(
    named_params: Iterable[tuple[str, "torch.Tensor"]],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    allowed: set[str] | None = None,
) -> list["torch.Tensor"]:
    _ensure_torch()
    include = [s for s in (include or []) if s]
    exclude = [s for s in (exclude or []) if s]
    out = []
    for name, param in named_params:
        if not hasattr(param, "requires_grad") or not param.requires_grad:
            continue
        if allowed is not None and name not in allowed:
            continue
        if include and not any(s in name for s in include):
            continue
        if exclude and any(s in name for s in exclude):
            continue
        out.append(param)
    return out


def _is_norm_module(module: "nn.Module") -> bool:
    if nn is None:
        return False
    return isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def _is_adapter_module(name: str, module: "nn.Module") -> bool:
    name_lower = name.lower()
    class_lower = module.__class__.__name__.lower()
    return any(token in name_lower for token in ("adapter", "lora")) or any(
        token in class_lower for token in ("adapter", "lora")
    )


def _collect_param_names(
    model: "nn.Module", predicate: Callable[[str, "nn.Module"], bool]
) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not predicate(module_name, module):
            continue
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            names.add(full_name)
    return names


def select_parameters(
    model: "nn.Module",
    *,
    update_filter: str = "all",
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list["torch.Tensor"]:
    _ensure_torch()
    update_filter = (update_filter or "all").lower()
    allowed: set[str] | None = None
    if update_filter == "norm_only":
        allowed = _collect_param_names(model, lambda name, module: _is_norm_module(module))
    elif update_filter == "adapter_only":
        allowed = _collect_param_names(model, _is_adapter_module)
    elif update_filter == "lora_only":
        allowed = _collect_param_names(
            model,
            lambda name, module: _is_adapter_module(name, module) and (
                "lora" in str(name).lower() or "lora" in module.__class__.__name__.lower()
            ),
        )
    elif update_filter == "lora_norm_only":
        lora_allowed = _collect_param_names(
            model,
            lambda name, module: _is_adapter_module(name, module) and (
                "lora" in str(name).lower() or "lora" in module.__class__.__name__.lower()
            ),
        )
        norm_allowed = _collect_param_names(model, lambda name, module: _is_norm_module(module))
        allowed = set(lora_allowed) | set(norm_allowed)
    elif update_filter != "all":
        raise ValueError(
            "update_filter must be one of: all, norm_only, adapter_only, lora_only, lora_norm_only"
        )
    return filter_parameters(
        model.named_parameters(),
        include=include,
        exclude=exclude,
        allowed=allowed,
    )


def _count_params(params: Iterable["torch.Tensor"]) -> int:
    seen: set[int] = set()
    total = 0
    for param in params:
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        total += int(param.numel())
    return total


def _default_recon_fn(output):
    if isinstance(output, dict):
        if "recon" in output:
            return output["recon"]
        if "logits" in output:
            return output["logits"]
    if isinstance(output, (list, tuple)) and output:
        return output[0]
    return output


def _default_geom_input(x: "torch.Tensor") -> "torch.Tensor":
    _ensure_torch()
    if x.ndim != 4:
        raise ValueError("structured MIM expects a 4D BCHW tensor")
    # Build a deterministic two-channel geometry surrogate from the input:
    # a full-support mask plus log-normalized luminance. This keeps the
    # compare workflow self-contained for repo-backed smoke fixtures.
    luminance = x.to(dtype=torch.float32).mean(dim=1, keepdim=True)
    support = torch.ones_like(luminance)
    eps = 1e-6
    flat = luminance.flatten(start_dim=2)
    ref = flat.median(dim=-1).values.view(int(x.shape[0]), 1, 1, 1)
    depth_like = torch.log(luminance.clamp_min(eps)) - torch.log(ref.clamp_min(eps))
    return torch.cat([support, depth_like], dim=1)


def supports_structured_mim(model: "nn.Module") -> bool:
    _ensure_torch()
    return bool(
        getattr(model, "enable_mim", False)
        and hasattr(model, "forward")
        and hasattr(model, "named_parameters")
    )


def _global_grad_norm(params: Iterable["torch.Tensor"]) -> float:
    _ensure_torch()
    total = None
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        g2 = g.detach()
        if total is None:
            total = torch.zeros((), device=g2.device, dtype=torch.float32)
        total = total + (g2.to(dtype=torch.float32) ** 2).sum()
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu().item())


def ttt_mim_step(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    x: "torch.Tensor",
    *,
    mask_prob: float = 0.6,
    patch_size: int = 16,
    mask_value: float = 0.0,
    recon_fn: Callable[[object], "torch.Tensor"] | None = None,
    loss_fn: Callable[["torch.Tensor", "torch.Tensor", "torch.Tensor"], "torch.Tensor"] | None = None,
    generator: "torch.Generator | None" = None,
    max_grad_norm: float | None = None,
) -> tuple["torch.Tensor", float, dict[str, float]]:
    _ensure_torch()
    recon_fn = recon_fn or _default_recon_fn
    if loss_fn is None:
        def loss_fn(pred, target, mask):
            return reconstruction_loss(pred, target, mask=mask)

    mask = generate_block_mask(x.shape[-2], x.shape[-1], patch_size=patch_size, mask_prob=mask_prob, generator=generator, device=x.device)
    masked = apply_mask(x, mask, mask_value=mask_value)

    output = model(masked)
    recon = recon_fn(output)
    loss = loss_fn(recon, x, mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group.get("params", [])]
    grad_norm = _global_grad_norm(params)
    grad_norm_clipped = grad_norm
    if max_grad_norm is not None:
        try:
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        except Exception:  # pragma: no cover
            pass
        grad_norm_clipped = _global_grad_norm(params)
    optimizer.step()

    mask_ratio = float(mask.float().mean().item())
    return (
        loss.detach(),
        mask_ratio,
        {"grad_norm": float(grad_norm), "grad_norm_clipped": float(grad_norm_clipped)},
    )


def ttt_structured_mim_step(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    x: "torch.Tensor",
    *,
    mask_prob: float = 0.6,
    patch_size: int = 16,
    mask_value: float = 0.0,
    generator: "torch.Generator | None" = None,
    max_grad_norm: float | None = None,
    entropy_weight: float = 0.1,
    geom_input_fn: Callable[["torch.Tensor"], "torch.Tensor"] | None = None,
) -> tuple["torch.Tensor", float, dict[str, float]]:
    _ensure_torch()
    geom_input_fn = geom_input_fn or _default_geom_input
    mask = generate_block_mask(
        int(x.shape[-2]),
        int(x.shape[-1]),
        patch_size=patch_size,
        mask_prob=mask_prob,
        generator=generator,
        device=x.device,
    )
    geom_input = geom_input_fn(x)
    output = model(x, geom_input=geom_input, feature_mask=mask, return_mim=True)
    if not isinstance(output, dict):
        raise RuntimeError("structured MIM expects model output dict")
    mim = output.get("mim")
    if not isinstance(mim, dict):
        raise RuntimeError("structured MIM expected output['mim']")
    recon_feat = mim.get("recon_feat")
    teacher_feat = mim.get("teacher_feat")
    if recon_feat is None or teacher_feat is None:
        raise RuntimeError("structured MIM expected recon_feat and teacher_feat")
    mask_out = mim.get("mask")
    try:
        from rtdetr_pose.losses import mim_reconstruction_loss as structured_loss_fn
    except ImportError:
        structured_loss_fn = reconstruction_loss
    loss_mim = structured_loss_fn(recon_feat, teacher_feat, mask=mask_out)
    entropy = mim.get("entropy")
    loss_entropy = None
    loss = loss_mim
    if entropy is not None:
        loss_entropy = entropy.to(dtype=loss_mim.dtype)
        loss = loss + (float(entropy_weight) * loss_entropy)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group.get("params", [])]
    grad_norm = _global_grad_norm(params)
    grad_norm_clipped = grad_norm
    if max_grad_norm is not None:
        try:
            torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
        except (RuntimeError, TypeError, ValueError):  # pragma: no cover
            pass
        grad_norm_clipped = _global_grad_norm(params)
    optimizer.step()

    metrics = {
        "grad_norm": float(grad_norm),
        "grad_norm_clipped": float(grad_norm_clipped),
        "loss_mim": float(loss_mim.detach().cpu().item()),
    }
    if loss_entropy is not None:
        metrics["loss_entropy"] = float(loss_entropy.detach().cpu().item())

    mask_tensor = mask_out if mask_out is not None else mask
    mask_ratio = float(mask_tensor.to(dtype=torch.float32).mean().detach().cpu().item())
    return loss.detach(), mask_ratio, metrics


def run_ttt_mim(
    model: "nn.Module",
    x: "torch.Tensor",
    *,
    steps: int = 1,
    lr: float = 1e-4,
    mask_prob: float = 0.6,
    patch_size: int = 16,
    mask_value: float = 0.0,
    update_filter: str = "all",
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    recon_fn: Callable[[object], "torch.Tensor"] | None = None,
    loss_fn: Callable[["torch.Tensor", "torch.Tensor", "torch.Tensor"], "torch.Tensor"] | None = None,
    generator: "torch.Generator | None" = None,
) -> TTTMIMResult:
    _ensure_torch()
    if steps <= 0:
        raise ValueError("steps must be > 0")

    was_training = model.training
    model.train()

    if optimizer is None:
        params = select_parameters(
            model,
            update_filter=update_filter,
            include=include,
            exclude=exclude,
        )
        if not params:
            raise ValueError("no parameters selected for TTT")
        optimizer = torch.optim.Adam(params, lr=float(lr))
    params = [p for group in optimizer.param_groups for p in group.get("params", [])]
    updated_param_count = _count_params(params)

    losses: list[float] = []
    mask_ratio = 0.0
    for _ in range(int(steps)):
        loss, mask_ratio, _metrics = ttt_mim_step(
            model,
            optimizer,
            x,
            mask_prob=mask_prob,
            patch_size=patch_size,
            mask_value=mask_value,
            recon_fn=recon_fn,
            loss_fn=loss_fn,
            generator=generator,
        )
        losses.append(float(loss.cpu().item()))

    if not was_training:
        model.eval()

    return TTTMIMResult(
        losses=losses,
        mask_ratio=mask_ratio,
        updated_param_count=updated_param_count,
    )
