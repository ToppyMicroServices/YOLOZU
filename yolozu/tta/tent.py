from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .base import TTARunner
from .ttt_mim import _count_params, select_parameters


@dataclass
class TentConfig:
    lr: float = 1e-4
    include: Iterable[str] | None = None
    exclude: Iterable[str] | None = None
    update_filter: str = "all"
    max_grad_norm: float | None = None

    # Multi-task auxiliary loss weights (0 = disabled).
    aux_pose_weight: float = 0.0
    aux_keypoints_weight: float = 0.0
    aux_depth_weight: float = 0.0
    aux_seg_weight: float = 0.0


def _ensure_torch():
    if torch is None or F is None:
        raise RuntimeError("torch is required for TentRunner")


def _extract_logits(output: Any) -> "torch.Tensor":
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "pred" in output:
            return output["pred"]
    return output


def _extract_outputs(output: Any) -> dict[str, Any]:
    """Extract the full output dict, or wrap scalar tensor."""
    if isinstance(output, dict):
        return output
    return {"logits": output}


def _entropy(logits: "torch.Tensor") -> "torch.Tensor":
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(torch.clamp(probs, min=1e-12))
    return -(probs * logp).sum(dim=-1).mean()


def _aux_consistency_loss(
    outputs: dict[str, Any],
    teacher_outputs: dict[str, Any] | None,
    *,
    pose_weight: float = 0.0,
    keypoints_weight: float = 0.0,
    depth_weight: float = 0.0,
    seg_weight: float = 0.0,
) -> tuple["torch.Tensor | None", dict[str, float]]:
    """Compute auxiliary consistency losses between student & teacher outputs.

    Each loss uses L1/smooth-L1 between the current model's prediction and a
    frozen teacher snapshot (or the detached current output when no teacher is
    provided).  Returns (total_aux_loss, {name: scalar}).
    """
    _ensure_torch()
    parts: dict[str, float] = {}
    total = None

    key_weights: list[tuple[str, float]] = [
        ("rot6d", pose_weight),
        ("log_z", pose_weight * 0.5),
        ("offsets", pose_weight * 0.5),
        ("k_delta", pose_weight * 0.3),
        ("keypoints", keypoints_weight),
        ("depth", depth_weight),
        ("depth_map", depth_weight),
        ("log_z", depth_weight * 0.5 if depth_weight > 0 and pose_weight == 0 else 0.0),
        ("seg_logits", seg_weight),
        ("mask_logits", seg_weight),
    ]

    for key, w in key_weights:
        if w <= 0.0:
            continue
        s_val = outputs.get(key)
        if s_val is None or not isinstance(s_val, torch.Tensor):
            continue
        if teacher_outputs is not None:
            t_val = teacher_outputs.get(key)
            if t_val is None or not isinstance(t_val, torch.Tensor):
                continue
            t_val = t_val.detach()
        else:
            t_val = s_val.detach()

        if s_val.shape != t_val.shape:
            continue

        if key in ("seg_logits", "mask_logits"):
            loss_k = F.binary_cross_entropy_with_logits(s_val, t_val.sigmoid())
        elif key in ("keypoints",):
            loss_k = F.smooth_l1_loss(
                s_val.reshape(s_val.shape[0], -1),
                t_val.reshape(t_val.shape[0], -1),
            )
        else:
            loss_k = F.l1_loss(s_val, t_val)

        loss_k = loss_k * w
        parts[f"aux_{key}"] = float(loss_k.detach().cpu().item())
        total = loss_k if total is None else (total + loss_k)

    return total, parts


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


class TentRunner(TTARunner):
    def __init__(self, model: "nn.Module", *, config: TentConfig | None = None):
        _ensure_torch()
        self.model = model
        self.config = config or TentConfig()
        self.params = select_parameters(
            model,
            update_filter=self.config.update_filter,
            include=self.config.include,
            exclude=self.config.exclude,
        )
        if not self.params:
            raise ValueError("no parameters selected for Tent")
        self.optimizer = torch.optim.Adam(self.params, lr=float(self.config.lr))
        self.updated_param_count = _count_params(self.params)
        self._teacher_outputs: dict[str, Any] | None = None
        self._has_aux = (
            float(self.config.aux_pose_weight) > 0
            or float(self.config.aux_keypoints_weight) > 0
            or float(self.config.aux_depth_weight) > 0
            or float(self.config.aux_seg_weight) > 0
        )

    def reset(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.config.lr)
        self._teacher_outputs = None

    def adapt_step(self, batch: Any) -> dict[str, float]:
        _ensure_torch()
        self.model.train()
        output = self.model(batch)
        outputs = _extract_outputs(output)
        logits = _extract_logits(output)
        loss = _entropy(logits)

        # Snapshot teacher on first step for auxiliary consistency losses.
        if self._has_aux and self._teacher_outputs is None:
            self._teacher_outputs = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }

        aux_metrics: dict[str, float] = {}
        if self._has_aux:
            aux_loss, aux_metrics = _aux_consistency_loss(
                outputs,
                self._teacher_outputs,
                pose_weight=float(self.config.aux_pose_weight),
                keypoints_weight=float(self.config.aux_keypoints_weight),
                depth_weight=float(self.config.aux_depth_weight),
                seg_weight=float(self.config.aux_seg_weight),
            )
            if aux_loss is not None:
                loss = loss + aux_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _global_grad_norm(self.params)
        grad_norm_clipped = grad_norm
        if self.config.max_grad_norm is not None:
            try:
                torch.nn.utils.clip_grad_norm_(self.params, float(self.config.max_grad_norm))
            except Exception:  # pragma: no cover
                pass
            grad_norm_clipped = _global_grad_norm(self.params)
        self.optimizer.step()
        result = {
            "loss_entropy": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "grad_norm_clipped": float(grad_norm_clipped),
        }
        result.update(aux_metrics)
        return result

    def maybe_log(self) -> dict[str, Any] | None:
        return {"updated_param_count": int(self.updated_param_count)}
