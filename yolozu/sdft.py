from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Literal

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

KlMode = Literal["forward", "reverse", "sym"]


@dataclass(frozen=True)
class SdftConfig:
    """SDFT-inspired self-distillation loss config.

    This is a lightweight, model-agnostic helper intended for continual
    fine-tuning where a frozen teacher checkpoint regularizes a student.
    """

    weight: float = 1.0
    temperature: float = 1.0
    kl: KlMode = "reverse"
    keys: tuple[str, ...] = ("logits", "bbox")

    logits_weight: float = 1.0
    bbox_weight: float = 1.0
    other_l1_weight: float = 1.0


def _require_torch() -> None:
    if torch is None or F is None:  # pragma: no cover
        raise RuntimeError("torch is required for yolozu.sdft")


def _validate_config(cfg: SdftConfig) -> None:
    if cfg.weight < 0.0:
        raise ValueError("weight must be >= 0")
    if cfg.temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if cfg.logits_weight < 0.0:
        raise ValueError("logits_weight must be >= 0")
    if cfg.bbox_weight < 0.0:
        raise ValueError("bbox_weight must be >= 0")
    if cfg.other_l1_weight < 0.0:
        raise ValueError("other_l1_weight must be >= 0")


def kl_divergence_from_logits(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    *,
    temperature: float = 1.0,
    mode: KlMode = "reverse",
) -> "torch.Tensor":
    """KL divergence between categorical distributions parameterized by logits.

    mode:
      - forward:  KL(teacher || student)  (classic distillation)
      - reverse:  KL(student || teacher)  (SDFT-style objective)
      - sym:      0.5 * (forward + reverse)
    """

    _require_torch()
    t = float(temperature)
    if not (t > 0.0):
        raise ValueError("temperature must be > 0")

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logits shape mismatch: student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )

    s_logits = student_logits / t
    t_logits = teacher_logits / t

    log_p_s = F.log_softmax(s_logits, dim=-1)
    log_p_t = F.log_softmax(t_logits, dim=-1)
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()

    log_p_s = log_p_s.reshape(-1, log_p_s.shape[-1])
    log_p_t = log_p_t.reshape(-1, log_p_t.shape[-1])
    p_s = p_s.reshape(-1, p_s.shape[-1])
    p_t = p_t.reshape(-1, p_t.shape[-1])

    if mode == "forward":
        # KL(teacher || student)
        loss = F.kl_div(log_p_s, p_t, reduction="batchmean")
    elif mode == "reverse":
        # KL(student || teacher)
        loss = F.kl_div(log_p_t, p_s, reduction="batchmean")
    elif mode == "sym":
        loss_f = F.kl_div(log_p_s, p_t, reduction="batchmean")
        loss_r = F.kl_div(log_p_t, p_s, reduction="batchmean")
        loss = 0.5 * (loss_f + loss_r)
    else:
        raise ValueError(f"unknown kl mode: {mode}")

    # Preserve gradient magnitudes as in temperature-scaled distillation.
    return loss * (t * t)


def compute_sdft_loss(
    student_outputs: Mapping[str, Any],
    teacher_outputs: Mapping[str, Any],
    cfg: SdftConfig,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    _require_torch()
    _validate_config(cfg)
    if not isinstance(student_outputs, Mapping) or not isinstance(teacher_outputs, Mapping):
        raise TypeError("student_outputs and teacher_outputs must be Mapping")

    reference = None
    for value in student_outputs.values():
        if isinstance(value, torch.Tensor):
            reference = value
            break
    if reference is None:
        for value in teacher_outputs.values():
            if isinstance(value, torch.Tensor):
                reference = value
                break

    total = None
    parts: dict[str, torch.Tensor] = {}

    for key in cfg.keys:
        if key not in student_outputs or key not in teacher_outputs:
            continue
        s_val = student_outputs[key]
        t_val = teacher_outputs[key]
        if not isinstance(s_val, torch.Tensor) or not isinstance(t_val, torch.Tensor):
            continue

        if s_val.shape != t_val.shape:
            raise ValueError(
                f"sdft shape mismatch for '{key}': student={tuple(s_val.shape)} teacher={tuple(t_val.shape)}"
            )

        t_val = t_val.detach()
        if t_val.device != s_val.device:
            t_val = t_val.to(device=s_val.device)

        if key == "logits":
            loss_k = kl_divergence_from_logits(
                s_val,
                t_val,
                temperature=float(cfg.temperature),
                mode=cfg.kl,
            )
            loss_k = loss_k * float(cfg.logits_weight)
            parts["loss_sdft_logits"] = loss_k
        elif key == "bbox":
            loss_k = F.l1_loss(s_val, t_val)
            loss_k = loss_k * float(cfg.bbox_weight)
            parts["loss_sdft_bbox"] = loss_k
        else:
            loss_k = F.l1_loss(s_val, t_val)
            loss_k = loss_k * float(cfg.other_l1_weight)
            parts[f"loss_sdft_{key}"] = loss_k

        total = loss_k if total is None else (total + loss_k)

    if total is None:
        if reference is not None:
            total = torch.zeros((), device=reference.device, dtype=reference.dtype)
        else:  # pragma: no cover
            total = torch.tensor(0.0)
    if total is None:
        if reference is not None:
            total = torch.zeros((), device=reference.device, dtype=reference.dtype)
        else:  # pragma: no cover
            total = torch.tensor(0.0)
    total = total * float(cfg.weight)
    if float(cfg.weight) != 1.0:
        for name in list(parts.keys()):
            parts[name] = parts[name] * float(cfg.weight)
    parts["loss_sdft"] = total
    return total, parts
