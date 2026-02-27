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

# Canonical output keys produced by RTDETRPose / multi-task heads.
POSE_KEYS = ("rot6d", "log_z", "offsets", "k_delta")
KEYPOINTS_KEYS = ("keypoints",)
DEPTH_KEYS = ("depth", "depth_map")
SEG_KEYS = ("seg_logits", "mask_logits")


@dataclass(frozen=True)
class SdftConfig:
    """SDFT-inspired self-distillation loss config.

    This is a lightweight, model-agnostic helper intended for continual
    fine-tuning where a frozen teacher checkpoint regularizes a student.

    Supports multi-task distillation for:
      - Detection: logits (KL) + bbox (L1)
      - Pose: rot6d (geodesic or L1), log_z, offsets, k_delta
      - Keypoints: smooth-L1 on (x, y) coords
      - Depth: scale-invariant L1 on depth maps
      - Segmentation: BCE on mask/seg logits
    """

    weight: float = 1.0
    temperature: float = 1.0
    kl: KlMode = "reverse"
    keys: tuple[str, ...] = ("logits", "bbox")

    logits_weight: float = 1.0
    bbox_weight: float = 1.0
    other_l1_weight: float = 1.0

    # -- Pose --
    rot6d_weight: float = 1.0
    log_z_weight: float = 1.0
    offsets_weight: float = 1.0
    k_delta_weight: float = 1.0

    # -- Keypoints --
    keypoints_weight: float = 1.0

    # -- Depth --
    depth_weight: float = 1.0
    depth_scale_invariant: bool = True

    # -- Segmentation --
    seg_weight: float = 1.0


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
    if cfg.rot6d_weight < 0.0:
        raise ValueError("rot6d_weight must be >= 0")
    if cfg.log_z_weight < 0.0:
        raise ValueError("log_z_weight must be >= 0")
    if cfg.offsets_weight < 0.0:
        raise ValueError("offsets_weight must be >= 0")
    if cfg.k_delta_weight < 0.0:
        raise ValueError("k_delta_weight must be >= 0")
    if cfg.keypoints_weight < 0.0:
        raise ValueError("keypoints_weight must be >= 0")
    if cfg.depth_weight < 0.0:
        raise ValueError("depth_weight must be >= 0")
    if cfg.seg_weight < 0.0:
        raise ValueError("seg_weight must be >= 0")


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


# ---------------------------------------------------------------------------
# Task-specific distillation losses
# ---------------------------------------------------------------------------


def _rot6d_geodesic_loss(
    student: "torch.Tensor", teacher: "torch.Tensor"
) -> "torch.Tensor":
    """Approximate geodesic loss via L2 on 6D rotation representations.

    rot6d is a continuous rotation representation (Zhou et al. 2019).
    L2 in rot6d space correlates with geodesic distance on SO(3) for
    well-conditioned rotations, while being numerically simpler than
    converting to rotation matrices and computing Frobenius norm.
    """
    _require_torch()
    return F.mse_loss(student, teacher)


def _keypoints_loss(
    student: "torch.Tensor", teacher: "torch.Tensor"
) -> "torch.Tensor":
    """Smooth-L1 loss on keypoint coordinates.

    Accepts shapes (B, Q, K, 2) or (B, Q, K*2) — automatically handles both.
    Uses SmoothL1Loss which is less sensitive to outliers than L1 for
    small coordinate deltas, important when keypoint visibility varies.
    """
    _require_torch()
    s = student.reshape(student.shape[0], -1)
    t = teacher.reshape(teacher.shape[0], -1)
    return F.smooth_l1_loss(s, t)


def _depth_loss(
    student: "torch.Tensor",
    teacher: "torch.Tensor",
    *,
    scale_invariant: bool = True,
) -> "torch.Tensor":
    """Scale-invariant depth distillation loss.

    For depth maps / per-query log-depth, scale-invariant loss removes the
    global scale ambiguity that arises from monocular depth estimation.
    Falls back to L1 when scale_invariant=False.
    """
    _require_torch()
    if not scale_invariant:
        return F.l1_loss(student, teacher)

    # Scale-invariant loss (Eigen et al. 2014): L1 of centered differences.
    s_flat = student.reshape(student.shape[0], -1).to(dtype=torch.float32)
    t_flat = teacher.reshape(teacher.shape[0], -1).to(dtype=torch.float32)
    diff = s_flat - t_flat
    mean_diff = diff.mean(dim=-1, keepdim=True)
    return (diff - mean_diff).abs().mean()


def _seg_bce_loss(
    student: "torch.Tensor", teacher: "torch.Tensor"
) -> "torch.Tensor":
    """Binary cross-entropy loss between student and teacher seg logits.

    Both inputs are raw logits; the teacher probabilities are derived with
    sigmoid for a stable BCE formulation.
    """
    _require_torch()
    # Teacher probabilities as soft targets.
    teacher_probs = teacher.detach().sigmoid()
    return F.binary_cross_entropy_with_logits(student, teacher_probs)


def _get_key_weight(cfg: SdftConfig, key: str) -> float:
    """Look up per-key weight from config, falling back to other_l1_weight."""
    weight_map: dict[str, str] = {
        "logits": "logits_weight",
        "bbox": "bbox_weight",
        "rot6d": "rot6d_weight",
        "log_z": "log_z_weight",
        "offsets": "offsets_weight",
        "k_delta": "k_delta_weight",
        "keypoints": "keypoints_weight",
        "depth": "depth_weight",
        "depth_map": "depth_weight",
        "seg_logits": "seg_weight",
        "mask_logits": "seg_weight",
    }
    attr = weight_map.get(key)
    if attr is not None:
        return float(getattr(cfg, attr, cfg.other_l1_weight))
    return float(cfg.other_l1_weight)


def _compute_key_loss(
    key: str,
    s_val: "torch.Tensor",
    t_val: "torch.Tensor",
    cfg: SdftConfig,
) -> "torch.Tensor":
    """Dispatch to the appropriate loss function for a given output key."""
    if key == "logits":
        return kl_divergence_from_logits(
            s_val, t_val, temperature=float(cfg.temperature), mode=cfg.kl,
        )
    elif key == "bbox":
        return F.l1_loss(s_val, t_val)
    elif key == "rot6d":
        return _rot6d_geodesic_loss(s_val, t_val)
    elif key in ("keypoints",):
        return _keypoints_loss(s_val, t_val)
    elif key in ("depth", "depth_map", "log_z"):
        return _depth_loss(s_val, t_val, scale_invariant=bool(cfg.depth_scale_invariant))
    elif key in ("seg_logits", "mask_logits"):
        return _seg_bce_loss(s_val, t_val)
    else:
        return F.l1_loss(s_val, t_val)


def compute_sdft_loss(
    student_outputs: Mapping[str, Any],
    teacher_outputs: Mapping[str, Any],
    cfg: SdftConfig,
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    """Compute multi-task SDFT distillation loss.

    Dispatches to task-specific losses based on output key names:
      * logits  → KL divergence (temperature-scaled)
      * bbox    → L1
      * rot6d   → MSE (geodesic proxy)
      * keypoints → Smooth-L1
      * depth / depth_map / log_z → Scale-invariant L1
      * seg_logits / mask_logits  → BCE with teacher sigmoid targets
      * (other) → L1 fallback

    Each key has a dedicated weight in ``SdftConfig``.
    """
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

        loss_k = _compute_key_loss(key, s_val, t_val, cfg)
        w = _get_key_weight(cfg, key)
        loss_k = loss_k * w
        parts[f"loss_sdft_{key}"] = loss_k

        total = loss_k if total is None else (total + loss_k)

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


# ---------------------------------------------------------------------------
# Convenience constructors for common TTT scenarios
# ---------------------------------------------------------------------------


def make_pose_sdft_config(
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    kl: KlMode = "reverse",
    logits_weight: float = 1.0,
    bbox_weight: float = 1.0,
    rot6d_weight: float = 1.0,
    log_z_weight: float = 0.5,
    offsets_weight: float = 0.5,
    k_delta_weight: float = 0.3,
) -> SdftConfig:
    """Construct an ``SdftConfig`` pre-tuned for 6D pose distillation."""
    return SdftConfig(
        weight=weight,
        temperature=temperature,
        kl=kl,
        keys=("logits", "bbox", "rot6d", "log_z", "offsets", "k_delta"),
        logits_weight=logits_weight,
        bbox_weight=bbox_weight,
        rot6d_weight=rot6d_weight,
        log_z_weight=log_z_weight,
        offsets_weight=offsets_weight,
        k_delta_weight=k_delta_weight,
    )


def make_keypoints_sdft_config(
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    kl: KlMode = "reverse",
    logits_weight: float = 1.0,
    bbox_weight: float = 1.0,
    keypoints_weight: float = 1.0,
) -> SdftConfig:
    """Construct an ``SdftConfig`` pre-tuned for keypoint distillation."""
    return SdftConfig(
        weight=weight,
        temperature=temperature,
        kl=kl,
        keys=("logits", "bbox", "keypoints"),
        logits_weight=logits_weight,
        bbox_weight=bbox_weight,
        keypoints_weight=keypoints_weight,
    )


def make_depth_sdft_config(
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    kl: KlMode = "reverse",
    logits_weight: float = 1.0,
    bbox_weight: float = 1.0,
    depth_weight: float = 1.0,
    depth_scale_invariant: bool = True,
) -> SdftConfig:
    """Construct an ``SdftConfig`` pre-tuned for depth distillation."""
    return SdftConfig(
        weight=weight,
        temperature=temperature,
        kl=kl,
        keys=("logits", "bbox", "depth"),
        logits_weight=logits_weight,
        bbox_weight=bbox_weight,
        depth_weight=depth_weight,
        depth_scale_invariant=depth_scale_invariant,
    )


def make_seg_sdft_config(
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    kl: KlMode = "reverse",
    logits_weight: float = 1.0,
    bbox_weight: float = 1.0,
    seg_weight: float = 1.0,
) -> SdftConfig:
    """Construct an ``SdftConfig`` pre-tuned for segmentation distillation."""
    return SdftConfig(
        weight=weight,
        temperature=temperature,
        kl=kl,
        keys=("logits", "bbox", "seg_logits"),
        logits_weight=logits_weight,
        bbox_weight=bbox_weight,
        seg_weight=seg_weight,
    )


def make_full_sdft_config(
    *,
    weight: float = 1.0,
    temperature: float = 1.0,
    kl: KlMode = "reverse",
    logits_weight: float = 1.0,
    bbox_weight: float = 1.0,
    rot6d_weight: float = 1.0,
    log_z_weight: float = 0.5,
    offsets_weight: float = 0.5,
    k_delta_weight: float = 0.3,
    keypoints_weight: float = 1.0,
    depth_weight: float = 1.0,
    seg_weight: float = 1.0,
    depth_scale_invariant: bool = True,
) -> SdftConfig:
    """Construct an ``SdftConfig`` with all multi-task heads active."""
    return SdftConfig(
        weight=weight,
        temperature=temperature,
        kl=kl,
        keys=(
            "logits", "bbox", "rot6d", "log_z", "offsets", "k_delta",
            "keypoints", "depth", "seg_logits",
        ),
        logits_weight=logits_weight,
        bbox_weight=bbox_weight,
        rot6d_weight=rot6d_weight,
        log_z_weight=log_z_weight,
        offsets_weight=offsets_weight,
        k_delta_weight=k_delta_weight,
        keypoints_weight=keypoints_weight,
        depth_weight=depth_weight,
        seg_weight=seg_weight,
        depth_scale_invariant=depth_scale_invariant,
    )
