from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SUPPORTED_TTT_METHODS = ("tent", "mim", "cotta", "eata", "sar")


@dataclass(frozen=True)
class TTTConfig:
    enabled: bool = False

    # "tent" (entropy minimization), "mim" (masked image modeling),
    # "cotta" (EMA-guided entropy minimization with safe restoration), or
    # "eata" (selective entropy adaptation with anti-forgetting regularization), or
    # "sar" (sharpness-aware entropy adaptation)
    method: str = "tent"

    # Reset strategy for applying adaptation across inference.
    # - "stream": adapt once and keep weights for subsequent predictions (default).
    # - "sample": caller is expected to reset weights per-sample (see tools/export_predictions.py).
    reset: str = "stream"

    # Total adaptation steps to run (steps over batches).
    steps: int = 1

    # Adapter-provided loader batch size.
    batch_size: int = 1

    # Optimizer learning rate (Tent or MIM).
    lr: float = 1e-4

    # Safety / guards (optional).
    stop_on_non_finite: bool = True
    rollback_on_stop: bool = True
    max_grad_norm: float | None = None
    max_update_norm: float | None = None
    max_total_update_norm: float | None = None
    max_loss_ratio: float | None = None
    max_loss_increase: float | None = None

    # Parameter selection strategy (see yolozu/tta/ttt_mim.py: select_parameters).
    update_filter: str = "all"  # all | norm_only | adapter_only | lora_only | lora_norm_only
    include: Iterable[str] | None = None
    exclude: Iterable[str] | None = None

    # Cap number of distinct batches used during adaptation.
    max_batches: int = 1

    # Seed for stochastic parts (e.g., MIM masks).
    seed: int | None = None

    # Optional JSON log path (typically used by CLI wrappers).
    log_out: str | None = None

    # MIM-specific options.
    mim_mask_prob: float = 0.6
    mim_patch_size: int = 16
    mim_mask_value: float = 0.0

    # CoTTA-specific options.
    cotta_ema_momentum: float = 0.999
    cotta_augmentations: tuple[str, ...] = ("identity", "hflip")
    cotta_aggregation: str = "confidence_weighted_mean"
    cotta_restore_prob: float = 0.01
    cotta_restore_interval: int = 1

    # EATA-specific options.
    eata_conf_min: float = 0.2
    eata_entropy_min: float = 0.05
    eata_entropy_max: float = 3.0
    eata_min_valid_dets: int = 1
    eata_anchor_lambda: float = 1e-3
    eata_selected_ratio_min: float = 0.0
    eata_max_skip_streak: int = 3

    # SAR-specific options.
    sar_rho: float = 0.05
    sar_adaptive: bool = False
    sar_first_step_scale: float = 1.0

    # Multi-task TTT options (pose / keypoints / depth / seg).
    # When any auxiliary loss weight > 0 and the corresponding output key is
    # present in the model output dict, the loss is included alongside the
    # primary entropy loss.
    aux_pose_weight: float = 0.0
    aux_keypoints_weight: float = 0.0
    aux_depth_weight: float = 0.0
    aux_seg_weight: float = 0.0
    # Self-distillation temperature for auxiliary heads during TTT.
    aux_temperature: float = 1.0
    # SDFT task preset (optional shortcut).
    # Setting this to one of "pose", "keypoints", "depth", "seg", "full" will
    # construct a matching SdftConfig and use it as an aux distillation loss
    # against the model's own frozen-copy outputs (teacher-free SDFT).
    sdft_task: str | None = None
