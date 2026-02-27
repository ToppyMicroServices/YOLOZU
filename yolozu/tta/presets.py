from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TTTPreset:
    method: str
    steps: int
    batch_size: int
    lr: float
    update_filter: str
    max_batches: int

    # Optional safety defaults (only applied when the corresponding arg is unset).
    max_grad_norm: float | None = None
    max_update_norm: float | None = None
    max_total_update_norm: float | None = None
    max_loss_ratio: float | None = None
    max_loss_increase: float | None = None


PRESETS: dict[str, TTTPreset] = {
    # Conservative defaults intended to be safe without requiring tuning.
    # They are not meant to maximize performance.
    "safe": TTTPreset(
        method="tent",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=1.0,
        max_loss_ratio=3.0,
    ),
    "adapter_only": TTTPreset(
        method="tent",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="adapter_only",
        max_batches=1,
        max_grad_norm=5.0,
        max_update_norm=5.0,
        max_total_update_norm=5.0,
        max_loss_ratio=3.0,
    ),
    "mim_safe": TTTPreset(
        method="mim",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="adapter_only",
        max_batches=1,
        max_grad_norm=5.0,
        max_update_norm=5.0,
        max_total_update_norm=5.0,
        max_loss_ratio=3.0,
    ),
    "cotta_safe": TTTPreset(
        method="cotta",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="lora_norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=1.0,
        max_loss_ratio=3.0,
    ),
    "eata_safe": TTTPreset(
        method="eata",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="lora_norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=1.0,
        max_loss_ratio=3.0,
    ),
    "sar_safe": TTTPreset(
        method="sar",
        steps=1,
        batch_size=1,
        lr=1e-4,
        update_filter="lora_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=1.0,
        max_loss_ratio=3.0,
    ),
    # -- Task-specific presets (pose / keypoints / depth / seg) --
    "pose_safe": TTTPreset(
        method="tent",
        steps=3,
        batch_size=1,
        lr=5e-5,
        update_filter="lora_norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=2.0,
        max_loss_ratio=3.0,
    ),
    "keypoints_safe": TTTPreset(
        method="tent",
        steps=2,
        batch_size=1,
        lr=5e-5,
        update_filter="lora_norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=2.0,
        max_loss_ratio=3.0,
    ),
    "depth_safe": TTTPreset(
        method="tent",
        steps=2,
        batch_size=1,
        lr=5e-5,
        update_filter="norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=2.0,
        max_loss_ratio=3.0,
    ),
    "seg_safe": TTTPreset(
        method="tent",
        steps=2,
        batch_size=1,
        lr=5e-5,
        update_filter="norm_only",
        max_batches=1,
        max_grad_norm=1.0,
        max_update_norm=1.0,
        max_total_update_norm=2.0,
        max_loss_ratio=3.0,
    ),
    "pose_mim": TTTPreset(
        method="mim",
        steps=3,
        batch_size=1,
        lr=5e-5,
        update_filter="adapter_only",
        max_batches=1,
        max_grad_norm=5.0,
        max_update_norm=5.0,
        max_total_update_norm=5.0,
        max_loss_ratio=3.0,
    ),
}


def _choose_default_preset_id(args: Any) -> str:
    method = str(getattr(args, "ttt_method", "tent") or "tent").lower()
    update_filter = str(getattr(args, "ttt_update_filter", "all") or "all").lower()
    sdft_task = str(getattr(args, "ttt_sdft_task", "") or "").lower()

    # Task-specific presets take priority when a task is explicitly requested.
    if sdft_task == "pose" and method in ("tent", "all"):
        return "pose_safe"
    if sdft_task == "keypoints" and method in ("tent", "all"):
        return "keypoints_safe"
    if sdft_task == "depth" and method in ("tent", "all"):
        return "depth_safe"
    if sdft_task == "seg" and method in ("tent", "all"):
        return "seg_safe"
    if sdft_task == "pose" and method == "mim":
        return "pose_mim"

    if method == "sar":
        return "sar_safe"
    if method == "eata":
        return "eata_safe"
    if method == "cotta":
        return "cotta_safe"
    if method == "mim":
        return "mim_safe"
    if update_filter == "adapter_only":
        return "adapter_only"
    return "safe"


def _ttt_core_is_defaultish(args: Any) -> bool:
    try:
        steps = int(getattr(args, "ttt_steps", 1))
        batch_size = int(getattr(args, "ttt_batch_size", 1))
        lr = float(getattr(args, "ttt_lr", 1e-4))
        update_filter = str(getattr(args, "ttt_update_filter", "all"))
        max_batches = int(getattr(args, "ttt_max_batches", 1))
    except Exception:
        return False

    return (
        steps == 1
        and batch_size == 1
        and abs(lr - 1e-4) <= 1e-12
        and update_filter == "all"
        and max_batches == 1
    )


def _fill_ttt_safety_defaults(args: Any, *, preset: TTTPreset) -> None:
    if getattr(args, "ttt_max_grad_norm", None) is None and preset.max_grad_norm is not None:
        args.ttt_max_grad_norm = float(preset.max_grad_norm)
    if getattr(args, "ttt_max_update_norm", None) is None and preset.max_update_norm is not None:
        args.ttt_max_update_norm = float(preset.max_update_norm)
    if getattr(args, "ttt_max_total_update_norm", None) is None and preset.max_total_update_norm is not None:
        args.ttt_max_total_update_norm = float(preset.max_total_update_norm)

    if getattr(args, "ttt_max_loss_ratio", None) is None and getattr(args, "ttt_max_loss_increase", None) is None:
        if preset.max_loss_ratio is not None:
            args.ttt_max_loss_ratio = float(preset.max_loss_ratio)
        elif preset.max_loss_increase is not None:
            args.ttt_max_loss_increase = float(preset.max_loss_increase)


def apply_ttt_preset_args(args: Any) -> None:
    """Apply a preset (or safe defaults) to an argparse Namespace in-place.

    - If `--ttt-preset` is provided, override core knobs and fill safety guards.
    - If `--ttt` is enabled without a preset:
      - When core knobs are left at defaults, auto-apply a conservative preset.
      - Otherwise, keep core knobs but fill safety guards when unset.
    """

    preset_id = getattr(args, "ttt_preset", None)

    if preset_id:
        preset = PRESETS.get(str(preset_id))
        if preset is None:
            raise ValueError(f"unknown preset: {preset_id}")

        args.ttt_method = str(preset.method)
        args.ttt_steps = int(preset.steps)
        args.ttt_batch_size = int(preset.batch_size)
        args.ttt_lr = float(preset.lr)
        args.ttt_update_filter = str(preset.update_filter)
        args.ttt_max_batches = int(preset.max_batches)
        _fill_ttt_safety_defaults(args, preset=preset)
        return

    if not bool(getattr(args, "ttt", False)):
        return

    default_id = _choose_default_preset_id(args)
    preset = PRESETS[default_id]

    if _ttt_core_is_defaultish(args):
        args.ttt_preset = str(default_id)
        args.ttt_method = str(preset.method)
        args.ttt_steps = int(preset.steps)
        args.ttt_batch_size = int(preset.batch_size)
        args.ttt_lr = float(preset.lr)
        args.ttt_update_filter = str(preset.update_filter)
        args.ttt_max_batches = int(preset.max_batches)
        _fill_ttt_safety_defaults(args, preset=preset)
        return

    _fill_ttt_safety_defaults(args, preset=preset)
