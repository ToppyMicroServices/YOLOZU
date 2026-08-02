from __future__ import annotations

import argparse
from typing import Any

from .config import SUPPORTED_TTT_METHODS, TTTConfig
from .presets import PRESETS

TTT_METHOD_CHOICES = SUPPORTED_TTT_METHODS
TTT_RESET_CHOICES = ("stream", "sample")
TTT_UPDATE_FILTER_CHOICES = ("all", "norm_only", "adapter_only", "lora_only", "lora_norm_only")
TTT_COTTA_AGGREGATION_CHOICES = ("confidence_weighted_mean", "mean")
TTT_PRESET_CHOICES = tuple(PRESETS.keys())
TTT_SDFT_TASK_CHOICES = ("pose", "keypoints", "depth", "seg", "full")


def add_ttt_arguments(parser: argparse.ArgumentParser, *, include_enable_flag: bool = True) -> None:
    if include_enable_flag:
        parser.add_argument("--ttt", action="store_true", help="Enable test-time training (TTT) before inference.")

    parser.add_argument(
        "--ttt-preset",
        choices=TTT_PRESET_CHOICES,
        default=None,
        help="Recommended TTT preset for method/update/safety defaults.",
    )
    parser.add_argument("--ttt-method", choices=TTT_METHOD_CHOICES, default="tent", help="TTT method (default: tent).")
    parser.add_argument(
        "--ttt-reset",
        choices=TTT_RESET_CHOICES,
        default="stream",
        help="TTT reset policy: stream keeps adapted weights; sample resets per image (default: stream).",
    )
    parser.add_argument("--ttt-steps", type=int, default=1, help="Total TTT steps to run (default: 1).")
    parser.add_argument("--ttt-batch-size", type=int, default=1, help="TTT batch size (default: 1).")
    parser.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate (default: 1e-4).")
    parser.add_argument(
        "--ttt-stop-on-non-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop TTT if loss/grad/update norms become non-finite (default: true).",
    )
    parser.add_argument(
        "--ttt-rollback-on-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback last TTT step when a guard triggers (default: true).",
    )
    parser.add_argument("--ttt-max-grad-norm", type=float, default=None, help="Optional grad clipping norm (default: none).")
    parser.add_argument(
        "--ttt-max-update-norm",
        type=float,
        default=None,
        help="Stop if per-step weight update L2 norm exceeds this (default: none).",
    )
    parser.add_argument(
        "--ttt-max-total-update-norm",
        type=float,
        default=None,
        help="Stop if total drift from initial weights exceeds this (default: none).",
    )
    parser.add_argument(
        "--ttt-max-loss-ratio",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss * ratio) (default: none).",
    )
    parser.add_argument(
        "--ttt-max-loss-increase",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss + delta) (default: none).",
    )
    parser.add_argument(
        "--ttt-update-filter",
        choices=TTT_UPDATE_FILTER_CHOICES,
        default="all",
        help="Which parameters to update during TTT (default: all).",
    )
    parser.add_argument(
        "--ttt-include",
        action="append",
        default=None,
        help="Only update parameters whose name contains this substring (repeatable).",
    )
    parser.add_argument(
        "--ttt-exclude",
        action="append",
        default=None,
        help="Exclude parameters whose name contains this substring (repeatable).",
    )
    parser.add_argument(
        "--ttt-max-batches",
        type=int,
        default=1,
        help="Cap number of distinct batches used for TTT (default: 1).",
    )
    parser.add_argument("--ttt-seed", type=int, default=None, help="Optional RNG seed for TTT.")
    parser.add_argument("--ttt-mask-prob", type=float, default=0.6, help="MIM mask probability (default: 0.6).")
    parser.add_argument("--ttt-patch-size", type=int, default=16, help="MIM patch size (default: 16).")
    parser.add_argument("--ttt-mask-value", type=float, default=0.0, help="MIM mask fill value (default: 0.0).")
    parser.add_argument("--ttt-cotta-ema-momentum", type=float, default=0.999, help="CoTTA EMA momentum (default: 0.999).")
    parser.add_argument(
        "--ttt-cotta-augmentations",
        action="append",
        default=None,
        help="CoTTA augmentation branch name (repeatable, e.g. identity/hflip).",
    )
    parser.add_argument(
        "--ttt-cotta-aggregation",
        choices=TTT_COTTA_AGGREGATION_CHOICES,
        default="confidence_weighted_mean",
        help="CoTTA logits aggregation mode (default: confidence_weighted_mean).",
    )
    parser.add_argument("--ttt-cotta-restore-prob", type=float, default=0.01, help="CoTTA stochastic restore probability (default: 0.01).")
    parser.add_argument("--ttt-cotta-restore-interval", type=int, default=1, help="CoTTA restore cadence in steps (default: 1).")
    parser.add_argument("--ttt-eata-conf-min", type=float, default=0.2, help="EATA minimum confidence threshold (default: 0.2).")
    parser.add_argument("--ttt-eata-entropy-min", type=float, default=0.05, help="EATA minimum entropy threshold (default: 0.05).")
    parser.add_argument("--ttt-eata-entropy-max", type=float, default=3.0, help="EATA maximum entropy threshold (default: 3.0).")
    parser.add_argument("--ttt-eata-min-valid-dets", type=int, default=1, help="EATA minimum valid detections per sample (default: 1).")
    parser.add_argument("--ttt-eata-anchor-lambda", type=float, default=1e-3, help="EATA anchor regularization weight (default: 1e-3).")
    parser.add_argument("--ttt-eata-selected-ratio-min", type=float, default=0.0, help="EATA minimum selected-sample ratio per step (default: 0.0).")
    parser.add_argument("--ttt-eata-max-skip-streak", type=int, default=3, help="EATA max consecutive skipped steps before stop (default: 3).")
    parser.add_argument("--ttt-sar-rho", type=float, default=0.05, help="SAR perturbation radius rho (default: 0.05).")
    parser.add_argument(
        "--ttt-sar-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use adaptive SAR perturbation scaling by parameter magnitude (default: false).",
    )
    parser.add_argument("--ttt-sar-first-step-scale", type=float, default=1.0, help="SAR first-step scaling factor (default: 1.0).")
    parser.add_argument(
        "--ttt-sdft-task",
        choices=TTT_SDFT_TASK_CHOICES,
        default=None,
        help="Task hint for safer preset auto-selection and multi-task auxiliary losses.",
    )
    parser.add_argument(
        "--ttt-aux-pose-weight",
        type=float,
        default=0.0,
        help="Auxiliary consistency loss weight for pose heads (0 disables).",
    )
    parser.add_argument(
        "--ttt-aux-keypoints-weight",
        type=float,
        default=0.0,
        help="Auxiliary consistency loss weight for keypoint heads (0 disables).",
    )
    parser.add_argument(
        "--ttt-aux-depth-weight",
        type=float,
        default=0.0,
        help="Auxiliary consistency loss weight for depth heads (0 disables).",
    )
    parser.add_argument(
        "--ttt-aux-seg-weight",
        type=float,
        default=0.0,
        help="Auxiliary consistency loss weight for segmentation heads (0 disables).",
    )
    parser.add_argument(
        "--ttt-aux-temperature",
        type=float,
        default=1.0,
        help="Auxiliary consistency temperature (default: 1.0).",
    )
    parser.add_argument(
        "--ttt-detector-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use selected foreground class/box response consistency for Tent (default: false).",
    )
    parser.add_argument("--ttt-response-conf-min", type=float, default=0.2, help="Minimum teacher foreground confidence (default: 0.2).")
    parser.add_argument("--ttt-response-topk", type=int, default=20, help="Maximum selected foreground queries per image; 0 is unlimited (default: 20).")
    parser.add_argument("--ttt-response-min-selected", type=int, default=1, help="Abstain from the detector-response update below this selected-query count (default: 1).")
    parser.add_argument("--ttt-response-class-weight", type=float, default=1.0, help="Selected foreground class consistency weight (default: 1.0).")
    parser.add_argument("--ttt-response-bbox-weight", type=float, default=1.0, help="Selected foreground box consistency weight (default: 1.0).")
    parser.add_argument("--ttt-response-entropy-weight", type=float, default=0.05, help="Selected foreground entropy weight (default: 0.05).")
    parser.add_argument("--ttt-log-out", default=None, help="Optional path to write TTT log JSON.")


def _sdft_task_from_args(args: Any) -> str | None:
    raw = getattr(args, "ttt_sdft_task", None)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    return value or None


def build_ttt_config_from_args(args: Any) -> TTTConfig:
    return TTTConfig(
        enabled=True,
        method=str(args.ttt_method),
        reset=str(args.ttt_reset),
        steps=int(args.ttt_steps),
        batch_size=int(args.ttt_batch_size),
        lr=float(args.ttt_lr),
        stop_on_non_finite=bool(args.ttt_stop_on_non_finite),
        rollback_on_stop=bool(args.ttt_rollback_on_stop),
        max_grad_norm=(float(args.ttt_max_grad_norm) if args.ttt_max_grad_norm is not None else None),
        max_update_norm=(float(args.ttt_max_update_norm) if args.ttt_max_update_norm is not None else None),
        max_total_update_norm=(float(args.ttt_max_total_update_norm) if args.ttt_max_total_update_norm is not None else None),
        max_loss_ratio=(float(args.ttt_max_loss_ratio) if args.ttt_max_loss_ratio is not None else None),
        max_loss_increase=(float(args.ttt_max_loss_increase) if args.ttt_max_loss_increase is not None else None),
        update_filter=str(args.ttt_update_filter),
        include=list(args.ttt_include) if args.ttt_include else None,
        exclude=list(args.ttt_exclude) if args.ttt_exclude else None,
        max_batches=int(args.ttt_max_batches),
        seed=args.ttt_seed,
        log_out=args.ttt_log_out,
        mim_mask_prob=float(args.ttt_mask_prob),
        mim_patch_size=int(args.ttt_patch_size),
        mim_mask_value=float(args.ttt_mask_value),
        cotta_ema_momentum=float(args.ttt_cotta_ema_momentum),
        cotta_augmentations=tuple(args.ttt_cotta_augmentations or ["identity", "hflip"]),
        cotta_aggregation=str(args.ttt_cotta_aggregation),
        cotta_restore_prob=float(args.ttt_cotta_restore_prob),
        cotta_restore_interval=int(args.ttt_cotta_restore_interval),
        eata_conf_min=float(args.ttt_eata_conf_min),
        eata_entropy_min=float(args.ttt_eata_entropy_min),
        eata_entropy_max=float(args.ttt_eata_entropy_max),
        eata_min_valid_dets=int(args.ttt_eata_min_valid_dets),
        eata_anchor_lambda=float(args.ttt_eata_anchor_lambda),
        eata_selected_ratio_min=float(args.ttt_eata_selected_ratio_min),
        eata_max_skip_streak=int(args.ttt_eata_max_skip_streak),
        sar_rho=float(args.ttt_sar_rho),
        sar_adaptive=bool(args.ttt_sar_adaptive),
        sar_first_step_scale=float(args.ttt_sar_first_step_scale),
        aux_pose_weight=float(getattr(args, "ttt_aux_pose_weight", 0.0)),
        aux_keypoints_weight=float(getattr(args, "ttt_aux_keypoints_weight", 0.0)),
        aux_depth_weight=float(getattr(args, "ttt_aux_depth_weight", 0.0)),
        aux_seg_weight=float(getattr(args, "ttt_aux_seg_weight", 0.0)),
        aux_temperature=float(getattr(args, "ttt_aux_temperature", 1.0)),
        sdft_task=_sdft_task_from_args(args),
        detector_response=bool(getattr(args, "ttt_detector_response", False)),
        response_conf_min=float(getattr(args, "ttt_response_conf_min", 0.2)),
        response_topk=int(getattr(args, "ttt_response_topk", 20)),
        response_min_selected=int(getattr(args, "ttt_response_min_selected", 1)),
        response_class_weight=float(getattr(args, "ttt_response_class_weight", 1.0)),
        response_bbox_weight=float(getattr(args, "ttt_response_bbox_weight", 1.0)),
        response_entropy_weight=float(getattr(args, "ttt_response_entropy_weight", 0.05)),
    )


def build_ttt_settings_from_args(args: Any) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(args, "ttt", False)),
        "preset": getattr(args, "ttt_preset", None),
        "method": str(getattr(args, "ttt_method", "tent")),
        "reset": str(getattr(args, "ttt_reset", "stream")),
        "steps": int(getattr(args, "ttt_steps", 1)),
        "batch_size": int(getattr(args, "ttt_batch_size", 1)),
        "lr": float(getattr(args, "ttt_lr", 1e-4)),
        "stop_on_non_finite": bool(getattr(args, "ttt_stop_on_non_finite", True)),
        "rollback_on_stop": bool(getattr(args, "ttt_rollback_on_stop", True)),
        "max_grad_norm": (
            float(getattr(args, "ttt_max_grad_norm"))
            if getattr(args, "ttt_max_grad_norm", None) is not None
            else None
        ),
        "max_update_norm": (
            float(getattr(args, "ttt_max_update_norm"))
            if getattr(args, "ttt_max_update_norm", None) is not None
            else None
        ),
        "max_total_update_norm": (
            float(getattr(args, "ttt_max_total_update_norm"))
            if getattr(args, "ttt_max_total_update_norm", None) is not None
            else None
        ),
        "max_loss_ratio": (
            float(getattr(args, "ttt_max_loss_ratio"))
            if getattr(args, "ttt_max_loss_ratio", None) is not None
            else None
        ),
        "max_loss_increase": (
            float(getattr(args, "ttt_max_loss_increase"))
            if getattr(args, "ttt_max_loss_increase", None) is not None
            else None
        ),
        "update_filter": str(getattr(args, "ttt_update_filter", "all")),
        "include": (
            list(getattr(args, "ttt_include"))
            if getattr(args, "ttt_include", None)
            else None
        ),
        "exclude": (
            list(getattr(args, "ttt_exclude"))
            if getattr(args, "ttt_exclude", None)
            else None
        ),
        "max_batches": int(getattr(args, "ttt_max_batches", 1)),
        "seed": getattr(args, "ttt_seed", None),
        "sdft_task": _sdft_task_from_args(args),
        "mim": {
            "mask_prob": float(getattr(args, "ttt_mask_prob", 0.6)),
            "patch_size": int(getattr(args, "ttt_patch_size", 16)),
            "mask_value": float(getattr(args, "ttt_mask_value", 0.0)),
        },
        "cotta": {
            "ema_momentum": float(getattr(args, "ttt_cotta_ema_momentum", 0.999)),
            "augmentations": list(getattr(args, "ttt_cotta_augmentations", None) or ["identity", "hflip"]),
            "aggregation": str(getattr(args, "ttt_cotta_aggregation", "confidence_weighted_mean")),
            "restore_prob": float(getattr(args, "ttt_cotta_restore_prob", 0.01)),
            "restore_interval": int(getattr(args, "ttt_cotta_restore_interval", 1)),
        },
        "eata": {
            "conf_min": float(getattr(args, "ttt_eata_conf_min", 0.2)),
            "entropy_min": float(getattr(args, "ttt_eata_entropy_min", 0.05)),
            "entropy_max": float(getattr(args, "ttt_eata_entropy_max", 3.0)),
            "min_valid_dets": int(getattr(args, "ttt_eata_min_valid_dets", 1)),
            "anchor_lambda": float(getattr(args, "ttt_eata_anchor_lambda", 1e-3)),
            "selected_ratio_min": float(getattr(args, "ttt_eata_selected_ratio_min", 0.0)),
            "max_skip_streak": int(getattr(args, "ttt_eata_max_skip_streak", 3)),
        },
        "sar": {
            "rho": float(getattr(args, "ttt_sar_rho", 0.05)),
            "adaptive": bool(getattr(args, "ttt_sar_adaptive", False)),
            "first_step_scale": float(getattr(args, "ttt_sar_first_step_scale", 1.0)),
        },
        "aux": {
            "pose_weight": float(getattr(args, "ttt_aux_pose_weight", 0.0)),
            "keypoints_weight": float(getattr(args, "ttt_aux_keypoints_weight", 0.0)),
            "depth_weight": float(getattr(args, "ttt_aux_depth_weight", 0.0)),
            "seg_weight": float(getattr(args, "ttt_aux_seg_weight", 0.0)),
            "temperature": float(getattr(args, "ttt_aux_temperature", 1.0)),
        },
        "detector_response": {
            "enabled": bool(getattr(args, "ttt_detector_response", False)),
            "confidence_min": float(getattr(args, "ttt_response_conf_min", 0.2)),
            "topk": int(getattr(args, "ttt_response_topk", 20)),
            "min_selected": int(getattr(args, "ttt_response_min_selected", 1)),
            "class_weight": float(getattr(args, "ttt_response_class_weight", 1.0)),
            "bbox_weight": float(getattr(args, "ttt_response_bbox_weight", 1.0)),
            "entropy_weight": float(getattr(args, "ttt_response_entropy_weight", 0.05)),
        },
    }


def build_ttt_cli_args(args: Any, *, include_enable_flag: bool = True) -> list[str]:
    enabled = bool(getattr(args, "ttt", False))
    if not enabled:
        return []

    cmd: list[str] = []
    if include_enable_flag:
        cmd.append("--ttt")
    if getattr(args, "ttt_preset", None):
        cmd.extend(["--ttt-preset", str(args.ttt_preset)])
    cmd.extend(["--ttt-method", str(args.ttt_method)])
    cmd.extend(["--ttt-reset", str(args.ttt_reset)])
    cmd.extend(["--ttt-steps", str(int(args.ttt_steps))])
    cmd.extend(["--ttt-batch-size", str(int(args.ttt_batch_size))])
    cmd.extend(["--ttt-lr", str(float(args.ttt_lr))])
    cmd.append("--ttt-stop-on-non-finite" if bool(args.ttt_stop_on_non_finite) else "--no-ttt-stop-on-non-finite")
    cmd.append("--ttt-rollback-on-stop" if bool(args.ttt_rollback_on_stop) else "--no-ttt-rollback-on-stop")
    if args.ttt_max_grad_norm is not None:
        cmd.extend(["--ttt-max-grad-norm", str(float(args.ttt_max_grad_norm))])
    if args.ttt_max_update_norm is not None:
        cmd.extend(["--ttt-max-update-norm", str(float(args.ttt_max_update_norm))])
    if args.ttt_max_total_update_norm is not None:
        cmd.extend(["--ttt-max-total-update-norm", str(float(args.ttt_max_total_update_norm))])
    if args.ttt_max_loss_ratio is not None:
        cmd.extend(["--ttt-max-loss-ratio", str(float(args.ttt_max_loss_ratio))])
    if args.ttt_max_loss_increase is not None:
        cmd.extend(["--ttt-max-loss-increase", str(float(args.ttt_max_loss_increase))])
    cmd.extend(["--ttt-update-filter", str(args.ttt_update_filter)])
    if args.ttt_include:
        for inc in args.ttt_include:
            cmd.extend(["--ttt-include", str(inc)])
    if args.ttt_exclude:
        for exc in args.ttt_exclude:
            cmd.extend(["--ttt-exclude", str(exc)])
    cmd.extend(["--ttt-max-batches", str(int(args.ttt_max_batches))])
    if args.ttt_seed is not None:
        cmd.extend(["--ttt-seed", str(int(args.ttt_seed))])
    cmd.extend(["--ttt-mask-prob", str(float(args.ttt_mask_prob))])
    cmd.extend(["--ttt-patch-size", str(int(args.ttt_patch_size))])
    cmd.extend(["--ttt-mask-value", str(float(args.ttt_mask_value))])
    cmd.extend(["--ttt-cotta-ema-momentum", str(float(args.ttt_cotta_ema_momentum))])
    if args.ttt_cotta_augmentations:
        for aug in args.ttt_cotta_augmentations:
            cmd.extend(["--ttt-cotta-augmentations", str(aug)])
    cmd.extend(["--ttt-cotta-aggregation", str(args.ttt_cotta_aggregation)])
    cmd.extend(["--ttt-cotta-restore-prob", str(float(args.ttt_cotta_restore_prob))])
    cmd.extend(["--ttt-cotta-restore-interval", str(int(args.ttt_cotta_restore_interval))])
    cmd.extend(["--ttt-eata-conf-min", str(float(args.ttt_eata_conf_min))])
    cmd.extend(["--ttt-eata-entropy-min", str(float(args.ttt_eata_entropy_min))])
    cmd.extend(["--ttt-eata-entropy-max", str(float(args.ttt_eata_entropy_max))])
    cmd.extend(["--ttt-eata-min-valid-dets", str(int(args.ttt_eata_min_valid_dets))])
    cmd.extend(["--ttt-eata-anchor-lambda", str(float(args.ttt_eata_anchor_lambda))])
    cmd.extend(["--ttt-eata-selected-ratio-min", str(float(args.ttt_eata_selected_ratio_min))])
    cmd.extend(["--ttt-eata-max-skip-streak", str(int(args.ttt_eata_max_skip_streak))])
    cmd.extend(["--ttt-sar-rho", str(float(args.ttt_sar_rho))])
    cmd.append("--ttt-sar-adaptive" if bool(args.ttt_sar_adaptive) else "--no-ttt-sar-adaptive")
    cmd.extend(["--ttt-sar-first-step-scale", str(float(args.ttt_sar_first_step_scale))])
    sdft_task = _sdft_task_from_args(args)
    if sdft_task:
        cmd.extend(["--ttt-sdft-task", sdft_task])
    cmd.extend(["--ttt-aux-pose-weight", str(float(args.ttt_aux_pose_weight))])
    cmd.extend(["--ttt-aux-keypoints-weight", str(float(args.ttt_aux_keypoints_weight))])
    cmd.extend(["--ttt-aux-depth-weight", str(float(args.ttt_aux_depth_weight))])
    cmd.extend(["--ttt-aux-seg-weight", str(float(args.ttt_aux_seg_weight))])
    cmd.extend(["--ttt-aux-temperature", str(float(args.ttt_aux_temperature))])
    cmd.append("--ttt-detector-response" if bool(args.ttt_detector_response) else "--no-ttt-detector-response")
    cmd.extend(["--ttt-response-conf-min", str(float(args.ttt_response_conf_min))])
    cmd.extend(["--ttt-response-topk", str(int(args.ttt_response_topk))])
    cmd.extend(["--ttt-response-min-selected", str(int(args.ttt_response_min_selected))])
    cmd.extend(["--ttt-response-class-weight", str(float(args.ttt_response_class_weight))])
    cmd.extend(["--ttt-response-bbox-weight", str(float(args.ttt_response_bbox_weight))])
    cmd.extend(["--ttt-response-entropy-weight", str(float(args.ttt_response_entropy_weight))])
    if args.ttt_log_out:
        cmd.extend(["--ttt-log-out", str(args.ttt_log_out)])
    return cmd
