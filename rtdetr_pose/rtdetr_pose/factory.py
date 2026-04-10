"""Config-driven factories for the RTDETRPose reference trainer.

Goal: make it easy to swap backbones/losses without touching the training loop.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .config import LossConfig, ModelConfig
from .losses import Losses
from .model import RTDETRPose
from .models.backbones import build_backbone as build_registered_backbone


def _normalize_activation_name(name: str) -> str:
    value = str(name or "silu").strip().lower().replace("-", "_")
    aliases = {
        "swish": "silu",
        "hard_swish": "hardswish",
        "leaky_relu": "leakyrelu",
    }
    return aliases.get(value, value)


def _resolve_activation_pair(cfg: ModelConfig) -> tuple[str, str]:
    preset = str(getattr(cfg, "activation_preset", "default") or "default").strip().lower()
    preset_map: dict[str, tuple[str, str]] = {
        "default": ("silu", "silu"),
        "all_silu": ("silu", "silu"),
        "all_gelu": ("gelu", "gelu"),
        "all_hardswish": ("hardswish", "hardswish"),
        "all_leakyrelu": ("leakyrelu", "leakyrelu"),
        "head_leakyrelu": ("silu", "leakyrelu"),
        "mobile_hardswish": ("hardswish", "hardswish"),
    }
    if preset not in preset_map:
        raise ValueError(
            "unknown model.activation_preset: "
            f"{preset} (supported: {sorted(preset_map.keys())})"
        )

    back, head = preset_map[preset]

    user_back_raw = str(getattr(cfg, "backbone_activation", "") or "").strip()
    user_head_raw = str(getattr(cfg, "head_activation", "") or "").strip()
    default_back = str(getattr(ModelConfig, "backbone_activation", "silu") or "silu").strip().lower()
    default_head = str(getattr(ModelConfig, "head_activation", "silu") or "silu").strip().lower()

    if user_back_raw and (user_back_raw.strip().lower() != default_back or preset == "default"):
        back = user_back_raw
    if user_head_raw and (user_head_raw.strip().lower() != default_head or preset == "default"):
        head = user_head_raw

    back = _normalize_activation_name(back)
    head = _normalize_activation_name(head)
    valid = {"silu", "gelu", "hardswish", "leakyrelu"}
    if back not in valid:
        raise ValueError(f"unsupported model.backbone_activation: {back}")
    if head not in valid:
        raise ValueError(f"unsupported model.head_activation: {head}")
    return back, head


def build_backbone(cfg: ModelConfig) -> tuple[Any, tuple[int, int, int]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_backbone")

    nested_backbone = getattr(cfg, "backbone", None)
    nested_backbone = nested_backbone if isinstance(nested_backbone, dict) else {}

    name = str(
        nested_backbone.get("name", getattr(cfg, "backbone_name", "cspresnet"))
        or "cspresnet"
    ).lower()

    kwargs = getattr(cfg, "backbone_kwargs", None)
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    nested_args = nested_backbone.get("args", None)
    if isinstance(nested_args, dict):
        kwargs.update(nested_args)

    norm = str(
        nested_backbone.get("norm", getattr(cfg, "backbone_norm", "bn"))
        or "bn"
    ).lower()

    activation_backbone, _ = _resolve_activation_pair(cfg)

    if name in ("cspresnet", "csp_resnet", "tiny_cnn", "tinycnn", "simple_cnn", "simplecnn"):
        if "stage_channels" not in kwargs:
            channels = tuple(int(x) for x in (getattr(cfg, "backbone_channels", None) or [64, 128, 256]))
            if len(channels) != 3:
                raise ValueError("model.backbone_channels must have length 3")
            kwargs["stage_channels"] = (int(channels[0]), int(channels[1]), int(channels[2]))
        if "stage_blocks" not in kwargs:
            kwargs["stage_blocks"] = tuple(int(x) for x in (getattr(cfg, "stage_blocks", None) or [1, 2, 2]))
        if "stem_channels" not in kwargs:
            kwargs["stem_channels"] = int(getattr(cfg, "stem_channels", 32))
        if "use_sppf" not in kwargs:
            kwargs["use_sppf"] = bool(getattr(cfg, "use_sppf", True))

    if name in ("cspdarknet_s",):
        kwargs.pop("stage_channels", None)
        kwargs.pop("stage_blocks", None)
        kwargs.pop("stem_channels", None)

    if name in ("resnet50", "convnext_tiny"):
        kwargs.pop("stage_channels", None)
        kwargs.pop("stage_blocks", None)
        kwargs.pop("stem_channels", None)
        kwargs.pop("use_sppf", None)
    kwargs.setdefault("activation", activation_backbone)
    kwargs.setdefault("norm", norm)

    backbone = build_registered_backbone(name, **kwargs)
    out_channels = tuple(int(c) for c in getattr(backbone, "out_channels", ()))
    if len(out_channels) != 3:
        raise ValueError(f"backbone {name} must define out_channels for [P3,P4,P5]")
    return backbone, out_channels


def build_model(cfg: ModelConfig) -> RTDETRPose:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_model")

    backbone, stage_channels = build_backbone(cfg)
    activation_backbone, activation_head = _resolve_activation_pair(cfg)
    projector_cfg = getattr(cfg, "projector", None)
    projector_cfg = projector_cfg if isinstance(projector_cfg, dict) else {}
    d_model = int(projector_cfg.get("d_model", getattr(cfg, "hidden_dim", 256)) or 256)

    return RTDETRPose(
        # Keep cfg.num_classes as the number of foreground classes and reserve the
        # last logit as "no-object" / background (RT-DETR-style).
        num_classes=int(getattr(cfg, "num_classes", 80)) + 1,
        num_keypoints=int(getattr(cfg, "num_keypoints", 0) or 0),
        enable_mim=bool(getattr(cfg, "enable_mim", False)),
        mim_geom_channels=int(getattr(cfg, "mim_geom_channels", 2) or 2),
        depth_mode=str(getattr(cfg, "depth_mode", "none") or "none"),
        depth_dropout=float(getattr(cfg, "depth_dropout", 0.0) or 0.0),
        hidden_dim=d_model,
        num_queries=int(getattr(cfg, "num_queries", 300)),
        use_uncertainty=bool(getattr(cfg, "use_uncertainty", False)),
        stem_channels=int(getattr(cfg, "stem_channels", 32)),
        backbone_channels=stage_channels,
        stage_blocks=tuple(int(x) for x in (getattr(cfg, "stage_blocks", None) or [1, 2, 2])),
        num_encoder_layers=int(getattr(cfg, "num_encoder_layers", 1)),
        num_decoder_layers=int(getattr(cfg, "num_decoder_layers", 3)),
        nhead=int(getattr(cfg, "nhead", 8)),
        encoder_dim_feedforward=getattr(cfg, "encoder_dim_feedforward", None),
        decoder_dim_feedforward=getattr(cfg, "decoder_dim_feedforward", None),
        backbone=backbone,
        backbone_activation=activation_backbone,
        head_activation=activation_head,
    )


def build_losses(cfg: LossConfig) -> Losses:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for build_losses")

    name = str(getattr(cfg, "name", "default") or "default").lower()
    if name not in ("default", "losses"):
        raise ValueError(f"unknown loss config name: {name}")

    weights = getattr(cfg, "weights", None)
    if not isinstance(weights, dict):
        weights = None
    task_aligner = str(getattr(cfg, "task_aligner", "none") or "none")
    return Losses(weights=weights, task_aligner=task_aligner)
