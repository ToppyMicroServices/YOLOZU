"""Backbone override helpers for train_minimal."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from typing import Any

from rtdetr_pose.config import ModelConfig


def _parse_backbone_args_json(raw: str | None) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError("--backbone-args must be a JSON object string") from exc
    if not isinstance(payload, dict):
        raise ValueError("--backbone-args must decode to a JSON object")
    return dict(payload)


def _namespace_to_minimal_model_cfg(args: Namespace) -> ModelConfig:
    return ModelConfig(
        num_classes=int(getattr(args, "num_classes", 80) or 80),
        num_keypoints=int(getattr(args, "num_keypoints", 0) or 0),
        enable_mim=bool(getattr(args, "enable_mim", False)),
        mim_geom_channels=2,
        depth_mode=str(getattr(args, "depth_mode", "none") or "none"),
        depth_dropout=float(getattr(args, "depth_dropout", 0.0) or 0.0),
        hidden_dim=int(getattr(args, "hidden_dim", 256) or 256),
        num_queries=int(getattr(args, "num_queries", 300) or 300),
        use_uncertainty=bool(getattr(args, "use_uncertainty", False)),
    )


def apply_backbone_overrides(
    model_cfg: ModelConfig | None,
    *,
    args: Namespace,
) -> tuple[ModelConfig | None, dict[str, Any] | None]:
    name = str(getattr(args, "backbone_name", "") or "").strip().lower()
    norm = str(getattr(args, "backbone_norm", "") or "").strip().lower()
    extra_args = _parse_backbone_args_json(getattr(args, "backbone_args", None))

    if not name and not norm and not extra_args:
        return model_cfg, None

    if model_cfg is None:
        model_cfg = _namespace_to_minimal_model_cfg(args)

    nested = getattr(model_cfg, "backbone", None)
    nested = dict(nested) if isinstance(nested, dict) else {}
    nested_args = nested.get("args")
    nested_args = dict(nested_args) if isinstance(nested_args, dict) else {}

    if name:
        nested["name"] = str(name)
        model_cfg.backbone_name = str(name)
    if norm:
        nested["norm"] = str(norm)
        model_cfg.backbone_norm = str(norm)

    if extra_args:
        nested_args.update(extra_args)
        nested["args"] = nested_args
        legacy_kwargs = getattr(model_cfg, "backbone_kwargs", None)
        legacy_kwargs = dict(legacy_kwargs) if isinstance(legacy_kwargs, dict) else {}
        legacy_kwargs.update(extra_args)
        model_cfg.backbone_kwargs = legacy_kwargs

    model_cfg.backbone = nested

    summary = {
        "name": (nested.get("name") or getattr(model_cfg, "backbone_name", None)),
        "norm": (nested.get("norm") or getattr(model_cfg, "backbone_norm", None)),
        "args": dict(nested_args),
    }
    return model_cfg, summary


def model_cfg_to_dict(model_cfg: ModelConfig | None) -> dict[str, Any] | None:
    if model_cfg is None:
        return None
    if is_dataclass(model_cfg):
        return asdict(model_cfg)
    return dict(vars(model_cfg))
