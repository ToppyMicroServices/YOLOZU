"""Training-time long-tail mitigation recipe builder.

Produces class priors, logit-adjustment biases, class-balanced loss
weights, and per-sample weights from dataset statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = [
    "build_class_priors",
    "build_logit_adjustment_bias",
    "build_class_balanced_weights",
    "build_sample_weights",
    "build_long_tail_recipe",
]

from .long_tail_metrics import class_frequency_counts


def build_class_priors(class_counts: dict[int, int]) -> dict[int, float]:
    total = float(sum(int(v) for v in class_counts.values()))
    if total <= 0:
        return {int(k): 0.0 for k in class_counts}
    return {int(k): float(int(v) / total) for k, v in class_counts.items()}


def build_logit_adjustment_bias(
    class_counts: dict[int, int],
    *,
    tau: float,
    eps: float = 1e-12,
) -> dict[int, float]:
    priors = build_class_priors(class_counts)
    out: dict[int, float] = {}
    for class_id, prior in priors.items():
        out[int(class_id)] = float(float(tau) * math.log(max(float(prior), float(eps))))
    return out


def build_class_balanced_weights(
    class_counts: dict[int, int],
    *,
    beta: float,
    eps: float = 1e-12,
) -> dict[int, float]:
    beta = float(beta)
    if beta < 0.0 or beta >= 1.0:
        raise ValueError("beta must be in [0, 1)")
    weights: dict[int, float] = {}
    for class_id, count in class_counts.items():
        n = max(0, int(count))
        if n <= 0:
            weights[int(class_id)] = 0.0
            continue
        effective_num = 1.0 - math.pow(beta, n)
        weights[int(class_id)] = float((1.0 - beta) / max(float(effective_num), float(eps)))

    vals = [v for v in weights.values() if v > 0]
    if vals:
        mean_val = float(sum(vals) / float(len(vals)))
        if mean_val > 0:
            for class_id in list(weights.keys()):
                if weights[class_id] > 0:
                    weights[class_id] = float(weights[class_id] / mean_val)
    return weights


def build_sample_weights(records: list[dict[str, Any]], class_weights: dict[int, float]) -> list[float]:
    out: list[float] = []
    for record in records:
        labels = list(record.get("labels", []) or [])
        if not labels:
            out.append(1.0)
            continue
        vals: list[float] = []
        for label in labels:
            try:
                class_id = int(label.get("class_id", 0))
            except Exception:
                class_id = 0
            vals.append(float(class_weights.get(class_id, 1.0)))
        out.append(float(sum(vals) / float(max(1, len(vals)))))
    return out


def _sha256_json(doc: dict[str, Any]) -> str:
    payload = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_long_tail_recipe(
    records: list[dict[str, Any]],
    *,
    seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    rebalance_sampler: str,
    loss_plugin: str,
    logit_adjustment_tau: float,
    lort_tau: float,
    class_balanced_beta: float,
    focal_gamma: float,
    ldam_margin: float,
) -> dict[str, Any]:
    if stage1_epochs < 0 or stage2_epochs < 0:
        raise ValueError("stage epochs must be >= 0")

    class_counts = class_frequency_counts(records)
    priors = build_class_priors(class_counts)
    class_balanced = build_class_balanced_weights(class_counts, beta=float(class_balanced_beta))
    sample_weights = build_sample_weights(records, class_balanced)
    logit_bias = build_logit_adjustment_bias(class_counts, tau=float(logit_adjustment_tau))

    plugins = {
        "sampler": {
            "name": str(rebalance_sampler),
            "class_balanced_beta": float(class_balanced_beta),
            "class_weights": {str(k): float(v) for k, v in sorted(class_balanced.items())},
            "sample_weight_summary": {
                "count": int(len(sample_weights)),
                "min": (None if not sample_weights else float(min(sample_weights))),
                "max": (None if not sample_weights else float(max(sample_weights))),
                "mean": (None if not sample_weights else float(sum(sample_weights) / float(len(sample_weights)))),
            },
        },
        "logit_adjustment": {
            "enabled": bool(float(logit_adjustment_tau) > 0.0),
            "tau": float(logit_adjustment_tau),
            "bias": {str(k): float(v) for k, v in sorted(logit_bias.items())},
        },
        "loss": {
            "name": str(loss_plugin),
            "focal_gamma": float(focal_gamma),
            "ldam_margin": float(ldam_margin),
        },
        "lort": {
            "enabled": bool(float(lort_tau) > 0.0),
            "tau": float(lort_tau),
            "note": "frequency-free logits retargeting stage option",
        },
    }

    recipe = {
        "kind": "yolozu_long_tail_recipe",
        "schema_version": 1,
        "seed": int(seed),
        "stages": {
            "stage1_representation": {
                "enabled": bool(stage1_epochs > 0),
                "epochs": int(stage1_epochs),
                "freeze_backbone": False,
                "freeze_classifier": False,
            },
            "stage2_classifier_retrain": {
                "enabled": bool(stage2_epochs > 0),
                "epochs": int(stage2_epochs),
                "freeze_backbone": True,
                "freeze_classifier": False,
                "decoupled": True,
            },
        },
        "plugins": plugins,
        "dataset_distribution": {
            "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
            "class_priors": {str(k): float(v) for k, v in sorted(priors.items())},
            "images": int(len(records)),
            "labels": int(sum(len(record.get("labels", []) or []) for record in records)),
        },
    }
    recipe["recipe_hash"] = _sha256_json(recipe)
    return recipe
