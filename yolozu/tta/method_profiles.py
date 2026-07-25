from __future__ import annotations

from copy import deepcopy
from typing import Any


DETECTOR_LOGITS_SEMANTICS: dict[str, Any] = {
    "class_axis": "last",
    "query_semantics": (
        "entropy is reduced over every non-class element, including detector queries"
    ),
    "no_object_semantics": (
        "the generic runner does not remove a detector no-object class; when the "
        "model exposes one in the final class axis, it participates in the loss"
    ),
    "scope": "model_output_logits_or_pred_logits",
}


TTT_METHOD_PROFILES: dict[str, dict[str, Any]] = {
    "tent": {
        "profile_id": "yolozu_detector_entropy_v2",
        "implementation_class": "detector_adapted",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "mean_categorical_entropy",
            "optional": "same_input_snapshot_auxiliary_consistency",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": (
            "Detector-adapted entropy minimization with configurable parameter "
            "selection and safety guards; it is not a reference Tent reproduction."
        ),
    },
    "mim": {
        "profile_id": "yolozu_mim_hook_conditional_v1",
        "implementation_class": "conditional_model_hook",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "structured_mim_when_model_hook_is_available",
            "fallback": "generic_masked_reconstruction",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": (
            "The structured path is conditional on the model exposing YOLOZU's MIM "
            "hook; otherwise the generic masked-reconstruction path is used."
        ),
    },
    "cotta": {
        "profile_id": "yolozu_phase1_variant",
        "implementation_class": "research_variant",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "augmented_detector_logit_entropy",
            "diagnostic": "ema_teacher_consistency_mse",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": "Phase-1 YOLOZU variant; not a reference-faithful CoTTA implementation.",
    },
    "eata": {
        "profile_id": "yolozu_phase1_variant",
        "implementation_class": "research_variant",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "selected_detector_logit_entropy_plus_parameter_anchor",
            "selection_unit": "batch_item_aggregated_over_queries",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": "Phase-1 YOLOZU variant; not a reference-faithful EATA implementation.",
    },
    "sar": {
        "profile_id": "yolozu_phase1_variant",
        "implementation_class": "research_variant",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "two_pass_sharpness_aware_detector_logit_entropy",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": "Phase-1 YOLOZU variant; not a reference-faithful SAR implementation.",
    },
}


def get_ttt_method_profile(method: str) -> dict[str, Any]:
    """Return an isolated machine-readable profile for a supported TTT method."""

    key = str(method or "").strip().lower()
    try:
        return deepcopy(TTT_METHOD_PROFILES[key])
    except KeyError as exc:
        choices = ", ".join(sorted(TTT_METHOD_PROFILES))
        raise ValueError(f"unknown TTT method profile {method!r}; expected one of: {choices}") from exc
