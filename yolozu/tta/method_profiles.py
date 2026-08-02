from __future__ import annotations

from copy import deepcopy
from typing import Any


DETECTOR_LOGITS_SEMANTICS: dict[str, Any] = {
    "class_axis": "last",
    "foreground_selection": "none",
    "query_semantics": (
        "entropy is reduced over every non-class element, including detector queries; "
        "cross-view and cross-step query correspondence is not established"
    ),
    "query_correspondence": "not_established",
    "no_object_semantics": ("included_if_present_in_final_axis_otherwise_unidentified"),
    "scope": "model_output_logits_or_pred_logits",
}


TTT_METHOD_PROFILES: dict[str, dict[str, Any]] = {
    "tent": {
        "profile_id": "yolozu_detector_entropy_v2",
        "runnable": True,
        "maturity": "research",
        "fidelity": "detector_adapted_not_reference_faithful",
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
        "profile_id": "yolozu_structured_mim_v1",
        "runnable": True,
        "runtime_preconditions": ["compatible_checkpoint", "structured_mim_model_hook"],
        "maturity": "research",
        "fidelity": "model_hook_conditional_not_reference_faithful",
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
        "runnable": True,
        "maturity": "research",
        "fidelity": "not_reference_faithful",
        "implementation_class": "research_variant",
        "reference_faithful": False,
        "efficacy": "not_established",
        "loss": {
            "primary": "augmented_detector_logit_entropy",
            "diagnostic": "ema_teacher_consistency_mse",
            "augmented_view_query_correspondence": "not_established",
            "detector_logits": deepcopy(DETECTOR_LOGITS_SEMANTICS),
        },
        "notes": "Phase-1 YOLOZU variant; not a reference-faithful CoTTA implementation.",
    },
    "eata": {
        "profile_id": "yolozu_phase1_variant",
        "runnable": True,
        "maturity": "research",
        "fidelity": "not_reference_faithful",
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
        "runnable": True,
        "maturity": "research",
        "fidelity": "not_reference_faithful",
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


def get_ttt_method_profile(method: str, *, detector_response: bool = False) -> dict[str, Any]:
    """Return an isolated machine-readable profile for a supported TTT method."""

    key = str(method or "").strip().lower()
    try:
        profile = deepcopy(TTT_METHOD_PROFILES[key])
    except KeyError as exc:
        choices = ", ".join(sorted(TTT_METHOD_PROFILES))
        raise ValueError(
            f"unknown TTT method profile {method!r}; expected one of: {choices}"
        ) from exc
    if detector_response:
        if key != "tent":
            raise ValueError("detector response selection is currently supported only for tent")
        profile["profile_id"] = "yolozu_detection_response_v1"
        profile["loss"] = {
            "primary": "selected_foreground_response_consistency",
            "components": ["foreground_class_kl", "bbox_smooth_l1", "foreground_entropy"],
            "detector_logits": {
                "class_axis": "last",
                "foreground_selection": "teacher_confidence_and_no_object_margin",
                "query_semantics": "same-query consistency between original and weak photometric views",
                "query_correspondence": "same_query_index_shape_preserving_view",
                "no_object_semantics": "final_class_excluded_from_foreground_distillation",
                "scope": "model_output_logits_and_bbox",
            },
        }
        profile["abstention"] = {
            "condition": "selected_foreground_queries_below_configured_minimum",
            "effect": "skip_backward_and_optimizer_step_and_restore_norm_buffers_when_no_auxiliary_loss_is_active",
            "report_fields": ["response_abstained", "update_abstained", "buffers_restored_on_abstention", "abstained_steps"],
        }
        profile["notes"] = (
            "YOLOZU detector-native research variant with frozen original-view "
            "responses; efficacy is not established."
        )
    return profile
