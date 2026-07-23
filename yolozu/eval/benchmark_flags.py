"""Shared benchmark flag semantics for both CLI parser surfaces."""

from __future__ import annotations

ARTIFACT_EVAL_TASKS = frozenset(
    {
        "classification",
        "obb",
        "segmentation",
        "keypoints",
        "depth",
        "pose6d",
    }
)
ARTIFACT_EVAL_INERT_BACKEND_FLAGS = frozenset({"half", "batch", "nms"})
BACKEND_EXECUTION_FLAG_DEFAULTS = {
    "half": False,
    "batch": 1,
    "nms": False,
}

HALF_HELP = (
    "Use FP16 in the torch detect backend. "
    "Must remain disabled when the effective latency source is artifact_eval."
)
BATCH_HELP = (
    "Torch detect backend batch size (default: 1). "
    "Must remain 1 when the effective latency source is artifact_eval."
)
NMS_HELP = (
    "Request NMS in the torch detect backend. "
    "Must remain disabled when the effective latency source is artifact_eval."
)
LATENCY_SOURCE_HELP = (
    "Benchmark source selection. auto uses dataset_pass_wall_time for detect and "
    "artifact_eval for classification, obb, segmentation, keypoints, depth, and pose6d. "
    "Detect rejects explicit artifact_eval before writes because no prepared detection-artifact "
    "evaluation path is implemented. Supported artifact_eval tasks consume prepared artifacts, "
    "so --half, --batch values other than 1, and --nms are rejected. Non-dry-run "
    "artifact-backed tasks cannot use dataset_pass_wall_time; use auto or artifact_eval."
)
OPENVINO_MODEL_HELP = (
    "OpenVINO-lane artifact override. Detect expects a compatible IR; artifact-backed tasks "
    "accept prepared task artifacts without checking or invoking the OpenVINO runtime."
)
PARITY_REFERENCE_HELP = (
    "Reference backend used when writing parity artifacts (default: auto prefers torch, "
    "then first eligible backend). OpenVINO detect requires a supplied IR and runtime; "
    "artifact-backed OpenVINO tasks use prepared artifacts without a runtime check."
)
STRICT_HELP = "Return exit code 2 if any requested format is skipped, fails, or is partial."


__all__ = [
    "ARTIFACT_EVAL_INERT_BACKEND_FLAGS",
    "ARTIFACT_EVAL_TASKS",
    "BACKEND_EXECUTION_FLAG_DEFAULTS",
    "BATCH_HELP",
    "HALF_HELP",
    "LATENCY_SOURCE_HELP",
    "NMS_HELP",
    "OPENVINO_MODEL_HELP",
    "PARITY_REFERENCE_HELP",
    "STRICT_HELP",
]
