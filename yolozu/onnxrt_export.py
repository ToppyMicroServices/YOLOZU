"""Backward-compatibility shim — canonical location: ``yolozu.inference.onnxrt_export``."""

# Re-export everything so ``from yolozu.onnxrt_export import X`` keeps working.
from yolozu.inference.onnxrt_export import *  # noqa: F401,F403
