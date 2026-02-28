"""Backward-compatibility shim — canonical location: ``yolozu.inference.onnxrt_quantize``."""

# Re-export everything so ``from yolozu.onnxrt_quantize import X`` keeps working.
from yolozu.inference.onnxrt_quantize import *  # noqa: F401,F403
