"""Backward-compatibility shim — canonical location: ``yolozu.inference.inference_utils``."""

# Re-export everything so ``from yolozu.inference_utils import X`` keeps working.
from yolozu.inference.inference_utils import *  # noqa: F401,F403
