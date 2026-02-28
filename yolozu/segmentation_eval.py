"""Backward-compatibility shim — canonical location: ``yolozu.eval.segmentation_eval``."""

# Re-export everything so ``from yolozu.segmentation_eval import X`` keeps working.
from yolozu.eval.segmentation_eval import *  # noqa: F401,F403
