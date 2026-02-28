"""Backward-compatibility shim — canonical location: ``yolozu.eval.instance_segmentation_eval``."""

# Re-export everything so ``from yolozu.instance_segmentation_eval import X`` keeps working.
from yolozu.eval.instance_segmentation_eval import *  # noqa: F401,F403
