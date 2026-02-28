"""Backward-compatibility shim — canonical location: ``yolozu.predictions.instance_segmentation_predictions``."""

# Re-export everything so ``from yolozu.instance_segmentation_predictions import X`` keeps working.
from yolozu.predictions.instance_segmentation_predictions import *  # noqa: F401,F403
