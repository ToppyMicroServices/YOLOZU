"""Backward-compatibility shim — canonical location: ``yolozu.predictions.segmentation_predictions``."""

# Re-export everything so ``from yolozu.segmentation_predictions import X`` keeps working.
from yolozu.predictions.segmentation_predictions import *  # noqa: F401,F403
