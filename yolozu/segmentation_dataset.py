"""Backward-compatibility shim — canonical location: ``yolozu.datasets.segmentation_dataset``."""

# Re-export everything so ``from yolozu.segmentation_dataset import X`` keeps working.
from yolozu.datasets.segmentation_dataset import *  # noqa: F401,F403
