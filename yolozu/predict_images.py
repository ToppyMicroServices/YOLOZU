"""Backward-compatibility shim — canonical location: ``yolozu.inference.predict_images``."""

# Re-export everything so ``from yolozu.predict_images import X`` keeps working.
from yolozu.inference.predict_images import *  # noqa: F401,F403
