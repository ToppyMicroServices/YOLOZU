"""Backward-compatibility shim — canonical location: ``yolozu.core.keypoints``."""

# Re-export everything so ``from yolozu.keypoints import X`` keeps working.
from yolozu.core.keypoints import *  # noqa: F401,F403
