"""Backward-compatibility shim — canonical location: ``yolozu.geometry.intrinsics``."""

# Re-export everything so ``from yolozu.intrinsics import X`` keeps working.
from yolozu.geometry.intrinsics import *  # noqa: F401,F403
