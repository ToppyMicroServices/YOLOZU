"""Backward-compatibility shim — canonical location: ``yolozu.geometry.jitter``."""

# Re-export everything so ``from yolozu.jitter import X`` keeps working.
from yolozu.geometry.jitter import *  # noqa: F401,F403
