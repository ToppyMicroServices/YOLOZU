"""Backward-compatibility shim — canonical location: ``yolozu.geometry.constraints``."""

# Re-export everything so ``from yolozu.constraints import X`` keeps working.
from yolozu.geometry.constraints import *  # noqa: F401,F403
