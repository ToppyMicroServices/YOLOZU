"""Backward-compatibility shim — canonical location: ``yolozu.inference.pipeline``."""

# Re-export everything so ``from yolozu.pipeline import X`` keeps working.
from yolozu.inference.pipeline import *  # noqa: F401,F403
