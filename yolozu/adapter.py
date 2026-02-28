"""Backward-compatibility shim — canonical location: ``yolozu.inference.adapter``."""

# Re-export everything so ``from yolozu.adapter import X`` keeps working.
from yolozu.inference.adapter import *  # noqa: F401,F403
