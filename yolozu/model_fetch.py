"""Backward-compatibility shim — canonical location: ``yolozu.inference.model_fetch``."""

# Re-export everything so ``from yolozu.model_fetch import X`` keeps working.
from yolozu.inference.model_fetch import *  # noqa: F401,F403
