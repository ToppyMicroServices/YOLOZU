"""Backward-compatibility shim — canonical location: ``yolozu.training.sdft``."""

# Re-export everything so ``from yolozu.sdft import X`` keeps working.
from yolozu.training.sdft import *  # noqa: F401,F403
