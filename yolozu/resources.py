"""Backward-compatibility shim — canonical location: ``yolozu.core.resources``."""

# Re-export everything so ``from yolozu.resources import X`` keeps working.
from yolozu.core.resources import *  # noqa: F401,F403
