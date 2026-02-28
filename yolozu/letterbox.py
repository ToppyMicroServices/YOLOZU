"""Backward-compatibility shim — canonical location: ``yolozu.core.letterbox``."""

# Re-export everything so ``from yolozu.letterbox import X`` keeps working.
from yolozu.core.letterbox import *  # noqa: F401,F403
