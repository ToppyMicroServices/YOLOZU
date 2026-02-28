"""Backward-compatibility shim — canonical location: ``yolozu.core.config``."""

# Re-export everything so ``from yolozu.config import X`` keeps working.
from yolozu.core.config import *  # noqa: F401,F403
