"""Backward-compatibility shim — canonical location: ``yolozu.core.canonical``."""

# Re-export everything so ``from yolozu.canonical import X`` keeps working.
from yolozu.core.canonical import *  # noqa: F401,F403
