"""Backward-compatibility shim — canonical location: ``yolozu.core.boxes``."""

# Re-export everything so ``from yolozu.boxes import X`` keeps working.
from yolozu.core.boxes import *  # noqa: F401,F403
