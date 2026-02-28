"""Backward-compatibility shim — canonical location: ``yolozu.core.image_size``."""

# Re-export everything so ``from yolozu.image_size import X`` keeps working.
from yolozu.core.image_size import *  # noqa: F401,F403
