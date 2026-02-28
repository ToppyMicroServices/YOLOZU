"""Backward-compatibility shim — canonical location: ``yolozu.core.image_keys``."""

# Re-export everything so ``from yolozu.image_keys import X`` keeps working.
from yolozu.core.image_keys import *  # noqa: F401,F403
