"""Backward-compatibility shim — canonical location: ``yolozu.core.doctor``."""

# Re-export everything so ``from yolozu.doctor import X`` keeps working.
from yolozu.core.doctor import *  # noqa: F401,F403
