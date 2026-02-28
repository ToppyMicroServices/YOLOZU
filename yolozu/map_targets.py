"""Backward-compatibility shim — canonical location: ``yolozu.training.map_targets``."""

# Re-export everything so ``from yolozu.map_targets import X`` keeps working.
from yolozu.training.map_targets import *  # noqa: F401,F403
