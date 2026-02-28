"""Backward-compatibility shim — canonical location: ``yolozu.training.gates``."""

# Re-export everything so ``from yolozu.gates import X`` keeps working.
from yolozu.training.gates import *  # noqa: F401,F403
