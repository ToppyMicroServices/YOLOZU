"""Backward-compatibility shim — canonical location: ``yolozu.training.distillation``."""

# Re-export everything so ``from yolozu.distillation import X`` keeps working.
from yolozu.training.distillation import *  # noqa: F401,F403
