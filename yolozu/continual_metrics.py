"""Backward-compatibility shim — canonical location: ``yolozu.eval.continual_metrics``."""

# Re-export everything so ``from yolozu.continual_metrics import X`` keeps working.
from yolozu.eval.continual_metrics import *  # noqa: F401,F403
