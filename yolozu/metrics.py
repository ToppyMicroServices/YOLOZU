"""Backward-compatibility shim — canonical location: ``yolozu.eval.metrics``."""

# Re-export everything so ``from yolozu.metrics import X`` keeps working.
from yolozu.eval.metrics import *  # noqa: F401,F403
