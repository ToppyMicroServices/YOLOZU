"""Backward-compatibility shim — canonical location: ``yolozu.eval.benchmark``."""

# Re-export everything so ``from yolozu.benchmark import X`` keeps working.
from yolozu.eval.benchmark import *  # noqa: F401,F403
