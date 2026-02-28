"""Backward-compatibility shim — canonical location: ``yolozu.datasets.splits``."""

# Re-export everything so ``from yolozu.splits import X`` keeps working.
from yolozu.datasets.splits import *  # noqa: F401,F403
