"""Backward-compatibility shim — canonical location: ``yolozu.datasets.dataset``."""

# Re-export everything so ``from yolozu.dataset import X`` keeps working.
from yolozu.datasets.dataset import *  # noqa: F401,F403
