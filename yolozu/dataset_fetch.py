"""Backward-compatibility shim — canonical location: ``yolozu.datasets.dataset_fetch``."""

# Re-export everything so ``from yolozu.dataset_fetch import X`` keeps working.
from yolozu.datasets.dataset_fetch import *  # noqa: F401,F403
