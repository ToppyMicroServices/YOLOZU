"""Backward-compatibility shim — canonical location: ``yolozu.datasets.dataset_validator``."""

# Re-export everything so ``from yolozu.dataset_validator import X`` keeps working.
from yolozu.datasets.dataset_validator import *  # noqa: F401,F403
