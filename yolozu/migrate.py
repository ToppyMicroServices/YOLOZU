"""Backward-compatibility shim — canonical location: ``yolozu.datasets.migrate``."""

# Re-export everything so ``from yolozu.migrate import X`` keeps working.
from yolozu.datasets.migrate import *  # noqa: F401,F403
