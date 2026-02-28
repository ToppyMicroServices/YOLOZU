"""Backward-compatibility shim — canonical location: ``yolozu.eval.metrics_report``."""

# Re-export everything so ``from yolozu.metrics_report import X`` keeps working.
from yolozu.eval.metrics_report import *  # noqa: F401,F403
