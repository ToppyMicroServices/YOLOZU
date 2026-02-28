"""Backward-compatibility shim — canonical location: ``yolozu.eval.long_tail_metrics``."""

# Re-export everything so ``from yolozu.long_tail_metrics import X`` keeps working.
from yolozu.eval.long_tail_metrics import *  # noqa: F401,F403
