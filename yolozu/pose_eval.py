"""Backward-compatibility shim — canonical location: ``yolozu.eval.pose_eval``."""

# Re-export everything so ``from yolozu.pose_eval import X`` keeps working.
from yolozu.eval.pose_eval import *  # noqa: F401,F403
