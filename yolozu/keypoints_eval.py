"""Backward-compatibility shim — canonical location: ``yolozu.eval.keypoints_eval``."""

# Re-export everything so ``from yolozu.keypoints_eval import X`` keeps working.
from yolozu.eval.keypoints_eval import *  # noqa: F401,F403
