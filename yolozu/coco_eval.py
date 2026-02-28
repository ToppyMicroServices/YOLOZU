"""Backward-compatibility shim — canonical location: ``yolozu.eval.coco_eval``."""

# Re-export everything so ``from yolozu.coco_eval import X`` keeps working.
from yolozu.eval.coco_eval import *  # noqa: F401,F403
