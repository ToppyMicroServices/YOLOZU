"""Backward-compatibility shim — canonical location: ``yolozu.eval.simple_map``."""

# Re-export everything so ``from yolozu.simple_map import X`` keeps working.
from yolozu.eval.simple_map import *  # noqa: F401,F403
