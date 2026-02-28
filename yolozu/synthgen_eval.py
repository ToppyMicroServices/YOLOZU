"""Backward-compatibility shim — canonical location: ``yolozu.eval.synthgen_eval``."""

# Re-export everything so ``from yolozu.synthgen_eval import X`` keeps working.
from yolozu.eval.synthgen_eval import *  # noqa: F401,F403
