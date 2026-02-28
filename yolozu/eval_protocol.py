"""Backward-compatibility shim — canonical location: ``yolozu.core.eval_protocol``."""

# Re-export everything so ``from yolozu.eval_protocol import X`` keeps working.
from yolozu.core.eval_protocol import *  # noqa: F401,F403
