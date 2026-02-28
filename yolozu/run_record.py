"""Backward-compatibility shim — canonical location: ``yolozu.core.run_record``."""

# Re-export everything so ``from yolozu.run_record import X`` keeps working.
from yolozu.core.run_record import *  # noqa: F401,F403
