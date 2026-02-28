"""Backward-compatibility shim — canonical location: ``yolozu.core.scenarios_cli``."""

# Re-export everything so ``from yolozu.scenarios_cli import X`` keeps working.
from yolozu.core.scenarios_cli import *  # noqa: F401,F403
