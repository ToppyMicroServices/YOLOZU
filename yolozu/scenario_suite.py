"""Backward-compatibility shim — canonical location: ``yolozu.core.scenario_suite``."""

# Re-export everything so ``from yolozu.scenario_suite import X`` keeps working.
from yolozu.core.scenario_suite import *  # noqa: F401,F403
