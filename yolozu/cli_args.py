"""Backward-compatibility shim — canonical location: ``yolozu.core.cli_args``."""

# Re-export everything so ``from yolozu.cli_args import X`` keeps working.
from yolozu.core.cli_args import *  # noqa: F401,F403
