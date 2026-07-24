"""Prediction I/O, normalisation, transforms, and schema governance.

Backward-compatible re-exports: ``from yolozu.predictions import X`` continues
to work for all symbols previously in the flat ``yolozu.predictions`` module.
"""

# Re-export the original predictions module's public API at package level.
from yolozu.predictions.predictions import *  # noqa: F401,F403
from yolozu.predictions.predictions import __all__ as _PREDICTIONS_ALL
from yolozu.predictions.validation_result import validate_predictions_path

__all__ = [*_PREDICTIONS_ALL, "validate_predictions_path"]
