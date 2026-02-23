"""Contract helpers for external dataset/model integrations."""

from .synthgen import (
    SYNTHGEN_REQUIRED_FIELDS,
    SynthGenValidationResult,
    normalize_synthgen_sample,
    validate_synthgen_sample,
)

__all__ = [
    "SYNTHGEN_REQUIRED_FIELDS",
    "SynthGenValidationResult",
    "normalize_synthgen_sample",
    "validate_synthgen_sample",
]
