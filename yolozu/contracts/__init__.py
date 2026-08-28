"""Contract helpers for external dataset/model integrations."""

from .synthgen import (
    SYNTHGEN_REQUIRED_FIELDS,
    SynthGenValidationResult,
    normalize_synthgen_sample,
    validate_synthgen_sample,
)
from .ocr import (
    OCRBundleInterface,
    OCRContractError,
    OCRResult,
    map_ocr_runner_result,
    privacy_safe_ocr_summary,
    validate_ocr_bundle_interface,
    validate_ocr_input_media,
    validate_ocr_result,
)

__all__ = [
    "SYNTHGEN_REQUIRED_FIELDS",
    "SynthGenValidationResult",
    "normalize_synthgen_sample",
    "validate_synthgen_sample",
    "OCRBundleInterface",
    "OCRContractError",
    "OCRResult",
    "map_ocr_runner_result",
    "privacy_safe_ocr_summary",
    "validate_ocr_bundle_interface",
    "validate_ocr_input_media",
    "validate_ocr_result",
]
