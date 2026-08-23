"""Experimental adaptive-routing interface contracts.

These helpers validate typed records and build privacy-bounded local input
observations. They do not select or execute a model.
"""

from .canonical import (
    CANONICAL_DECIMAL_V1_PATTERN,
    canonical_decimal_v1,
    canonical_json_v1,
    canonical_sha256_v1,
)
from .contracts import (
    EnvironmentProfile,
    ImageJobSpec,
    QualificationWorkloadProfile,
    build_qualification_workload_profile,
    compute_environment_fingerprint,
    compute_workload_fingerprint,
    validate_environment_profile,
    validate_image_job_spec,
    validate_qualification_workload_profile,
)
from .inventory import (
    DecodedInputInventory,
    DecodedInputObservation,
    build_decoded_input_inventory,
)

__all__ = [
    "CANONICAL_DECIMAL_V1_PATTERN",
    "DecodedInputInventory",
    "DecodedInputObservation",
    "EnvironmentProfile",
    "ImageJobSpec",
    "QualificationWorkloadProfile",
    "build_decoded_input_inventory",
    "build_qualification_workload_profile",
    "canonical_decimal_v1",
    "canonical_json_v1",
    "canonical_sha256_v1",
    "compute_environment_fingerprint",
    "compute_workload_fingerprint",
    "validate_environment_profile",
    "validate_image_job_spec",
    "validate_qualification_workload_profile",
]
