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
from .artifact_resolver import ArtifactResolver, VerifiedArtifact, VerifiedArtifactSet
from .bundles import (
    AlgorithmBundleRegistry,
    AlgorithmBundleSpec,
    BundleLifecycleProjection,
    BundleLifecycleRecord,
    SupportProfileProjection,
    SupportProfileRecord,
    SupportProfileSpec,
    build_fixed_class_mapping,
    map_fixed_class_outputs,
    map_text_prompt_outputs,
    project_bundle_lifecycle,
    project_support_profiles,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
    validate_bundle_lifecycle_record,
    validate_support_profile_record,
    validate_support_profile_spec,
)
from .control_records import (
    load_bounded_json,
    load_bounded_json_bytes,
    load_bounded_jsonl,
    load_bounded_jsonl_bytes,
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
    "AlgorithmBundleRegistry",
    "AlgorithmBundleSpec",
    "ArtifactResolver",
    "BundleLifecycleProjection",
    "BundleLifecycleRecord",
    "CANONICAL_DECIMAL_V1_PATTERN",
    "DecodedInputInventory",
    "DecodedInputObservation",
    "EnvironmentProfile",
    "ImageJobSpec",
    "QualificationWorkloadProfile",
    "SupportProfileProjection",
    "SupportProfileRecord",
    "SupportProfileSpec",
    "VerifiedArtifact",
    "VerifiedArtifactSet",
    "build_fixed_class_mapping",
    "build_decoded_input_inventory",
    "build_qualification_workload_profile",
    "canonical_decimal_v1",
    "canonical_json_v1",
    "canonical_sha256_v1",
    "compute_environment_fingerprint",
    "compute_workload_fingerprint",
    "load_bounded_json",
    "load_bounded_json_bytes",
    "load_bounded_jsonl",
    "load_bounded_jsonl_bytes",
    "map_fixed_class_outputs",
    "map_text_prompt_outputs",
    "project_bundle_lifecycle",
    "project_support_profiles",
    "validate_algorithm_bundle_registry",
    "validate_algorithm_bundle_spec",
    "validate_bundle_lifecycle_record",
    "validate_environment_profile",
    "validate_image_job_spec",
    "validate_qualification_workload_profile",
    "validate_support_profile_record",
    "validate_support_profile_spec",
]
