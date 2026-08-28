"""Immutable bundle and append-only lifecycle interface contracts."""

from __future__ import annotations

import copy
import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .canonical import canonical_decimal_v1, canonical_json_v1, canonical_sha256_v1

__all__ = [
    "AlgorithmBundleRegistry",
    "AlgorithmBundleSpec",
    "BundleLifecycleProjection",
    "BundleLifecycleRecord",
    "CODE_OWNED_RUNNER_IDS",
    "SupportProfileProjection",
    "SupportProfileRecord",
    "SupportProfileSpec",
    "build_fixed_class_mapping",
    "map_fixed_class_outputs",
    "map_text_prompt_outputs",
    "project_bundle_lifecycle",
    "project_support_profiles",
    "validate_algorithm_bundle_registry",
    "validate_algorithm_bundle_spec",
    "validate_bundle_lifecycle_record",
    "validate_support_profile_record",
    "validate_support_profile_spec",
    "validate_support_profile_snapshot",
]


ZERO_DIGEST = "0" * 64
EMPTY_PROFILE_SET_DIGEST = canonical_sha256_v1([])
MAX_BUNDLE_SPEC_BYTES = 4 * 1024 * 1024
MAX_REGISTRY_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES = 17_179_869_184
MAX_ARTIFACT_SET_BYTES = 68_719_476_736

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_BUNDLE_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,63}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_ROLE_ID_RE = re.compile(r"(?:repo_maintainer|release_reviewer|site_operator|automation)\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|\s)(?:/(?:Users|home|private|tmp|var|etc)/|[A-Za-z]:\\)"
)
_IP_TOKEN_RE = re.compile(r"(?<![0-9A-Fa-f:.])(?:[0-9A-Fa-f:.]{3,45})(?![0-9A-Fa-f:.])")

TASKS = frozenset({"object_detection", "instance_segmentation"})
PROMPT_MODES = frozenset({"fixed_classes", "text"})
PROVENANCE_CLASSES = frozenset({"existing_code_owned", "screened_candidate"})
EXECUTION_TRUST_CLASSES = frozenset({"code_owned_audited", "third_party_isolated"})
EXECUTION_BINDING_STATES = frozenset({"bound", "unbound"})
EXECUTION_ARTIFACT_SCOPES = frozenset(
    {"runner_consumed", "fetchable_model_assets"}
)
EXECUTION_UNAVAILABLE_REASONS = frozenset(
    {
        "adaptive_runner_unavailable",
        "runner_artifact_set_incomplete",
        "runtime_unqualified",
    }
)
ARTIFACT_ROLES = frozenset(
    {
        "code_archive",
        "weight",
        "config",
        "class_vocabulary",
        "tokenizer",
        "engine",
        "auxiliary",
    }
)
# Reserved metadata identifiers for audited code-owned factories. Membership
# validates a spec identifier; it does not claim that an adapter is available.
CODE_OWNED_RUNNER_IDS = frozenset(
    {"yolo_runtime", "onnxruntime", "tensorrt", "torchvision", "coreml"}
)
LOADER_FORMATS = frozenset(
    {
        "safetensors",
        "onnx",
        "tensorrt_engine",
        "torchscript",
        "pytorch_pickle",
        "python_archive",
        "native_plugin",
    }
)
UNSAFE_LOADER_FORMATS = frozenset(
    {"torchscript", "pytorch_pickle", "python_archive", "native_plugin"}
)
PRECISIONS = frozenset({"fp32", "tf32", "fp16", "bf16", "int8"})
CHANNELS = frozenset({"Candidate", "Experimental", "Stable"})
TRUST_DOMAINS = frozenset(
    {"yolozu_managed", "site_managed", "operator_asserted", "unknown"}
)
LICENSE_REVIEW_STATES = frozenset({"approved", "unknown", "blocked"})


def _copy(record: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(record))


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected object")
    return dict(value)


def _list(value: Any, *, field: str, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected array")
    if len(value) < minimum or len(value) > maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum} items")
    return list(value)


def _keys(
    value: Mapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field}: unknown keys: {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{field}: missing required keys: {', '.join(missing)}")


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum}")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field}: expected boolean")
    return value


def _enum(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field}: unsupported value")
    return value


def _token(
    value: Any,
    *,
    field: str,
    pattern: re.Pattern[str] = _ID_RE,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field}: invalid identifier")
    return value


def _sha256(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")
    if not allow_zero and value == ZERO_DIGEST:
        raise ValueError(f"{field}: zero sentinel is not valid here")
    return value


def _safe_text(value: Any, *, field: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty string")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field}: exceeds {maximum_bytes} UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field}: control characters are invalid")
    return value


def _public_text(value: Any, *, field: str, maximum_bytes: int) -> str:
    text = _safe_text(value, field=field, maximum_bytes=maximum_bytes)
    if any(unicodedata.category(character).startswith("C") for character in text):
        raise ValueError(f"{field}: private control/format characters are invalid")
    if _UUID_RE.search(text) or _EMAIL_RE.search(text) or _ABSOLUTE_PATH_RE.search(text):
        raise ValueError(f"{field}: private identifiers and paths are invalid")
    for match in _IP_TOKEN_RE.finditer(text):
        token = match.group(0).strip(".:")
        try:
            ipaddress.ip_address(token)
        except ValueError:
            continue
        raise ValueError(f"{field}: IP addresses are invalid")
    return text


def _utc(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected exact RFC3339 UTC second")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{field}: invalid UTC calendar time") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{field}: non-canonical UTC time")
    return value


def _unique_tokens(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str],
    minimum: int = 1,
    maximum: int,
) -> list[str]:
    items = _list(value, field=field, minimum=minimum, maximum=maximum)
    out = [_enum(item, field=f"{field}[]", allowed=allowed) for item in items]
    if len(out) != len(set(out)):
        raise ValueError(f"{field}: duplicate values")
    return out


def _validate_cache_key(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value.encode("ascii", "ignore")) != len(value):
        raise ValueError(f"{field}: expected ASCII cache key")
    if len(value.encode("ascii")) > 512:
        raise ValueError(f"{field}: cache key exceeds 512 bytes")
    parts = value.split("/")
    if len(parts) < 1 or len(parts) > 8:
        raise ValueError(f"{field}: cache key requires 1..8 components")
    for part in parts:
        if part in {"", ".", ".."} or _COMPONENT_RE.fullmatch(part) is None:
            raise ValueError(f"{field}: invalid cache-key component")
    return value


def _validate_artifact(value: Any, *, expected_order: int) -> dict[str, Any]:
    record = _mapping(value, field=f"artifacts[{expected_order}]")
    allowed = frozenset(
        {
            "artifact_id",
            "order",
            "role",
            "source_id",
            "source_revision",
            "expected_size_bytes",
            "sha256",
            "license_expression",
            "license_source",
            "license_revision",
            "cache_key",
        }
    )
    _keys(record, field="artifact", allowed=allowed, required=allowed)
    out = {
        "artifact_id": _token(
            record["artifact_id"], field="artifact.artifact_id", pattern=_COMPONENT_RE
        ),
        "order": _integer(
            record["order"], field="artifact.order", minimum=0, maximum=31
        ),
        "role": _enum(record["role"], field="artifact.role", allowed=ARTIFACT_ROLES),
        "source_id": _safe_text(
            record["source_id"], field="artifact.source_id", maximum_bytes=512
        ),
        "source_revision": _safe_text(
            record["source_revision"],
            field="artifact.source_revision",
            maximum_bytes=256,
        ),
        "expected_size_bytes": _integer(
            record["expected_size_bytes"],
            field="artifact.expected_size_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
        "sha256": _sha256(record["sha256"], field="artifact.sha256"),
        "license_expression": _safe_text(
            record["license_expression"],
            field="artifact.license_expression",
            maximum_bytes=256,
        ),
        "license_source": _safe_text(
            record["license_source"],
            field="artifact.license_source",
            maximum_bytes=512,
        ),
        "license_revision": _safe_text(
            record["license_revision"],
            field="artifact.license_revision",
            maximum_bytes=256,
        ),
        "cache_key": _validate_cache_key(
            record["cache_key"], field="artifact.cache_key"
        ),
    }
    if out["order"] != expected_order:
        raise ValueError("artifacts: order must be contiguous and match array order")
    return out


def _validate_identity(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    allowed = frozenset({"id", "version", "digest"})
    _keys(record, field=field, allowed=allowed, required=allowed)
    return {
        "id": _token(record["id"], field=f"{field}.id"),
        "version": _token(record["version"], field=f"{field}.version"),
        "digest": _sha256(record["digest"], field=f"{field}.digest"),
    }


def _validate_runner_options(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="runner_options")
    allowed = frozenset(
        {"device_policy", "intra_op_threads", "inter_op_threads", "optimization_level"}
    )
    _keys(record, field="runner_options", allowed=allowed, required=frozenset())
    out: dict[str, Any] = {}
    if "device_policy" in record:
        out["device_policy"] = _enum(
            record["device_policy"],
            field="runner_options.device_policy",
            allowed=frozenset({"job_controlled"}),
        )
    for key in ("intra_op_threads", "inter_op_threads"):
        if key in record:
            out[key] = _integer(
                record[key], field=f"runner_options.{key}", minimum=1, maximum=1024
            )
    if "optimization_level" in record:
        out["optimization_level"] = _enum(
            record["optimization_level"],
            field="runner_options.optimization_level",
            allowed=frozenset({"disabled", "basic", "extended", "all"}),
        )
    return out


def _validate_execution_binding(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="execution_binding")
    allowed = frozenset({"status", "artifact_scope", "reason_code"})
    _keys(record, field="execution_binding", allowed=allowed, required=allowed)
    status = _enum(
        record["status"],
        field="execution_binding.status",
        allowed=EXECUTION_BINDING_STATES,
    )
    scope = _enum(
        record["artifact_scope"],
        field="execution_binding.artifact_scope",
        allowed=EXECUTION_ARTIFACT_SCOPES,
    )
    reason = record["reason_code"]
    if status == "bound":
        if scope != "runner_consumed" or reason is not None:
            raise ValueError(
                "bound execution requires runner_consumed artifacts and null reason"
            )
    else:
        if scope != "fetchable_model_assets":
            raise ValueError(
                "unbound execution permits only fetchable_model_assets"
            )
        reason = _enum(
            reason,
            field="execution_binding.reason_code",
            allowed=EXECUTION_UNAVAILABLE_REASONS,
        )
    return {"status": status, "artifact_scope": scope, "reason_code": reason}


def _validate_runtime(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="runtime")
    allowed = frozenset(
        {
            "runtime_id",
            "runtime_version",
            "provider_id",
            "provider_version",
            "precision",
            "architecture",
            "accelerator_requirement",
            "minimum_accelerator_memory_bytes",
        }
    )
    required = allowed - {"minimum_accelerator_memory_bytes"}
    _keys(record, field="runtime", allowed=allowed, required=required)
    requirement = _enum(
        record["accelerator_requirement"],
        field="runtime.accelerator_requirement",
        allowed=frozenset({"none", "optional", "required"}),
    )
    out: dict[str, Any] = {
        "runtime_id": _token(record["runtime_id"], field="runtime.runtime_id"),
        "runtime_version": _token(
            record["runtime_version"], field="runtime.runtime_version"
        ),
        "provider_id": _token(record["provider_id"], field="runtime.provider_id"),
        "provider_version": _token(
            record["provider_version"], field="runtime.provider_version"
        ),
        "precision": _enum(
            record["precision"], field="runtime.precision", allowed=PRECISIONS
        ),
        "architecture": _token(
            record["architecture"], field="runtime.architecture"
        ),
        "accelerator_requirement": requirement,
    }
    if requirement == "none":
        if "minimum_accelerator_memory_bytes" in record:
            raise ValueError("runtime: CPU-only requirement forbids accelerator memory")
    else:
        if "minimum_accelerator_memory_bytes" not in record:
            raise ValueError("runtime: accelerator memory requirement is missing")
        out["minimum_accelerator_memory_bytes"] = _integer(
            record["minimum_accelerator_memory_bytes"],
            field="runtime.minimum_accelerator_memory_bytes",
            minimum=1,
            maximum=MAX_ARTIFACT_SET_BYTES,
        )
    return out


def _validate_input_shapes(value: Any) -> list[dict[str, Any]]:
    entries = _list(value, field="model_input_shapes", minimum=1, maximum=8)
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(entries):
        record = _mapping(item, field=f"model_input_shapes[{index}]")
        allowed = frozenset({"name", "layout", "dimensions"})
        _keys(record, field="model_input_shape", allowed=allowed, required=allowed)
        name = _token(record["name"], field="model_input_shape.name")
        if name in names:
            raise ValueError("model_input_shapes: duplicate name")
        names.add(name)
        dimensions = [
            _integer(
                dimension,
                field="model_input_shape.dimensions[]",
                minimum=1,
                maximum=65536,
            )
            for dimension in _list(
                record["dimensions"],
                field="model_input_shape.dimensions",
                minimum=1,
                maximum=8,
            )
        ]
        out.append(
            {
                "name": name,
                "layout": _enum(
                    record["layout"],
                    field="model_input_shape.layout",
                    allowed=frozenset({"NCHW", "NHWC", "CHW", "HWC"}),
                ),
                "dimensions": dimensions,
            }
        )
    return out


def _normalize_labels(value: Any) -> list[str]:
    entries = _list(value, field="class_vocabulary.labels", minimum=1, maximum=10_000)
    labels: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, str):
            raise ValueError(f"class_vocabulary.labels[{index}]: expected string")
        label = unicodedata.normalize("NFKC", item).strip()
        if not label or len(label) > 256:
            raise ValueError("class_vocabulary.labels: expected 1..256 code points")
        if any(ord(character) < 32 or ord(character) == 127 for character in label):
            raise ValueError("class_vocabulary.labels: control character")
        if label != item:
            raise ValueError("class_vocabulary.labels: value is not normalized")
        if label in seen:
            raise ValueError("class_vocabulary.labels: duplicate normalized label")
        seen.add(label)
        labels.append(label)
    return labels


@dataclass(frozen=True)
class AlgorithmBundleSpec:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def spec_digest(self) -> str:
        return str(self._record["spec_digest"])

    @property
    def artifact_set_digest(self) -> str:
        return str(self._record["artifact_set_digest"])


@dataclass(frozen=True)
class AlgorithmBundleRegistry:
    _record: dict[str, Any]
    bundles: tuple[AlgorithmBundleSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def registry_digest(self) -> str:
        return str(self._record["registry_digest"])

    def by_spec_digest(self) -> dict[str, AlgorithmBundleSpec]:
        return {bundle.spec_digest: bundle for bundle in self.bundles}


def validate_algorithm_bundle_spec(value: Mapping[str, Any]) -> AlgorithmBundleSpec:
    """Validate one immutable AlgorithmBundleSpec v1."""

    record = _mapping(value, field="AlgorithmBundleSpec")
    allowed = frozenset(
        {
            "schema_version",
            "family_id",
            "bundle_id",
            "bundle_version",
            "spec_digest",
            "provenance_class",
            "screening_binding",
            "test_only",
            "tasks",
            "prompt_modes",
            "adapter_backend_id",
            "execution_binding",
            "runner_id",
            "runner_version",
            "execution_trust_class",
            "execution_isolation_policy_digest",
            "loader_format",
            "unsafe_deserialization_required",
            "model_source_id",
            "model_revision",
            "runtime",
            "model_input_shapes",
            "decoder",
            "preprocess",
            "postprocess",
            "execution_network_required",
            "runner_options",
            "artifacts",
            "artifact_set_digest",
            "class_vocabulary",
            "text_prompt_support",
        }
    )
    execution_fields = frozenset(
        {
            "runner_id",
            "runner_version",
            "execution_trust_class",
            "execution_isolation_policy_digest",
            "loader_format",
            "unsafe_deserialization_required",
            "runtime",
            "model_input_shapes",
            "decoder",
            "preprocess",
            "postprocess",
            "execution_network_required",
            "runner_options",
        }
    )
    required = allowed - execution_fields - {
        "screening_binding",
        "class_vocabulary",
        "text_prompt_support",
    }
    _keys(record, field="AlgorithmBundleSpec", allowed=allowed, required=required)
    if record.get("schema_version") != 1 or isinstance(record.get("schema_version"), bool):
        raise ValueError("AlgorithmBundleSpec.schema_version: expected 1")

    provenance = _enum(
        record["provenance_class"],
        field="provenance_class",
        allowed=PROVENANCE_CLASSES,
    )
    screening: dict[str, Any] | None = None
    if provenance == "screened_candidate":
        if "screening_binding" not in record:
            raise ValueError("screened candidate requires screening_binding")
        raw = _mapping(record["screening_binding"], field="screening_binding")
        keys = frozenset(
            {"stream_key", "pass_record_id", "pass_record_digest", "source_revision"}
        )
        _keys(raw, field="screening_binding", allowed=keys, required=keys)
        screening = {
            "stream_key": _token(raw["stream_key"], field="screening_binding.stream_key"),
            "pass_record_id": _token(
                raw["pass_record_id"], field="screening_binding.pass_record_id"
            ),
            "pass_record_digest": _sha256(
                raw["pass_record_digest"], field="screening_binding.pass_record_digest"
            ),
            "source_revision": _safe_text(
                raw["source_revision"],
                field="screening_binding.source_revision",
                maximum_bytes=256,
            ),
        }
    elif "screening_binding" in record:
        raise ValueError("existing_code_owned forbids screening_binding")

    prompt_modes = _unique_tokens(
        record["prompt_modes"],
        field="prompt_modes",
        allowed=PROMPT_MODES,
        maximum=2,
    )
    artifacts = [
        _validate_artifact(item, expected_order=index)
        for index, item in enumerate(
            _list(record["artifacts"], field="artifacts", minimum=1, maximum=32)
        )
    ]
    artifact_ids = [item["artifact_id"] for item in artifacts]
    cache_keys = [item["cache_key"] for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("artifacts: duplicate artifact_id")
    if len(cache_keys) != len(set(cache_keys)):
        raise ValueError("artifacts: duplicate cache_key")
    if sum(item["expected_size_bytes"] for item in artifacts) > MAX_ARTIFACT_SET_BYTES:
        raise ValueError("artifacts: expected total exceeds 64 GiB")
    expected_artifact_digest = canonical_sha256_v1(artifacts)
    artifact_set_digest = _sha256(
        record["artifact_set_digest"], field="artifact_set_digest"
    )
    if artifact_set_digest != expected_artifact_digest:
        raise ValueError("artifact_set_digest: digest mismatch")

    execution_binding = _validate_execution_binding(record["execution_binding"])
    bound = execution_binding["status"] == "bound"
    present_execution_fields = execution_fields.intersection(record)
    if bound:
        required_execution_fields = execution_fields - {
            "execution_isolation_policy_digest"
        }
        missing = sorted(required_execution_fields - set(record))
        if missing:
            raise ValueError(
                "bound execution is missing fields: " + ", ".join(missing)
            )
    elif present_execution_fields:
        raise ValueError(
            "unbound execution forbids runner fields: "
            + ", ".join(sorted(present_execution_fields))
        )

    output: dict[str, Any] = {
        "schema_version": 1,
        "family_id": _token(
            record["family_id"], field="family_id", pattern=_COMPONENT_RE
        ),
        "bundle_id": _token(
            record["bundle_id"], field="bundle_id", pattern=_COMPONENT_RE
        ),
        "bundle_version": _token(
            record["bundle_version"],
            field="bundle_version",
            pattern=_BUNDLE_VERSION_RE,
        ),
        "spec_digest": _sha256(record["spec_digest"], field="spec_digest"),
        "provenance_class": provenance,
        "test_only": _boolean(record["test_only"], field="test_only"),
        "tasks": _unique_tokens(
            record["tasks"], field="tasks", allowed=TASKS, maximum=2
        ),
        "prompt_modes": prompt_modes,
        "adapter_backend_id": _token(
            record["adapter_backend_id"], field="adapter_backend_id"
        ),
        "execution_binding": execution_binding,
        "model_source_id": _safe_text(
            record["model_source_id"], field="model_source_id", maximum_bytes=512
        ),
        "model_revision": _safe_text(
            record["model_revision"], field="model_revision", maximum_bytes=256
        ),
        "artifacts": artifacts,
        "artifact_set_digest": artifact_set_digest,
    }
    if bound:
        trust_class = _enum(
            record["execution_trust_class"],
            field="execution_trust_class",
            allowed=EXECUTION_TRUST_CLASSES,
        )
        loader_format = _enum(
            record["loader_format"], field="loader_format", allowed=LOADER_FORMATS
        )
        unsafe = _boolean(
            record["unsafe_deserialization_required"],
            field="unsafe_deserialization_required",
        )
        needs_isolation = (
            unsafe
            or loader_format in UNSAFE_LOADER_FORMATS
            or any(item["role"] == "code_archive" for item in artifacts)
        )
        isolation_digest: str | None = None
        if trust_class == "code_owned_audited":
            if needs_isolation:
                raise ValueError(
                    "bundle-supplied code/unsafe loader requires third_party_isolated"
                )
            if "execution_isolation_policy_digest" in record:
                raise ValueError("code_owned_audited forbids isolation-policy digest")
        else:
            if "execution_isolation_policy_digest" not in record:
                raise ValueError(
                    "third_party_isolated requires isolation-policy digest"
                )
            isolation_digest = _sha256(
                record["execution_isolation_policy_digest"],
                field="execution_isolation_policy_digest",
            )
        if (loader_format in UNSAFE_LOADER_FORMATS) != unsafe:
            raise ValueError(
                "loader_format and unsafe_deserialization_required disagree"
            )
        output.update(
            {
                "runner_id": _enum(
                    record["runner_id"],
                    field="runner_id",
                    allowed=CODE_OWNED_RUNNER_IDS,
                ),
                "runner_version": _token(
                    record["runner_version"], field="runner_version"
                ),
                "execution_trust_class": trust_class,
                "loader_format": loader_format,
                "unsafe_deserialization_required": unsafe,
                "runtime": _validate_runtime(record["runtime"]),
                "model_input_shapes": _validate_input_shapes(
                    record["model_input_shapes"]
                ),
                "decoder": _validate_identity(record["decoder"], field="decoder"),
                "preprocess": _validate_identity(
                    record["preprocess"], field="preprocess"
                ),
                "postprocess": _validate_identity(
                    record["postprocess"], field="postprocess"
                ),
                "execution_network_required": _boolean(
                    record["execution_network_required"],
                    field="execution_network_required",
                ),
                "runner_options": _validate_runner_options(record["runner_options"]),
            }
        )
        if isolation_digest is not None:
            output["execution_isolation_policy_digest"] = isolation_digest
    if screening is not None:
        output["screening_binding"] = screening

    if "fixed_classes" in prompt_modes:
        if "class_vocabulary" not in record:
            raise ValueError("fixed_classes requires inline class_vocabulary")
        vocabulary = _mapping(record["class_vocabulary"], field="class_vocabulary")
        vocabulary_keys = frozenset({"id", "digest", "labels"})
        _keys(
            vocabulary,
            field="class_vocabulary",
            allowed=vocabulary_keys,
            required=vocabulary_keys,
        )
        labels = _normalize_labels(vocabulary["labels"])
        vocabulary_id = _token(vocabulary["id"], field="class_vocabulary.id")
        digest = _sha256(vocabulary["digest"], field="class_vocabulary.digest")
        if digest != canonical_sha256_v1({"id": vocabulary_id, "labels": labels}):
            raise ValueError("class_vocabulary.digest: digest mismatch")
        output["class_vocabulary"] = {
            "id": vocabulary_id,
            "digest": digest,
            "labels": labels,
        }
    elif "class_vocabulary" in record:
        raise ValueError("class_vocabulary is valid only for fixed_classes")

    if "text" in prompt_modes:
        if "text_prompt_support" not in record:
            raise ValueError("text prompt mode requires text_prompt_support")
        text_support = _mapping(record["text_prompt_support"], field="text_prompt_support")
        text_keys = frozenset({"mode", "output_label_semantics"})
        _keys(text_support, field="text_prompt_support", allowed=text_keys, required=text_keys)
        if text_support != {
            "mode": "dynamic_text",
            "output_label_semantics": "request_prompt_index_v1",
        }:
            raise ValueError("text_prompt_support: unsupported semantics")
        output["text_prompt_support"] = text_support
    elif "text_prompt_support" in record:
        raise ValueError("text_prompt_support is valid only for text prompt mode")

    if len(canonical_json_v1(output)) > MAX_BUNDLE_SPEC_BYTES:
        raise ValueError("AlgorithmBundleSpec exceeds 4 MiB")
    expected_spec_digest = canonical_sha256_v1(output, own_digest_field="spec_digest")
    if output["spec_digest"] != expected_spec_digest:
        raise ValueError("spec_digest: digest mismatch")
    return AlgorithmBundleSpec(output)


def validate_algorithm_bundle_registry(
    value: Mapping[str, Any],
) -> AlgorithmBundleRegistry:
    """Validate one immutable AlgorithmBundleRegistry v1."""

    record = _mapping(value, field="AlgorithmBundleRegistry")
    allowed = frozenset({"schema_version", "registry_id", "bundles", "registry_digest"})
    _keys(record, field="AlgorithmBundleRegistry", allowed=allowed, required=allowed)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("AlgorithmBundleRegistry.schema_version: expected 1")
    bundles = tuple(
        validate_algorithm_bundle_spec(item)
        for item in _list(record["bundles"], field="bundles", minimum=0, maximum=128)
    )
    spec_digests = [bundle.spec_digest for bundle in bundles]
    identities = [
        (
            bundle.to_dict()["family_id"],
            bundle.to_dict()["bundle_id"],
            bundle.to_dict()["bundle_version"],
        )
        for bundle in bundles
    ]
    if len(spec_digests) != len(set(spec_digests)):
        raise ValueError("AlgorithmBundleRegistry: duplicate spec_digest")
    if len(identities) != len(set(identities)):
        raise ValueError("AlgorithmBundleRegistry: duplicate versioned bundle identity")
    normalized = {
        "schema_version": 1,
        "registry_id": _enum(
            record["registry_id"],
            field="registry_id",
            allowed=frozenset({"yolozu-bundle-registry-v1"}),
        ),
        "bundles": [bundle.to_dict() for bundle in bundles],
        "registry_digest": _sha256(
            record["registry_digest"], field="registry_digest", allow_zero=True
        ),
    }
    expected = canonical_sha256_v1(normalized, own_digest_field="registry_digest")
    if normalized["registry_digest"] != expected:
        raise ValueError("registry_digest: digest mismatch")
    if len(canonical_json_v1(normalized)) > MAX_REGISTRY_BYTES:
        raise ValueError("AlgorithmBundleRegistry exceeds 128 MiB")
    return AlgorithmBundleRegistry(normalized, bundles)


def build_fixed_class_mapping(
    bundle: AlgorithmBundleSpec, requested_labels: Sequence[str]
) -> dict[str, Any]:
    """Build the exact request/bundle fixed-class mapping without guessing."""

    record = bundle.to_dict()
    vocabulary = record.get("class_vocabulary")
    if not isinstance(vocabulary, dict) or "fixed_classes" not in record["prompt_modes"]:
        raise ValueError("bundle does not support fixed_classes")
    normalized = _normalize_labels(list(requested_labels))
    index = {label: position for position, label in enumerate(vocabulary["labels"])}
    if any(label not in index for label in normalized):
        raise ValueError("requested fixed class is not in the immutable vocabulary")
    request_to_bundle = [index[label] for label in normalized]
    bundle_to_request = [
        {"bundle_class_index": bundle_index, "request_index": request_index}
        for request_index, bundle_index in enumerate(request_to_bundle)
    ]
    bundle_to_request.sort(key=lambda item: item["bundle_class_index"])
    return {
        "vocabulary_id": vocabulary["id"],
        "vocabulary_digest": vocabulary["digest"],
        "bundle_class_count": len(vocabulary["labels"]),
        "requested_labels": normalized,
        "request_to_bundle_class_index": request_to_bundle,
        "retained_bundle_to_request": bundle_to_request,
    }


def map_fixed_class_outputs(
    mapping: Mapping[str, Any], native_class_indices: Sequence[Any]
) -> list[dict[str, Any]]:
    """Filter unrequested native classes and map retained integral indices."""

    retained = {
        entry["bundle_class_index"]: entry["request_index"]
        for entry in mapping["retained_bundle_to_request"]
    }
    labels = list(mapping["requested_labels"])
    bundle_class_count = _integer(
        mapping["bundle_class_count"],
        field="fixed_class_mapping.bundle_class_count",
        minimum=1,
        maximum=10_000,
    )
    output: list[dict[str, Any]] = []
    for value in native_class_indices:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("native class index must be a nonnegative integer")
        if value >= bundle_class_count:
            raise ValueError("native class index exceeds the immutable vocabulary")
        if value in retained:
            request_index = retained[value]
            if request_index < 0 or request_index >= len(labels):
                raise ValueError("fixed-class mapping is invalid")
            output.append(
                {"request_index": request_index, "label": labels[request_index]}
            )
    return output


def map_text_prompt_outputs(
    requested_prompts: Sequence[str], request_prompt_indices: Sequence[Any]
) -> list[dict[str, Any]]:
    """Map only integral request-prompt indices to caller-supplied phrases."""

    prompts = _normalize_labels(list(requested_prompts))
    output: list[dict[str, Any]] = []
    for value in request_prompt_indices:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= len(prompts)
        ):
            raise ValueError("request prompt index is out of range")
        output.append({"request_index": value, "label": prompts[value]})
    return output


def _validate_quality_requirement(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset(
        {
            "metric_id",
            "direction",
            "threshold",
            "evaluation_dataset_id",
            "evaluation_dataset_sha256",
            "evaluation_protocol_sha256",
            "evaluation_vocabulary_id",
        }
    )
    _keys(record, field=field, allowed=keys, required=keys)
    return {
        "metric_id": _token(record["metric_id"], field=f"{field}.metric_id"),
        "direction": _enum(
            record["direction"],
            field=f"{field}.direction",
            allowed=frozenset({"higher_is_better", "lower_is_better"}),
        ),
        "threshold": canonical_decimal_v1(record["threshold"], field=f"{field}.threshold"),
        "evaluation_dataset_id": _token(
            record["evaluation_dataset_id"], field=f"{field}.evaluation_dataset_id"
        ),
        "evaluation_dataset_sha256": _sha256(
            record["evaluation_dataset_sha256"],
            field=f"{field}.evaluation_dataset_sha256",
        ),
        "evaluation_protocol_sha256": _sha256(
            record["evaluation_protocol_sha256"],
            field=f"{field}.evaluation_protocol_sha256",
        ),
        "evaluation_vocabulary_id": _token(
            record["evaluation_vocabulary_id"],
            field=f"{field}.evaluation_vocabulary_id",
        ),
    }


def _validate_advertised_constraints(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="advertised_constraints")
    allowed = frozenset(
        {
            "execution_mode",
            "max_cold_start_ms",
            "max_p95_latency_ms",
            "max_p99_latency_ms",
            "max_runner_tree_peak_rss_bytes",
            "max_accelerator_process_tree_peak_bytes",
            "min_repeat_throughput_fps",
            "min_sustained_fps",
            "quality_requirement",
        }
    )
    _keys(
        record,
        field="advertised_constraints",
        allowed=allowed,
        required=frozenset({"execution_mode"}),
    )
    mode = _enum(
        record["execution_mode"],
        field="advertised_constraints.execution_mode",
        allowed=frozenset({"batch", "soft_realtime"}),
    )
    out: dict[str, Any] = {"execution_mode": mode}
    for key in ("max_cold_start_ms", "max_p95_latency_ms", "max_p99_latency_ms"):
        if key in record:
            out[key] = canonical_decimal_v1(
                record[key], field=f"advertised_constraints.{key}", positive=True
            )
    for key in (
        "max_runner_tree_peak_rss_bytes",
        "max_accelerator_process_tree_peak_bytes",
    ):
        if key in record:
            out[key] = _integer(
                record[key],
                field=f"advertised_constraints.{key}",
                minimum=1,
                maximum=2**63 - 1,
            )
    if mode == "batch":
        if "min_sustained_fps" in record:
            raise ValueError("batch support profile forbids min_sustained_fps")
        if "min_repeat_throughput_fps" in record:
            out["min_repeat_throughput_fps"] = canonical_decimal_v1(
                record["min_repeat_throughput_fps"],
                field="advertised_constraints.min_repeat_throughput_fps",
                positive=True,
            )
    else:
        if "min_repeat_throughput_fps" in record:
            raise ValueError("soft_realtime profile forbids repeat throughput")
        if "min_sustained_fps" in record:
            out["min_sustained_fps"] = canonical_decimal_v1(
                record["min_sustained_fps"],
                field="advertised_constraints.min_sustained_fps",
                positive=True,
            )
    if "quality_requirement" in record:
        out["quality_requirement"] = _validate_quality_requirement(
            record["quality_requirement"], field="advertised_constraints.quality_requirement"
        )
    if len(out) == 1:
        raise ValueError("advertised_constraints requires at least one measured gate")
    return out


@dataclass(frozen=True)
class SupportProfileSpec:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def profile_digest(self) -> str:
        return str(self._record["profile_digest"])


def validate_support_profile_spec(value: Mapping[str, Any]) -> SupportProfileSpec:
    """Validate one immutable public SupportProfileSpec v1."""

    record = _mapping(value, field="SupportProfileSpec")
    allowed = frozenset(
        {
            "schema_version",
            "profile_id",
            "profile_digest",
            "task",
            "environment_fingerprint",
            "qualification_workload_fingerprint",
            "protocol_fingerprint",
            "advertised_constraints",
            "public_limitations",
        }
    )
    _keys(record, field="SupportProfileSpec", allowed=allowed, required=allowed)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("SupportProfileSpec.schema_version: expected 1")
    limitations = [
        _public_text(item, field="public_limitations[]", maximum_bytes=512)
        for item in _list(
            record["public_limitations"],
            field="public_limitations",
            minimum=1,
            maximum=16,
        )
    ]
    if len(limitations) != len(set(limitations)):
        raise ValueError("public_limitations: duplicate value")
    normalized = {
        "schema_version": 1,
        "profile_id": _token(
            record["profile_id"], field="profile_id", pattern=_COMPONENT_RE
        ),
        "profile_digest": _sha256(record["profile_digest"], field="profile_digest"),
        "task": _enum(record["task"], field="task", allowed=TASKS),
        "environment_fingerprint": _sha256(
            record["environment_fingerprint"], field="environment_fingerprint"
        ),
        "qualification_workload_fingerprint": _sha256(
            record["qualification_workload_fingerprint"],
            field="qualification_workload_fingerprint",
        ),
        "protocol_fingerprint": _sha256(
            record["protocol_fingerprint"], field="protocol_fingerprint"
        ),
        "advertised_constraints": _validate_advertised_constraints(
            record["advertised_constraints"]
        ),
        "public_limitations": limitations,
    }
    expected = canonical_sha256_v1(normalized, own_digest_field="profile_digest")
    if normalized["profile_digest"] != expected:
        raise ValueError("profile_digest: digest mismatch")
    return SupportProfileSpec(normalized)


def _validate_review_reference(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="review_reference")
    kind = record.get("kind")
    if kind == "public_repository_id":
        allowed = frozenset({"kind", "value"})
        _keys(record, field="review_reference", allowed=allowed, required=allowed)
        return {
            "kind": kind,
            "value": _token(record["value"], field="review_reference.value"),
        }
    if kind == "site_local_status":
        allowed = frozenset({"kind", "status"})
        _keys(record, field="review_reference", allowed=allowed, required=allowed)
        return {
            "kind": kind,
            "status": _enum(
                record["status"],
                field="review_reference.status",
                allowed=frozenset({"present", "not_applicable"}),
            ),
        }
    raise ValueError("review_reference.kind: unsupported value")


def _validate_profile_references(value: Any, *, minimum: int) -> list[dict[str, str]]:
    entries = _list(value, field="profiles", minimum=minimum, maximum=32)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        record = _mapping(item, field=f"profiles[{index}]")
        keys = frozenset({"profile_id", "profile_digest"})
        _keys(record, field="profile reference", allowed=keys, required=keys)
        profile_id = _token(
            record["profile_id"], field="profile_reference.profile_id", pattern=_COMPONENT_RE
        )
        if profile_id in seen:
            raise ValueError("profiles: duplicate profile_id")
        seen.add(profile_id)
        out.append(
            {
                "profile_id": profile_id,
                "profile_digest": _sha256(
                    record["profile_digest"], field="profile_reference.profile_digest"
                ),
            }
        )
    return out


@dataclass(frozen=True)
class SupportProfileRecord:
    _record: dict[str, Any]
    source_trust_domain: str

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def record_digest(self) -> str:
        return str(self._record["record_digest"])


@dataclass(frozen=True)
class SupportProfileProjection:
    head_digest: str
    definitions: dict[str, SupportProfileSpec]
    assignments: dict[tuple[str, str], dict[str, Any]]
    record_by_digest: dict[str, SupportProfileRecord]


_SUPPORT_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "stream_id",
        "sequence",
        "previous_record_digest",
        "record_id",
        "kind",
        "reviewer_role_id",
        "review_reference",
        "issuer_claim",
        "reason",
        "occurred_at",
        "record_digest",
    }
)


def validate_support_profile_record(
    value: Mapping[str, Any],
    *,
    source_trust_domain: str = "operator_asserted",
) -> SupportProfileRecord:
    """Validate one append-only SupportProfileRecord.

    Trust is supplied by the loader/path authority and is never read from JSON.
    """

    trust = _enum(
        source_trust_domain, field="source_trust_domain", allowed=TRUST_DOMAINS
    )
    record = _mapping(value, field="SupportProfileRecord")
    kind = record.get("kind")
    variant_keys: frozenset[str]
    if kind == "profile_definition":
        variant_keys = frozenset({"profile"})
    elif kind == "profile_set_assignment":
        variant_keys = frozenset(
            {"family_id", "channel", "profiles", "profile_set_digest"}
        )
    else:
        raise ValueError("SupportProfileRecord.kind: unsupported value")
    allowed = _SUPPORT_COMMON_KEYS | variant_keys
    _keys(record, field="SupportProfileRecord", allowed=allowed, required=allowed)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("SupportProfileRecord.schema_version: expected 1")
    sequence = _integer(record["sequence"], field="sequence", minimum=1, maximum=128)
    previous = _sha256(
        record["previous_record_digest"], field="previous_record_digest", allow_zero=True
    )
    if (sequence == 1) != (previous == ZERO_DIGEST):
        raise ValueError("previous_record_digest: zero sentinel only for sequence 1")
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "stream_id": _enum(
            record["stream_id"],
            field="stream_id",
            allowed=frozenset({"support-profiles-v1"}),
        ),
        "sequence": sequence,
        "previous_record_digest": previous,
        "record_id": _token(
            record["record_id"], field="record_id", pattern=_COMPONENT_RE
        ),
        "kind": kind,
        "reviewer_role_id": _token(
            record["reviewer_role_id"],
            field="reviewer_role_id",
            pattern=_ROLE_ID_RE,
        ),
        "review_reference": _validate_review_reference(record["review_reference"]),
        "issuer_claim": _enum(
            record["issuer_claim"],
            field="issuer_claim",
            allowed=frozenset(
                {"repository_source", "site_source", "operator_source", "unknown"}
            ),
        ),
        "reason": _safe_text(record["reason"], field="reason", maximum_bytes=512),
        "occurred_at": _utc(record["occurred_at"], field="occurred_at"),
        "record_digest": _sha256(record["record_digest"], field="record_digest"),
    }
    if kind == "profile_definition":
        normalized["profile"] = validate_support_profile_spec(record["profile"]).to_dict()
    else:
        profiles = _validate_profile_references(record["profiles"], minimum=1)
        set_digest = _sha256(record["profile_set_digest"], field="profile_set_digest")
        if set_digest != canonical_sha256_v1(profiles):
            raise ValueError("profile_set_digest: digest mismatch")
        normalized.update(
            {
                "family_id": _token(
                    record["family_id"], field="family_id", pattern=_COMPONENT_RE
                ),
                "channel": _enum(
                    record["channel"], field="channel", allowed=CHANNELS
                ),
                "profiles": profiles,
                "profile_set_digest": set_digest,
            }
        )
    expected = canonical_sha256_v1(normalized, own_digest_field="record_digest")
    if normalized["record_digest"] != expected:
        raise ValueError("record_digest: digest mismatch")
    return SupportProfileRecord(normalized, trust)


def project_support_profiles(
    records: Sequence[Mapping[str, Any] | SupportProfileRecord],
    *,
    source_trust_domain: str = "operator_asserted",
) -> SupportProfileProjection:
    """Validate and project the one global support-profile chain by sequence."""

    if len(records) > 128:
        raise ValueError("support-profile stream exceeds 128 records")
    definitions: dict[str, SupportProfileSpec] = {}
    assignments: dict[tuple[str, str], dict[str, Any]] = {}
    record_by_digest: dict[str, SupportProfileRecord] = {}
    previous = ZERO_DIGEST
    expected_sequence = 1
    seen_record_ids: set[str] = set()
    for item in records:
        validated = (
            item
            if isinstance(item, SupportProfileRecord)
            else validate_support_profile_record(
                item, source_trust_domain=source_trust_domain
            )
        )
        record = validated.to_dict()
        if record["sequence"] != expected_sequence:
            raise ValueError("support-profile stream has sequence gap or duplicate")
        if record["previous_record_digest"] != previous:
            raise ValueError("support-profile stream has predecessor gap or fork")
        if record["record_id"] in seen_record_ids:
            raise ValueError("support-profile stream has duplicate record_id")
        seen_record_ids.add(record["record_id"])
        if record["kind"] == "profile_definition":
            profile = validate_support_profile_spec(record["profile"])
            profile_id = profile.to_dict()["profile_id"]
            if profile_id in definitions:
                raise ValueError("support-profile definition is immutable and unique")
            definitions[profile_id] = profile
        else:
            for reference in record["profiles"]:
                current = definitions.get(reference["profile_id"])
                if current is None or current.profile_digest != reference["profile_digest"]:
                    raise ValueError("profile-set assignment references missing/changed profile")
            assignments[(record["family_id"], record["channel"])] = {
                "record_id": record["record_id"],
                "record_digest": record["record_digest"],
                "profiles": copy.deepcopy(record["profiles"]),
                "profile_set_digest": record["profile_set_digest"],
                "support_profile_index_head": record["record_digest"],
            }
        previous = record["record_digest"]
        record_by_digest[previous] = validated
        expected_sequence += 1
    return SupportProfileProjection(previous, definitions, assignments, record_by_digest)


def _validate_artifact_reviews(
    value: Any, bundle: AlgorithmBundleSpec, *, field: str
) -> list[dict[str, str]]:
    artifacts = bundle.to_dict()["artifacts"]
    entries = _list(value, field=field, minimum=len(artifacts), maximum=len(artifacts))
    out: list[dict[str, str]] = []
    for artifact, item in zip(artifacts, entries, strict=True):
        record = _mapping(item, field=f"{field}[]")
        keys = frozenset({"artifact_id", "review_state"})
        _keys(record, field=field, allowed=keys, required=keys)
        artifact_id = _token(
            record["artifact_id"], field=f"{field}.artifact_id", pattern=_COMPONENT_RE
        )
        if artifact_id != artifact["artifact_id"]:
            raise ValueError(f"{field}: must cover artifacts in exact order")
        out.append(
            {
                "artifact_id": artifact_id,
                "review_state": _enum(
                    record["review_state"],
                    field=f"{field}.review_state",
                    allowed=LICENSE_REVIEW_STATES,
                ),
            }
        )
    return out


def _validate_evidence_bindings(value: Any, profiles: list[dict[str, str]]) -> list[dict[str, str]]:
    entries = _list(
        value, field="evidence_bindings", minimum=len(profiles), maximum=len(profiles)
    )
    out: list[dict[str, str]] = []
    for profile, item in zip(profiles, entries, strict=True):
        record = _mapping(item, field="evidence_bindings[]")
        keys = frozenset(
            {
                "profile_id",
                "profile_digest",
                "activation_id",
                "activation_digest",
                "trust_domain_claim",
            }
        )
        _keys(record, field="evidence binding", allowed=keys, required=keys)
        normalized = {
            "profile_id": _token(
                record["profile_id"],
                field="evidence_binding.profile_id",
                pattern=_COMPONENT_RE,
            ),
            "profile_digest": _sha256(
                record["profile_digest"], field="evidence_binding.profile_digest"
            ),
            "activation_id": _token(
                record["activation_id"], field="evidence_binding.activation_id"
            ),
            "activation_digest": _sha256(
                record["activation_digest"], field="evidence_binding.activation_digest"
            ),
            "trust_domain_claim": _enum(
                record["trust_domain_claim"],
                field="evidence_binding.trust_domain_claim",
                allowed=TRUST_DOMAINS,
            ),
        }
        if (
            normalized["profile_id"] != profile["profile_id"]
            or normalized["profile_digest"] != profile["profile_digest"]
        ):
            raise ValueError("evidence binding does not match ordered profile")
        out.append(normalized)
    return out


@dataclass(frozen=True)
class BundleLifecycleRecord:
    _record: dict[str, Any]
    source_trust_domain: str

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def event_digest(self) -> str:
        return str(self._record["event_digest"])


@dataclass(frozen=True)
class BundleLifecycleProjection:
    head_digest: str
    bundle_states: dict[str, dict[str, Any]]
    channel_pointers: dict[tuple[str, str], dict[str, Any] | None]
    events: tuple[BundleLifecycleRecord, ...]

    def is_lifecycle_eligible(self, *, family_id: str, channel: str) -> bool:
        """Return only the lifecycle/license gate, not routing availability."""

        if channel not in {"Experimental", "Stable"}:
            return False
        pointer = self.channel_pointers.get((family_id, channel))
        if pointer is None:
            return False
        state = self.bundle_states.get(pointer["bundle_spec_digest"])
        return bool(
            state
            and state["bundle_state"] == "enabled"
            and all(
                review["review_state"] == "approved"
                for review in state["artifact_license_reviews"]
            )
        )


_LIFECYCLE_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "stream_id",
        "sequence",
        "previous_event_digest",
        "event_scope",
        "event_type",
        "reviewer_role_id",
        "review_reference",
        "issuer_claim",
        "reason",
        "occurred_at",
        "event_digest",
        "maintenance_operation",
        "actor_role_id",
        "review_status",
    }
)
_LIFECYCLE_OPTIONAL_COMMON_KEYS = frozenset(
    {"maintenance_operation", "actor_role_id", "review_status"}
)
_GLOBAL_KEYS = frozenset(
    {
        "family_id",
        "bundle_spec_digest",
        "artifact_set_digest",
        "bundle_state",
        "artifact_license_reviews",
        "artifact_members",
        "existing_runs_reproducible",
    }
)
_GLOBAL_OPTIONAL_KEYS = frozenset(
    {"artifact_members", "existing_runs_reproducible"}
)
_ASSIGNMENT_KEYS = frozenset(
    {
        "family_id",
        "channel",
        "target_bundle_spec_digest",
        "target_artifact_set_digest",
        "target_artifact_license_reviews",
        "support_profile_index_head",
        "profile_set_record_id",
        "profile_set_record_digest",
        "profile_set_digest",
        "profiles",
        "evidence_bindings",
        "rollback_target_prior_assignment_digest",
        "promotion_source_channel",
        "promotion_source_pointer_digest",
        "promotion_target_pointer_digest",
        "rollback_target_status",
        "stable_comparator_status",
        "failure_drill_report_digest",
        "failure_drill_reference",
        "automated_pass_reference",
    }
)
_ASSIGNMENT_OPTIONAL_KEYS = frozenset(
    {
        "rollback_target_prior_assignment_digest",
        "promotion_source_channel",
        "promotion_source_pointer_digest",
        "promotion_target_pointer_digest",
        "rollback_target_status",
        "stable_comparator_status",
        "failure_drill_report_digest",
        "failure_drill_reference",
        "automated_pass_reference",
    }
)
_PROMOTION_KEYS = frozenset(
    {
        "promotion_source_channel",
        "promotion_source_pointer_digest",
        "promotion_target_pointer_digest",
        "rollback_target_status",
        "stable_comparator_status",
        "failure_drill_report_digest",
        "failure_drill_reference",
        "automated_pass_reference",
    }
)
_NONE_KEYS = frozenset(
    {
        "family_id",
        "channel",
        "profile_set_digest",
        "profiles",
        "evidence_bindings",
        "prior_bundle_spec_digest",
        "prior_artifact_set_digest",
        "prior_support_profile_index_head",
        "prior_profile_set_record_digest",
        "prior_profile_set_digest",
    }
)


def _lifecycle_common(record: Mapping[str, Any]) -> dict[str, Any]:
    sequence = _integer(record["sequence"], field="sequence", minimum=1, maximum=4096)
    previous = _sha256(
        record["previous_event_digest"], field="previous_event_digest", allow_zero=True
    )
    if (sequence == 1) != (previous == ZERO_DIGEST):
        raise ValueError("previous_event_digest: zero sentinel only for sequence 1")
    normalized = {
        "schema_version": 1,
        "stream_id": _enum(
            record["stream_id"],
            field="stream_id",
            allowed=frozenset({"bundle-lifecycle-v1"}),
        ),
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_scope": record["event_scope"],
        "event_type": record["event_type"],
        "reviewer_role_id": _token(
            record["reviewer_role_id"],
            field="reviewer_role_id",
            pattern=_ROLE_ID_RE,
        ),
        "review_reference": _validate_review_reference(record["review_reference"]),
        "issuer_claim": _enum(
            record["issuer_claim"],
            field="issuer_claim",
            allowed=frozenset(
                {"repository_source", "site_source", "operator_source", "unknown"}
            ),
        ),
        "reason": _safe_text(record["reason"], field="reason", maximum_bytes=512),
        "occurred_at": _utc(record["occurred_at"], field="occurred_at"),
        "event_digest": _sha256(record["event_digest"], field="event_digest"),
    }
    optional = _LIFECYCLE_OPTIONAL_COMMON_KEYS.intersection(record)
    if optional and optional != _LIFECYCLE_OPTIONAL_COMMON_KEYS:
        raise ValueError("maintenance lifecycle metadata must be complete")
    if optional:
        actor = _token(
            record["actor_role_id"],
            field="actor_role_id",
            pattern=_ROLE_ID_RE,
        )
        if actor != normalized["reviewer_role_id"]:
            raise ValueError("actor_role_id must match reviewer_role_id")
        normalized.update(
            {
                "maintenance_operation": _enum(
                    record["maintenance_operation"],
                    field="maintenance_operation",
                    allowed=frozenset(
                        {
                            "disable",
                            "enable",
                            "review_license",
                            "revoke",
                            "rollback_channel",
                            "promote_candidate_to_experimental",
                            "promote_experimental_to_stable",
                        }
                    ),
                ),
                "actor_role_id": actor,
                "review_status": _enum(
                    record["review_status"],
                    field="review_status",
                    allowed=frozenset({"approved", "pending", "rejected"}),
                ),
            }
        )
    return normalized


def _validate_artifact_members(
    value: Any,
    bundle: AlgorithmBundleSpec,
) -> list[dict[str, Any]]:
    items = _list(value, field="artifact_members", minimum=1, maximum=32)
    expected = [
        {
            "artifact_id": artifact["artifact_id"],
            "expected_size_bytes": artifact["expected_size_bytes"],
            "sha256": artifact["sha256"],
        }
        for artifact in bundle.to_dict()["artifacts"]
    ]
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        record = _mapping(item, field=f"artifact_members[{index}]")
        keys = frozenset({"artifact_id", "expected_size_bytes", "sha256"})
        _keys(
            record,
            field=f"artifact_members[{index}]",
            allowed=keys,
            required=keys,
        )
        normalized.append(
            {
                "artifact_id": _token(
                    record["artifact_id"],
                    field=f"artifact_members[{index}].artifact_id",
                    pattern=_COMPONENT_RE,
                ),
                "expected_size_bytes": _integer(
                    record["expected_size_bytes"],
                    field=f"artifact_members[{index}].expected_size_bytes",
                    minimum=1,
                    maximum=MAX_ARTIFACT_BYTES,
                ),
                "sha256": _sha256(
                    record["sha256"],
                    field=f"artifact_members[{index}].sha256",
                ),
            }
        )
    if normalized != expected:
        raise ValueError("artifact_members must exactly match the immutable bundle")
    return normalized


def validate_bundle_lifecycle_record(
    value: Mapping[str, Any],
    *,
    registry: AlgorithmBundleRegistry,
    source_trust_domain: str = "operator_asserted",
) -> BundleLifecycleRecord:
    """Validate one strict discriminated BundleLifecycleRecord."""

    trust = _enum(
        source_trust_domain, field="source_trust_domain", allowed=TRUST_DOMAINS
    )
    record = _mapping(value, field="BundleLifecycleRecord")
    if record.get("schema_version") != 1 or isinstance(record.get("schema_version"), bool):
        raise ValueError("BundleLifecycleRecord.schema_version: expected 1")
    scope = record.get("event_scope")
    event_type = record.get("event_type")
    if scope == "bundle_global" and event_type in {
        "register_global",
        "disable",
        "enable",
        "license_review",
        "revoke",
    }:
        variant_keys = _GLOBAL_KEYS
    elif scope == "channel_assignment" and event_type in {
        "candidate_registration",
        "public_assignment",
    }:
        variant_keys = _ASSIGNMENT_KEYS
    elif scope == "channel_none" and event_type == "channel_none":
        variant_keys = _NONE_KEYS
    else:
        raise ValueError("BundleLifecycleRecord: invalid scope/type combination")
    allowed = _LIFECYCLE_COMMON_KEYS | variant_keys
    required = allowed - _LIFECYCLE_OPTIONAL_COMMON_KEYS
    if scope == "bundle_global":
        required -= _GLOBAL_OPTIONAL_KEYS
    elif scope == "channel_assignment":
        required -= _ASSIGNMENT_OPTIONAL_KEYS
    _keys(record, field="BundleLifecycleRecord", allowed=allowed, required=required)
    normalized = _lifecycle_common(record)
    operation = normalized.get("maintenance_operation")
    if operation is not None:
        if normalized["actor_role_id"] not in {
            "repo_maintainer",
            "release_reviewer",
        }:
            raise ValueError(
                "lifecycle maintenance requires a repository review role"
            )
        if normalized["review_status"] != "approved":
            raise ValueError("lifecycle maintenance requires approved review status")
        if normalized["review_reference"]["kind"] != "public_repository_id":
            raise ValueError(
                "lifecycle maintenance requires a public repository review"
            )
        if normalized["issuer_claim"] != "repository_source":
            raise ValueError(
                "lifecycle maintenance requires repository source provenance"
            )
        expected_operation = {
            "disable": "disable",
            "enable": "enable",
            "license_review": "review_license",
            "revoke": "revoke",
            "channel_none": "rollback_channel",
        }.get(str(event_type))
        if event_type == "public_assignment":
            if operation not in {
                "rollback_channel",
                "promote_candidate_to_experimental",
                "promote_experimental_to_stable",
            }:
                raise ValueError(
                    "maintenance_operation does not match the lifecycle event"
                )
            expected_operation = operation
        if operation != expected_operation:
            raise ValueError("maintenance_operation does not match the lifecycle event")
        if scope == "bundle_global" and not _GLOBAL_OPTIONAL_KEYS.issubset(record):
            raise ValueError(
                "bundle maintenance requires immutable artifact audit metadata"
            )
        if (
            operation == "rollback_channel"
            and scope == "channel_assignment"
            and "rollback_target_prior_assignment_digest" not in record
        ):
            raise ValueError(
                "non-none rollback requires its prior target assignment digest"
            )
    elif "rollback_target_prior_assignment_digest" in record:
        raise ValueError(
            "prior target assignment digest is reserved for reviewed rollback"
        )
    bundles = registry.by_spec_digest()

    if scope == "bundle_global":
        spec_digest = _sha256(
            record["bundle_spec_digest"], field="bundle_spec_digest"
        )
        bundle = bundles.get(spec_digest)
        if bundle is None:
            raise ValueError("bundle_global references unknown bundle spec")
        bundle_record = bundle.to_dict()
        family_id = _token(
            record["family_id"], field="family_id", pattern=_COMPONENT_RE
        )
        artifact_set_digest = _sha256(
            record["artifact_set_digest"], field="artifact_set_digest"
        )
        if (
            family_id != bundle_record["family_id"]
            or artifact_set_digest != bundle.artifact_set_digest
        ):
            raise ValueError("bundle_global family/artifact binding mismatch")
        state = _enum(
            record["bundle_state"],
            field="bundle_state",
            allowed=frozenset({"enabled", "disabled", "revoked"}),
        )
        expected_state = {
            "register_global": frozenset({"enabled", "disabled"}),
            "disable": frozenset({"disabled"}),
            "enable": frozenset({"enabled"}),
            "license_review": frozenset({"enabled", "disabled"}),
            "revoke": frozenset({"revoked"}),
        }[event_type]
        if state not in expected_state:
            raise ValueError("bundle_global event type/state mismatch")
        normalized.update(
            {
                "family_id": family_id,
                "bundle_spec_digest": spec_digest,
                "artifact_set_digest": artifact_set_digest,
                "bundle_state": state,
                "artifact_license_reviews": _validate_artifact_reviews(
                    record["artifact_license_reviews"],
                    bundle,
                    field="artifact_license_reviews",
                ),
            }
        )
        global_optional = _GLOBAL_OPTIONAL_KEYS.intersection(record)
        if global_optional and global_optional != _GLOBAL_OPTIONAL_KEYS:
            raise ValueError("bundle maintenance audit metadata must be complete")
        if global_optional:
            normalized.update(
                {
                    "artifact_members": _validate_artifact_members(
                        record["artifact_members"], bundle
                    ),
                    "existing_runs_reproducible": _boolean(
                        record["existing_runs_reproducible"],
                        field="existing_runs_reproducible",
                    ),
                }
            )
    elif scope == "channel_assignment":
        spec_digest = _sha256(
            record["target_bundle_spec_digest"], field="target_bundle_spec_digest"
        )
        bundle = bundles.get(spec_digest)
        if bundle is None:
            raise ValueError("channel assignment references unknown bundle spec")
        bundle_record = bundle.to_dict()
        family_id = _token(
            record["family_id"], field="family_id", pattern=_COMPONENT_RE
        )
        channel = _enum(record["channel"], field="channel", allowed=CHANNELS)
        artifact_set_digest = _sha256(
            record["target_artifact_set_digest"],
            field="target_artifact_set_digest",
        )
        if (
            family_id != bundle_record["family_id"]
            or artifact_set_digest != bundle.artifact_set_digest
        ):
            raise ValueError("channel assignment family/artifact binding mismatch")
        reviews = _validate_artifact_reviews(
            record["target_artifact_license_reviews"],
            bundle,
            field="target_artifact_license_reviews",
        )
        support_head = _sha256(
            record["support_profile_index_head"],
            field="support_profile_index_head",
            allow_zero=True,
        )
        if event_type == "candidate_registration":
            if channel != "Candidate":
                raise ValueError("candidate_registration requires Candidate channel")
            if record["profile_set_record_id"] is not None:
                raise ValueError("candidate_registration forbids profile-set record ID")
            if record["profile_set_record_digest"] is not None:
                raise ValueError("candidate_registration forbids profile-set record digest")
            profiles = _validate_profile_references(record["profiles"], minimum=0)
            bindings = _validate_evidence_bindings(record["evidence_bindings"], profiles)
            if profiles or bindings or record["profile_set_digest"] != EMPTY_PROFILE_SET_DIGEST:
                raise ValueError("candidate_registration requires empty profile/evidence set")
            profile_record_id = None
            profile_record_digest = None
            profile_set_digest = EMPTY_PROFILE_SET_DIGEST
        else:
            if channel not in {"Experimental", "Stable"}:
                raise ValueError("public_assignment requires Experimental or Stable")
            if bundle_record["test_only"]:
                raise ValueError("test-only bundle cannot receive a public assignment")
            if bundle_record["execution_network_required"]:
                raise ValueError("network-requiring bundle is ineligible for public assignment")
            profiles = _validate_profile_references(record["profiles"], minimum=1)
            profile_set_digest = _sha256(
                record["profile_set_digest"], field="profile_set_digest"
            )
            if profile_set_digest != canonical_sha256_v1(profiles):
                raise ValueError("public assignment profile-set digest mismatch")
            profile_record_id = _token(
                record["profile_set_record_id"],
                field="profile_set_record_id",
                pattern=_COMPONENT_RE,
            )
            profile_record_digest = _sha256(
                record["profile_set_record_digest"],
                field="profile_set_record_digest",
            )
            bindings = _validate_evidence_bindings(
                record["evidence_bindings"], profiles
            )
            if any(
                binding["trust_domain_claim"] != "yolozu_managed"
                for binding in bindings
            ):
                raise ValueError("public assignment requires yolozu_managed evidence claim")
        normalized.update(
            {
                "family_id": family_id,
                "channel": channel,
                "target_bundle_spec_digest": spec_digest,
                "target_artifact_set_digest": artifact_set_digest,
                "target_artifact_license_reviews": reviews,
                "support_profile_index_head": support_head,
                "profile_set_record_id": profile_record_id,
                "profile_set_record_digest": profile_record_digest,
                "profile_set_digest": profile_set_digest,
                "profiles": profiles,
                "evidence_bindings": bindings,
            }
        )
        if "rollback_target_prior_assignment_digest" in record:
            normalized["rollback_target_prior_assignment_digest"] = _sha256(
                record["rollback_target_prior_assignment_digest"],
                field="rollback_target_prior_assignment_digest",
            )
        promotion_keys = _PROMOTION_KEYS.intersection(record)
        if operation in {
            "promote_candidate_to_experimental",
            "promote_experimental_to_stable",
        } and promotion_keys != _PROMOTION_KEYS:
            raise ValueError("reviewed promotion requires complete lifecycle metadata")
        if promotion_keys and promotion_keys != _PROMOTION_KEYS:
            raise ValueError("promotion lifecycle metadata must be complete")
        if promotion_keys:
            if operation not in {
                "promote_candidate_to_experimental",
                "promote_experimental_to_stable",
            }:
                raise ValueError("promotion metadata is reserved for reviewed promotion")
            source_channel = _enum(
                record["promotion_source_channel"],
                field="promotion_source_channel",
                allowed=frozenset({"Candidate", "Experimental"}),
            )
            source_pointer = _sha256(
                record["promotion_source_pointer_digest"],
                field="promotion_source_pointer_digest",
            )
            target_pointer_raw = record["promotion_target_pointer_digest"]
            target_pointer = (
                None
                if target_pointer_raw is None
                else _sha256(
                    target_pointer_raw,
                    field="promotion_target_pointer_digest",
                )
            )
            rollback_status = _enum(
                record["rollback_target_status"],
                field="rollback_target_status",
                allowed=frozenset({"none_abstention", "prior_assignment"}),
            )
            comparator_status = _enum(
                record["stable_comparator_status"],
                field="stable_comparator_status",
                allowed=frozenset(
                    {
                        "not_applicable_candidate_to_experimental",
                        "comparator_not_applicable_first_assignment",
                        "exact_current_stable",
                    }
                ),
            )
            drill_digest_raw = record["failure_drill_report_digest"]
            drill_reference_raw = record["failure_drill_reference"]
            automated_reference_raw = record["automated_pass_reference"]
            drill_digest = (
                None
                if drill_digest_raw is None
                else _sha256(drill_digest_raw, field="failure_drill_report_digest")
            )
            drill_reference = (
                None
                if drill_reference_raw is None
                else _token(drill_reference_raw, field="failure_drill_reference")
            )
            automated_reference = (
                None
                if automated_reference_raw is None
                else _token(automated_reference_raw, field="automated_pass_reference")
            )
            if operation == "promote_candidate_to_experimental":
                if source_channel != "Candidate" or channel != "Experimental":
                    raise ValueError("Candidate promotion channel pair is invalid")
                if comparator_status != "not_applicable_candidate_to_experimental":
                    raise ValueError("Candidate promotion comparator status is invalid")
                if any(
                    item is not None
                    for item in (drill_digest, drill_reference, automated_reference)
                ):
                    raise ValueError("Candidate promotion forbids Stable drill metadata")
            else:
                if source_channel != "Experimental" or channel != "Stable":
                    raise ValueError("Stable promotion channel pair is invalid")
                if comparator_status == "not_applicable_candidate_to_experimental":
                    raise ValueError("Stable promotion comparator status is invalid")
                if any(
                    item is None
                    for item in (drill_digest, drill_reference, automated_reference)
                ):
                    raise ValueError("Stable promotion requires complete drill metadata")
            prior_digest_present = "rollback_target_prior_assignment_digest" in record
            if (rollback_status == "prior_assignment") != prior_digest_present:
                raise ValueError("promotion rollback target metadata is inconsistent")
            normalized.update(
                {
                    "promotion_source_channel": source_channel,
                    "promotion_source_pointer_digest": source_pointer,
                    "promotion_target_pointer_digest": target_pointer,
                    "rollback_target_status": rollback_status,
                    "stable_comparator_status": comparator_status,
                    "failure_drill_report_digest": drill_digest,
                    "failure_drill_reference": drill_reference,
                    "automated_pass_reference": automated_reference,
                }
            )
    else:
        family_id = _token(
            record["family_id"], field="family_id", pattern=_COMPONENT_RE
        )
        channel = _enum(record["channel"], field="channel", allowed=CHANNELS)
        profiles = _validate_profile_references(record["profiles"], minimum=0)
        bindings = _validate_evidence_bindings(record["evidence_bindings"], profiles)
        if profiles or bindings or record["profile_set_digest"] != EMPTY_PROFILE_SET_DIGEST:
            raise ValueError("channel_none requires empty target profile/evidence set")
        prior_spec = record["prior_bundle_spec_digest"]
        prior_artifact = record["prior_artifact_set_digest"]
        prior_set_record = record["prior_profile_set_record_digest"]
        if prior_spec is None:
            if prior_artifact is not None or prior_set_record is not None:
                raise ValueError("prior-none audit fields must all be null")
            prior_head = _sha256(
                record["prior_support_profile_index_head"],
                field="prior_support_profile_index_head",
                allow_zero=True,
            )
            prior_set_digest = _sha256(
                record["prior_profile_set_digest"],
                field="prior_profile_set_digest",
                allow_zero=True,
            )
            if prior_head != ZERO_DIGEST or prior_set_digest != EMPTY_PROFILE_SET_DIGEST:
                raise ValueError("prior-none requires zero head and empty set digest")
        else:
            prior_spec = _sha256(prior_spec, field="prior_bundle_spec_digest")
            bundle = bundles.get(prior_spec)
            if bundle is None or bundle.to_dict()["family_id"] != family_id:
                raise ValueError("channel_none prior spec is unknown/wrong family")
            prior_artifact = _sha256(
                prior_artifact, field="prior_artifact_set_digest"
            )
            if prior_artifact != bundle.artifact_set_digest:
                raise ValueError("channel_none prior artifact mismatch")
            prior_head = _sha256(
                record["prior_support_profile_index_head"],
                field="prior_support_profile_index_head",
                allow_zero=True,
            )
            prior_set_record = (
                None
                if prior_set_record is None
                else _sha256(
                    prior_set_record, field="prior_profile_set_record_digest"
                )
            )
            prior_set_digest = _sha256(
                record["prior_profile_set_digest"],
                field="prior_profile_set_digest",
                allow_zero=True,
            )
        normalized.update(
            {
                "family_id": family_id,
                "channel": channel,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
                "prior_bundle_spec_digest": prior_spec,
                "prior_artifact_set_digest": prior_artifact,
                "prior_support_profile_index_head": prior_head,
                "prior_profile_set_record_digest": prior_set_record,
                "prior_profile_set_digest": prior_set_digest,
            }
        )

    expected = canonical_sha256_v1(normalized, own_digest_field="event_digest")
    if normalized["event_digest"] != expected:
        raise ValueError("event_digest: digest mismatch")
    return BundleLifecycleRecord(normalized, trust)


def validate_support_profile_snapshot(
    event: dict[str, Any], support_profiles: SupportProfileProjection
) -> None:
    """Require one lifecycle pointer to bind an exact historical set snapshot."""
    head = event["support_profile_index_head"]
    if head != ZERO_DIGEST and head not in support_profiles.record_by_digest:
        raise ValueError("lifecycle assignment references unknown support-profile head")
    if event["event_type"] == "candidate_registration":
        return
    set_digest = event["profile_set_record_digest"]
    set_record = support_profiles.record_by_digest.get(set_digest)
    if set_record is None:
        raise ValueError("lifecycle assignment references unknown profile-set record")
    set_payload = set_record.to_dict()
    if (
        set_payload["kind"] != "profile_set_assignment"
        or set_payload["record_id"] != event["profile_set_record_id"]
        or set_payload["family_id"] != event["family_id"]
        or set_payload["channel"] != event["channel"]
        or set_payload["profiles"] != event["profiles"]
        or set_payload["profile_set_digest"] != event["profile_set_digest"]
    ):
        raise ValueError("lifecycle support-profile snapshot mismatch")
    if head == ZERO_DIGEST:
        raise ValueError("public assignment requires a nonzero support-profile head")
    head_record = support_profiles.record_by_digest[head].to_dict()
    if head_record["sequence"] < set_payload["sequence"]:
        raise ValueError("profile-set record is not in the referenced support prefix")


def project_bundle_lifecycle(
    registry: AlgorithmBundleRegistry,
    records: Sequence[Mapping[str, Any] | BundleLifecycleRecord],
    *,
    source_trust_domain: str = "operator_asserted",
    support_profiles: SupportProfileProjection | None = None,
) -> BundleLifecycleProjection:
    """Project the global lifecycle stream strictly by sequence."""

    if len(records) > 4096:
        raise ValueError("bundle lifecycle exceeds 4096 records")
    bundle_states: dict[str, dict[str, Any]] = {}
    pointers: dict[tuple[str, str], dict[str, Any] | None] = {}
    events: list[BundleLifecycleRecord] = []
    candidate_registered: set[str] = set()
    previous = ZERO_DIGEST
    expected_sequence = 1
    bundles = registry.by_spec_digest()
    for item in records:
        validated = (
            item
            if isinstance(item, BundleLifecycleRecord)
            else validate_bundle_lifecycle_record(
                item,
                registry=registry,
                source_trust_domain=source_trust_domain,
            )
        )
        event = validated.to_dict()
        if event["sequence"] != expected_sequence:
            raise ValueError("bundle lifecycle has sequence gap or duplicate")
        if event["previous_event_digest"] != previous:
            raise ValueError("bundle lifecycle has predecessor gap or fork")

        if event["event_scope"] == "bundle_global":
            spec_digest = event["bundle_spec_digest"]
            prior = bundle_states.get(spec_digest)
            event_type = event["event_type"]
            if event_type == "register_global":
                if prior is not None:
                    raise ValueError("bundle spec may be registered only once")
            else:
                if prior is None:
                    raise ValueError("bundle global transition precedes registration")
                if prior["bundle_state"] == "revoked":
                    raise ValueError("revoked bundle spec is terminal")
                if event_type == "disable":
                    if prior["bundle_state"] != "enabled":
                        raise ValueError("disable requires enabled bundle")
                    if event["artifact_license_reviews"] != prior["artifact_license_reviews"]:
                        raise ValueError("disable cannot also rewrite license review")
                elif event_type == "enable":
                    if prior["bundle_state"] != "disabled":
                        raise ValueError("enable requires disabled bundle")
                    if event["artifact_license_reviews"] != prior["artifact_license_reviews"]:
                        raise ValueError("enable cannot also rewrite license review")
                elif event_type == "license_review":
                    if event["bundle_state"] != prior["bundle_state"]:
                        raise ValueError("license review cannot change bundle state")
                    if event["artifact_license_reviews"] == prior["artifact_license_reviews"]:
                        raise ValueError("license review must change at least one review")
                elif event_type == "revoke":
                    if event["artifact_license_reviews"] != prior["artifact_license_reviews"]:
                        raise ValueError("revoke cannot also rewrite license review")
            bundle_states[spec_digest] = {
                "family_id": event["family_id"],
                "artifact_set_digest": event["artifact_set_digest"],
                "bundle_state": event["bundle_state"],
                "artifact_license_reviews": copy.deepcopy(
                    event["artifact_license_reviews"]
                ),
                "event_digest": event["event_digest"],
            }
        elif event["event_scope"] == "channel_assignment":
            spec_digest = event["target_bundle_spec_digest"]
            state = bundle_states.get(spec_digest)
            if state is None:
                raise ValueError("channel assignment precedes global registration")
            if state["artifact_license_reviews"] != event["target_artifact_license_reviews"]:
                raise ValueError("channel assignment artifact review snapshot mismatch")
            if event["event_type"] == "public_assignment":
                if spec_digest not in candidate_registered:
                    raise ValueError("public assignment target lacks candidate registration")
                if state["bundle_state"] != "enabled" or any(
                    review["review_state"] != "approved"
                    for review in state["artifact_license_reviews"]
                ):
                    raise ValueError("public assignment target is not currently eligible")
                if support_profiles is None:
                    raise ValueError(
                        "public assignment requires a validated support-profile projection"
                    )
                validate_support_profile_snapshot(event, support_profiles)
                if event.get("maintenance_operation") == "rollback_channel":
                    prior_digest = event[
                        "rollback_target_prior_assignment_digest"
                    ]
                    if not any(
                        prior.to_dict()["event_digest"] == prior_digest
                        and prior.to_dict()["event_type"] == "public_assignment"
                        and prior.to_dict().get("family_id") == event["family_id"]
                        and prior.to_dict().get("channel") == event["channel"]
                        and prior.to_dict().get("target_bundle_spec_digest")
                        == spec_digest
                        for prior in events
                    ):
                        raise ValueError(
                            "rollback target assignment digest does not reference "
                            "an exact prior assignment"
                        )
                elif event.get("maintenance_operation") in {
                    "promote_candidate_to_experimental",
                    "promote_experimental_to_stable",
                }:
                    source_key = (
                        event["family_id"],
                        event["promotion_source_channel"],
                    )
                    source_pointer = pointers.get(source_key)
                    if (
                        source_pointer is None
                        or source_pointer["bundle_spec_digest"] != spec_digest
                        or source_pointer["lifecycle_event_digest"]
                        != event["promotion_source_pointer_digest"]
                    ):
                        raise ValueError(
                            "promotion source is not the exact current pointer"
                        )
                    target_key = (event["family_id"], event["channel"])
                    target_pointer = pointers.get(target_key)
                    target_digest = (
                        None
                        if target_pointer is None
                        else target_pointer["lifecycle_event_digest"]
                    )
                    if target_digest != event["promotion_target_pointer_digest"]:
                        raise ValueError(
                            "promotion target expectation is stale"
                        )
                    if event["rollback_target_status"] == "prior_assignment":
                        if (
                            target_pointer is None
                            or event["rollback_target_prior_assignment_digest"]
                            != target_pointer["lifecycle_event_digest"]
                        ):
                            raise ValueError(
                                "promotion prior rollback target is not current"
                            )
                    operation = event["maintenance_operation"]
                    comparator = event["stable_comparator_status"]
                    if operation == "promote_experimental_to_stable":
                        if target_pointer is None:
                            if comparator != "comparator_not_applicable_first_assignment":
                                raise ValueError(
                                    "first Stable promotion requires no-comparator status"
                                )
                        elif (
                            comparator != "exact_current_stable"
                            or target_pointer["profiles"] != event["profiles"]
                        ):
                            raise ValueError(
                                "Stable replacement changed its exact profile set"
                            )
            elif support_profiles is not None:
                head = event["support_profile_index_head"]
                if head != ZERO_DIGEST and head not in support_profiles.record_by_digest:
                    raise ValueError("candidate registration has unknown support head")
            elif event["support_profile_index_head"] != ZERO_DIGEST:
                raise ValueError(
                    "nonzero candidate support head requires a validated projection"
                )
            if event["event_type"] == "candidate_registration":
                candidate_registered.add(spec_digest)
            key = (event["family_id"], event["channel"])
            pointers[key] = {
                "bundle_spec_digest": spec_digest,
                "artifact_set_digest": event["target_artifact_set_digest"],
                "lifecycle_event_digest": event["event_digest"],
                "support_profile_index_head": event["support_profile_index_head"],
                "profile_set_record_id": event["profile_set_record_id"],
                "profile_set_record_digest": event["profile_set_record_digest"],
                "profile_set_digest": event["profile_set_digest"],
                "profiles": copy.deepcopy(event["profiles"]),
                "evidence_bindings": copy.deepcopy(event["evidence_bindings"]),
            }
        else:
            key = (event["family_id"], event["channel"])
            prior = pointers.get(key)
            prior_spec = None if prior is None else prior["bundle_spec_digest"]
            prior_artifact = None if prior is None else prior["artifact_set_digest"]
            prior_head = ZERO_DIGEST if prior is None else prior["support_profile_index_head"]
            prior_set_record = (
                None if prior is None else prior["profile_set_record_digest"]
            )
            prior_set_digest = (
                EMPTY_PROFILE_SET_DIGEST if prior is None else prior["profile_set_digest"]
            )
            if (
                event["prior_bundle_spec_digest"] != prior_spec
                or event["prior_artifact_set_digest"] != prior_artifact
                or event["prior_support_profile_index_head"] != prior_head
                or event["prior_profile_set_record_digest"] != prior_set_record
                or event["prior_profile_set_digest"] != prior_set_digest
            ):
                raise ValueError("channel_none does not audit-bind the current pointer")
            pointers[key] = None

        previous = event["event_digest"]
        expected_sequence += 1
        events.append(validated)

    # Registry membership is immutable. This final pass catches projection keys
    # that somehow refer to a foreign family after future validator extensions.
    for (family_id, _channel), pointer in pointers.items():
        if pointer is not None:
            bundle = bundles[pointer["bundle_spec_digest"]]
            if bundle.to_dict()["family_id"] != family_id:
                raise ValueError("channel pointer family mismatch")
    return BundleLifecycleProjection(previous, bundle_states, pointers, tuple(events))
