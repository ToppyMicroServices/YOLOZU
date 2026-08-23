"""Strict v1 adaptive image job, workload, and environment records."""

from __future__ import annotations

import copy
import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .canonical import canonical_decimal_v1, canonical_sha256_v1

if TYPE_CHECKING:  # pragma: no cover
    from .inventory import DecodedInputInventory

__all__ = [
    "EnvironmentProfile",
    "ImageJobSpec",
    "QualificationWorkloadProfile",
    "build_qualification_workload_profile",
    "compute_environment_fingerprint",
    "compute_workload_fingerprint",
    "validate_environment_profile",
    "validate_image_job_spec",
    "validate_qualification_workload_profile",
]


SCHEMA_VERSION = 1
TASKS = frozenset({"object_detection", "instance_segmentation"})
INPUT_MODES = frozenset({"single_image", "bounded_directory"})
EXECUTION_MODES = frozenset({"batch", "soft_realtime"})
PROMPT_MODES = frozenset({"fixed_classes", "text"})
RANKING_POLICIES = frozenset(
    {"accuracy_first", "latency_first", "throughput_first", "memory_first"}
)
COMPUTE_POLICIES = frozenset({"auto", "cpu_only", "accelerator_required"})
MATURITIES = ("Stable", "Experimental")
PRECISIONS = ("fp32", "tf32", "fp16", "bf16", "int8")
COLOR_MODES = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr", "LAB", "HSV", "I", "F", "I;16"})
PROBE_STATUSES = frozenset({"present", "absent", "unsupported", "failed"})

CPU_PROVIDER_IDS = frozenset({"cpu", "onnxruntime_cpu", "openvino_cpu"})
ACCELERATOR_PROVIDER_IDS = frozenset(
    {
        "cuda",
        "tensorrt",
        "mps",
        "coreml",
        "onnxruntime_cuda",
        "onnxruntime_tensorrt",
        "onnxruntime_coreml",
    }
)

LATENCY_INTERVAL_ID = "image_e2e_validated_handoff_v1"
HANDOFF_ID = "image_result_mask_handoff_v1"
HANDOFF_VERSION = 1
HANDOFF_SCRATCH_CAP_BYTES = 536_870_912
HANDOFF_MAX_MASK_ARTIFACTS = 1_000
HANDOFF_MAX_OUTPUT_FILES = 1_003
HANDOFF_MAX_OUTPUT_BYTES = 4_294_967_296
MAX_SUSTAINED_SAMPLES = 1_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_SHORT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

_IMAGE_JOB_KEYS = frozenset(
    {
        "schema_version",
        "task",
        "prompt_mode",
        "fixed_classes",
        "text_prompts",
        "input_mode",
        "execution_mode",
        "batch_size",
        "concurrency",
        "max_images",
        "max_results_per_image",
        "job_timeout_seconds",
        "ranking_policy",
        "allowed_maturities",
        "network_policy",
        "compute_policy",
        "provider_allowlist",
        "precision_allowlist",
        "spdx_allowlist",
        "max_cold_start_ms",
        "max_p95_latency_ms",
        "max_runner_tree_peak_rss_bytes",
        "max_accelerator_process_tree_peak_bytes",
        "min_repeat_throughput_fps",
        "min_sustained_fps",
        "quality_requirement",
    }
)
_IMAGE_JOB_REQUIRED = frozenset(
    {
        "schema_version",
        "task",
        "prompt_mode",
        "input_mode",
        "execution_mode",
        "batch_size",
        "concurrency",
        "max_images",
        "max_results_per_image",
        "job_timeout_seconds",
        "ranking_policy",
        "allowed_maturities",
        "network_policy",
        "compute_policy",
    }
)
_QUALITY_KEYS = frozenset(
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

_WORKLOAD_KEYS = frozenset(
    {
        "schema_version",
        "collector_id",
        "collector_version",
        "task",
        "input_mode",
        "execution_mode",
        "compute_policy",
        "provider_allowlist",
        "precision_allowlist",
        "input_count",
        "input_order",
        "decoded_inputs",
        "decoder",
        "batch_size",
        "concurrency",
        "max_results_per_image",
        "latency_interval_id",
        "handoff",
        "max_sustained_samples",
        "prompt_characteristics",
        "quality_identity",
        "workload_fingerprint",
    }
)
_WORKLOAD_REQUIRED = _WORKLOAD_KEYS - {"max_sustained_samples", "quality_identity"}

_ENVIRONMENT_KEYS = frozenset(
    {
        "schema_version",
        "collector_id",
        "collector_version",
        "collected_at",
        "os",
        "cpu",
        "total_memory",
        "accelerators",
        "runtimes",
        "power_performance_mode",
        "probe_issues",
        "environment_fingerprint",
    }
)


def _deepcopy_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


@dataclass(frozen=True)
class ImageJobSpec:
    """Validated, normalized ImageJobSpec v1."""

    _record: dict[str, Any]
    local_job_digest: str

    def to_dict(self) -> dict[str, Any]:
        return _deepcopy_dict(self._record)

    @property
    def prompt_phrases(self) -> tuple[str, ...]:
        field = "fixed_classes" if self._record["prompt_mode"] == "fixed_classes" else "text_prompts"
        return tuple(self._record[field])


@dataclass(frozen=True)
class QualificationWorkloadProfile:
    """Validated, shareable performance-relevant workload profile."""

    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _deepcopy_dict(self._record)

    @property
    def workload_fingerprint(self) -> str:
        return str(self._record["workload_fingerprint"])


@dataclass(frozen=True)
class EnvironmentProfile:
    """Validated privacy-safe environment observation."""

    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _deepcopy_dict(self._record)

    @property
    def environment_fingerprint(self) -> str:
        return str(self._record["environment_fingerprint"])


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected object")
    return dict(value)


def _check_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field}: unknown keys: {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{field}: missing required keys: {', '.join(missing)}")


def _exact_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum}")
    return value


def _enum(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field}: unsupported value")
    return value


def _identifier(value: Any, *, field: str, short: bool = False) -> str:
    pattern = _SHORT_ID_RE if short else _ID_RE
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{field}: invalid identifier")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field}: expected lowercase SHA-256")
    return value


def _sorted_unique_ids(
    value: Any,
    *,
    field: str,
    maximum: int,
    allowed: Sequence[str] | None = None,
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{field}: expected list with at most {maximum} entries")
    normalized = [_identifier(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field}: duplicate entries are invalid")
    if allowed is not None and any(item not in allowed for item in normalized):
        raise ValueError(f"{field}: unsupported entry")
    return sorted(normalized, key=lambda item: item.encode("utf-8"))


def _normalize_prompts(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not (1 <= len(value) <= 128):
        raise ValueError(f"{field}: expected 1..128 phrases")
    normalized: list[str] = []
    total_bytes = 0
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}]: expected string")
        normalized_raw = unicodedata.normalize("NFKC", item)
        if any(unicodedata.category(character).startswith("C") for character in normalized_raw):
            raise ValueError(f"{field}[{index}]: control characters are invalid")
        phrase = normalized_raw.strip()
        if not (1 <= len(phrase) <= 256):
            raise ValueError(f"{field}[{index}]: expected 1..256 Unicode code points")
        total_bytes += len(phrase.encode("utf-8"))
        normalized.append(phrase)
    if total_bytes > 4096:
        raise ValueError(f"{field}: normalized payload exceeds 4096 UTF-8 bytes")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field}: duplicate phrases after normalization")
    return normalized


def _quality_requirement(value: Any) -> dict[str, Any]:
    quality = _mapping(value, field="quality_requirement")
    _check_keys(
        quality,
        field="quality_requirement",
        allowed=_QUALITY_KEYS,
        required=_QUALITY_KEYS,
    )
    return {
        "metric_id": _identifier(quality["metric_id"], field="quality_requirement.metric_id"),
        "direction": _enum(
            quality["direction"],
            field="quality_requirement.direction",
            allowed=frozenset({"higher_is_better", "lower_is_better"}),
        ),
        "threshold": canonical_decimal_v1(
            quality["threshold"], field="quality_requirement.threshold"
        ),
        "evaluation_dataset_id": _identifier(
            quality["evaluation_dataset_id"], field="quality_requirement.evaluation_dataset_id"
        ),
        "evaluation_dataset_sha256": _sha256(
            quality["evaluation_dataset_sha256"],
            field="quality_requirement.evaluation_dataset_sha256",
        ),
        "evaluation_protocol_sha256": _sha256(
            quality["evaluation_protocol_sha256"],
            field="quality_requirement.evaluation_protocol_sha256",
        ),
        "evaluation_vocabulary_id": _identifier(
            quality["evaluation_vocabulary_id"],
            field="quality_requirement.evaluation_vocabulary_id",
        ),
    }


def validate_image_job_spec(value: Any) -> ImageJobSpec:
    """Validate and normalize one ImageJobSpec v1 object."""

    record = _mapping(value, field="ImageJobSpec")
    _check_keys(
        record,
        field="ImageJobSpec",
        allowed=_IMAGE_JOB_KEYS,
        required=_IMAGE_JOB_REQUIRED,
    )
    schema_version = _exact_int(
        record["schema_version"], field="schema_version", minimum=1, maximum=1
    )
    task = _enum(record["task"], field="task", allowed=TASKS)
    prompt_mode = _enum(record["prompt_mode"], field="prompt_mode", allowed=PROMPT_MODES)
    input_mode = _enum(record["input_mode"], field="input_mode", allowed=INPUT_MODES)
    execution_mode = _enum(
        record["execution_mode"], field="execution_mode", allowed=EXECUTION_MODES
    )
    batch_size = _exact_int(record["batch_size"], field="batch_size", minimum=1, maximum=1)
    concurrency = _exact_int(record["concurrency"], field="concurrency", minimum=1, maximum=1)
    max_images = _exact_int(record["max_images"], field="max_images", minimum=1, maximum=100)
    if input_mode == "single_image" and max_images != 1:
        raise ValueError("max_images: single_image requires exactly 1")
    max_results = _exact_int(
        record["max_results_per_image"],
        field="max_results_per_image",
        minimum=1,
        maximum=1_000,
    )
    timeout = _exact_int(
        record["job_timeout_seconds"],
        field="job_timeout_seconds",
        minimum=1,
        maximum=3_600,
    )
    ranking = _enum(record["ranking_policy"], field="ranking_policy", allowed=RANKING_POLICIES)
    compute = _enum(record["compute_policy"], field="compute_policy", allowed=COMPUTE_POLICIES)
    if record["network_policy"] != "deny":
        raise ValueError("network_policy: v1 requires deny")

    maturities_raw = record["allowed_maturities"]
    if not isinstance(maturities_raw, list) or not (1 <= len(maturities_raw) <= 2):
        raise ValueError("allowed_maturities: expected 1..2 entries")
    if any(item not in MATURITIES for item in maturities_raw):
        raise ValueError("allowed_maturities: Candidate and Research are not selectable")
    if len(set(maturities_raw)) != len(maturities_raw):
        raise ValueError("allowed_maturities: duplicate entries are invalid")
    maturities = [item for item in MATURITIES if item in maturities_raw]

    provider_allowlist = _sorted_unique_ids(
        record.get("provider_allowlist", []), field="provider_allowlist", maximum=8
    )
    precision_allowlist = _sorted_unique_ids(
        record.get("precision_allowlist", []),
        field="precision_allowlist",
        maximum=len(PRECISIONS),
        allowed=PRECISIONS,
    )
    spdx_allowlist = _sorted_unique_ids(
        record.get("spdx_allowlist", []), field="spdx_allowlist", maximum=64
    )

    if compute == "cpu_only" and any(item in ACCELERATOR_PROVIDER_IDS for item in provider_allowlist):
        raise ValueError("provider_allowlist: cpu_only contradicts an accelerator provider")
    if compute == "accelerator_required" and any(item in CPU_PROVIDER_IDS for item in provider_allowlist):
        raise ValueError("provider_allowlist: accelerator_required contradicts a CPU provider")
    if ranking == "memory_first" and compute == "auto":
        raise ValueError("ranking_policy: memory_first requires an explicit compute policy")
    if compute == "cpu_only" and "max_accelerator_process_tree_peak_bytes" in record:
        raise ValueError("max_accelerator_process_tree_peak_bytes: not applicable to cpu_only")

    has_fixed = "fixed_classes" in record
    has_text = "text_prompts" in record
    if prompt_mode == "fixed_classes":
        if not has_fixed or has_text:
            raise ValueError("prompt_mode=fixed_classes requires only fixed_classes")
        prompts = _normalize_prompts(record["fixed_classes"], field="fixed_classes")
        prompt_field = "fixed_classes"
    else:
        if not has_text or has_fixed:
            raise ValueError("prompt_mode=text requires only text_prompts")
        prompts = _normalize_prompts(record["text_prompts"], field="text_prompts")
        prompt_field = "text_prompts"

    normalized: dict[str, Any] = {
        "schema_version": schema_version,
        "task": task,
        "prompt_mode": prompt_mode,
        prompt_field: prompts,
        "input_mode": input_mode,
        "execution_mode": execution_mode,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "max_images": max_images,
        "max_results_per_image": max_results,
        "job_timeout_seconds": timeout,
        "ranking_policy": ranking,
        "allowed_maturities": maturities,
        "network_policy": "deny",
        "compute_policy": compute,
        "provider_allowlist": provider_allowlist,
        "precision_allowlist": precision_allowlist,
        "spdx_allowlist": spdx_allowlist,
    }

    for field in ("max_cold_start_ms", "max_p95_latency_ms"):
        if field in record:
            normalized[field] = canonical_decimal_v1(
                record[field], field=field, positive=True
            )
    for field in (
        "max_runner_tree_peak_rss_bytes",
        "max_accelerator_process_tree_peak_bytes",
    ):
        if field in record:
            normalized[field] = _exact_int(
                record[field], field=field, minimum=1, maximum=9_223_372_036_854_775_807
            )

    if execution_mode == "batch":
        if "min_sustained_fps" in record:
            raise ValueError("min_sustained_fps: valid only for soft_realtime")
        if "min_repeat_throughput_fps" in record:
            normalized["min_repeat_throughput_fps"] = canonical_decimal_v1(
                record["min_repeat_throughput_fps"],
                field="min_repeat_throughput_fps",
                positive=True,
            )
    else:
        if "min_repeat_throughput_fps" in record:
            raise ValueError("min_repeat_throughput_fps: valid only for batch")
        if "min_sustained_fps" in record:
            normalized["min_sustained_fps"] = canonical_decimal_v1(
                record["min_sustained_fps"], field="min_sustained_fps", positive=True
            )

    quality = None
    if "quality_requirement" in record:
        quality = _quality_requirement(record["quality_requirement"])
        normalized["quality_requirement"] = quality
    if ranking == "accuracy_first" and quality is None:
        raise ValueError("quality_requirement: accuracy_first requires a complete quality object")

    local_job_digest = canonical_sha256_v1(normalized)
    return ImageJobSpec(_record=normalized, local_job_digest=local_job_digest)


def _prompt_bucket(max_codepoints: int) -> str:
    if max_codepoints <= 16:
        return "1-16"
    if max_codepoints <= 32:
        return "17-32"
    if max_codepoints <= 64:
        return "33-64"
    if max_codepoints <= 128:
        return "65-128"
    return "129-256"


def _bytes_bucket(total_bytes: int) -> str:
    if total_bytes <= 256:
        return "1-256"
    if total_bytes <= 1024:
        return "257-1024"
    if total_bytes <= 2048:
        return "1025-2048"
    return "2049-4096"


def compute_workload_fingerprint(record: Mapping[str, Any]) -> str:
    """Compute the shareable workload fingerprint, omitting its own field."""

    normalized = _deepcopy_dict(record)
    normalized.pop("workload_fingerprint", None)
    for field in ("provider_allowlist", "precision_allowlist"):
        if isinstance(normalized.get(field), list):
            normalized[field] = sorted(normalized[field], key=lambda item: str(item).encode("utf-8"))
    return canonical_sha256_v1(normalized)


def build_qualification_workload_profile(
    job: ImageJobSpec,
    inventory: "DecodedInputInventory",
) -> QualificationWorkloadProfile:
    """Derive a workload profile without prompt text or input content identity."""

    job_record = job.to_dict()
    if inventory.input_mode != job_record["input_mode"]:
        raise ValueError("inventory.input_mode: does not match ImageJobSpec")
    if inventory.input_count > job_record["max_images"]:
        raise ValueError("inventory.input_count: exceeds ImageJobSpec.max_images")
    phrases = job.prompt_phrases
    total_prompt_bytes = sum(len(item.encode("utf-8")) for item in phrases)
    prompt_characteristics = {
        "mode": job_record["prompt_mode"],
        "count": len(phrases),
        "maximum_codepoint_bucket": _prompt_bucket(max(len(item) for item in phrases)),
        "total_utf8_byte_bucket": _bytes_bucket(total_prompt_bytes),
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "collector_id": "yolozu_qualification_workload",
        "collector_version": "1",
        "task": job_record["task"],
        "input_mode": job_record["input_mode"],
        "execution_mode": job_record["execution_mode"],
        "compute_policy": job_record["compute_policy"],
        "provider_allowlist": list(job_record["provider_allowlist"]),
        "precision_allowlist": list(job_record["precision_allowlist"]),
        "input_count": inventory.input_count,
        "input_order": inventory.input_order,
        "decoded_inputs": [item.to_workload_dict() for item in inventory.inputs],
        "decoder": {"id": inventory.decoder_id, "version": inventory.decoder_version},
        "batch_size": job_record["batch_size"],
        "concurrency": job_record["concurrency"],
        "max_results_per_image": job_record["max_results_per_image"],
        "latency_interval_id": LATENCY_INTERVAL_ID,
        "handoff": {
            "id": HANDOFF_ID,
            "version": HANDOFF_VERSION,
            "scratch_cap_bytes": HANDOFF_SCRATCH_CAP_BYTES,
            "max_mask_artifacts": HANDOFF_MAX_MASK_ARTIFACTS,
            "max_output_files": HANDOFF_MAX_OUTPUT_FILES,
            "max_output_bytes": HANDOFF_MAX_OUTPUT_BYTES,
        },
        "prompt_characteristics": prompt_characteristics,
    }
    if job_record["execution_mode"] == "soft_realtime":
        record["max_sustained_samples"] = MAX_SUSTAINED_SAMPLES
    quality = job_record.get("quality_requirement")
    if quality is not None:
        record["quality_identity"] = {
            "metric_id": quality["metric_id"],
            "direction": quality["direction"],
            "evaluation_dataset_id": quality["evaluation_dataset_id"],
            "evaluation_dataset_sha256": quality["evaluation_dataset_sha256"],
            "evaluation_protocol_sha256": quality["evaluation_protocol_sha256"],
            "evaluation_vocabulary_id": quality["evaluation_vocabulary_id"],
        }
    record["workload_fingerprint"] = compute_workload_fingerprint(record)
    return validate_qualification_workload_profile(record)


def _validate_workload_decoded_inputs(value: Any, *, input_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != input_count:
        raise ValueError("decoded_inputs: expected exactly input_count entries")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        record = _mapping(item, field=f"decoded_inputs[{index}]")
        _check_keys(
            record,
            field=f"decoded_inputs[{index}]",
            allowed=frozenset({"index", "width", "height", "color_mode", "orientation_policy"}),
            required=frozenset({"index", "width", "height", "color_mode", "orientation_policy"}),
        )
        if _exact_int(record["index"], field=f"decoded_inputs[{index}].index", minimum=0, maximum=99) != index:
            raise ValueError("decoded_inputs: indices must be consecutive and ordered")
        normalized.append(
            {
                "index": index,
                "width": _exact_int(
                    record["width"], field=f"decoded_inputs[{index}].width", minimum=1, maximum=16_384
                ),
                "height": _exact_int(
                    record["height"], field=f"decoded_inputs[{index}].height", minimum=1, maximum=16_384
                ),
                "color_mode": _enum(
                    record["color_mode"],
                    field=f"decoded_inputs[{index}].color_mode",
                    allowed=COLOR_MODES,
                ),
                "orientation_policy": _enum(
                    record["orientation_policy"],
                    field=f"decoded_inputs[{index}].orientation_policy",
                    allowed=frozenset({"exif_transpose_v1"}),
                ),
            }
        )
    return normalized


def validate_qualification_workload_profile(value: Any) -> QualificationWorkloadProfile:
    """Validate a shareable QualificationWorkloadProfile v1 object."""

    record = _mapping(value, field="QualificationWorkloadProfile")
    _check_keys(
        record,
        field="QualificationWorkloadProfile",
        allowed=_WORKLOAD_KEYS,
        required=_WORKLOAD_REQUIRED,
    )
    normalized: dict[str, Any] = {
        "schema_version": _exact_int(record["schema_version"], field="schema_version", minimum=1, maximum=1),
        "collector_id": _identifier(record["collector_id"], field="collector_id", short=True),
        "collector_version": _identifier(record["collector_version"], field="collector_version"),
        "task": _enum(record["task"], field="task", allowed=TASKS),
        "input_mode": _enum(record["input_mode"], field="input_mode", allowed=INPUT_MODES),
        "execution_mode": _enum(record["execution_mode"], field="execution_mode", allowed=EXECUTION_MODES),
        "compute_policy": _enum(record["compute_policy"], field="compute_policy", allowed=COMPUTE_POLICIES),
        "provider_allowlist": _sorted_unique_ids(
            record["provider_allowlist"], field="provider_allowlist", maximum=8
        ),
        "precision_allowlist": _sorted_unique_ids(
            record["precision_allowlist"],
            field="precision_allowlist",
            maximum=len(PRECISIONS),
            allowed=PRECISIONS,
        ),
        "input_count": _exact_int(record["input_count"], field="input_count", minimum=1, maximum=100),
        "input_order": _enum(
            record["input_order"],
            field="input_order",
            allowed=frozenset({"single_image_v1", "normalized_basename_utf8_v1"}),
        ),
        "batch_size": _exact_int(record["batch_size"], field="batch_size", minimum=1, maximum=1),
        "concurrency": _exact_int(record["concurrency"], field="concurrency", minimum=1, maximum=1),
        "max_results_per_image": _exact_int(
            record["max_results_per_image"], field="max_results_per_image", minimum=1, maximum=1_000
        ),
    }
    if normalized["collector_id"] != "yolozu_qualification_workload":
        raise ValueError("collector_id: unsupported workload collector")
    if normalized["collector_version"] != "1":
        raise ValueError("collector_version: unsupported workload collector version")
    if normalized["input_mode"] == "single_image" and normalized["input_count"] != 1:
        raise ValueError("input_count: single_image requires exactly one input")
    expected_order = (
        "single_image_v1"
        if normalized["input_mode"] == "single_image"
        else "normalized_basename_utf8_v1"
    )
    if normalized["input_order"] != expected_order:
        raise ValueError("input_order: does not match input_mode")
    normalized["decoded_inputs"] = _validate_workload_decoded_inputs(
        record["decoded_inputs"], input_count=normalized["input_count"]
    )

    decoder = _mapping(record["decoder"], field="decoder")
    _check_keys(
        decoder,
        field="decoder",
        allowed=frozenset({"id", "version"}),
        required=frozenset({"id", "version"}),
    )
    normalized["decoder"] = {
        "id": _identifier(decoder["id"], field="decoder.id", short=True),
        "version": _identifier(decoder["version"], field="decoder.version"),
    }
    if normalized["decoder"]["id"] != "pillow":
        raise ValueError("decoder.id: unsupported decoder")
    if record["latency_interval_id"] != LATENCY_INTERVAL_ID:
        raise ValueError("latency_interval_id: unsupported interval")
    normalized["latency_interval_id"] = LATENCY_INTERVAL_ID

    handoff = _mapping(record["handoff"], field="handoff")
    expected_handoff = {
        "id": HANDOFF_ID,
        "version": HANDOFF_VERSION,
        "scratch_cap_bytes": HANDOFF_SCRATCH_CAP_BYTES,
        "max_mask_artifacts": HANDOFF_MAX_MASK_ARTIFACTS,
        "max_output_files": HANDOFF_MAX_OUTPUT_FILES,
        "max_output_bytes": HANDOFF_MAX_OUTPUT_BYTES,
    }
    _check_keys(
        handoff,
        field="handoff",
        allowed=frozenset(expected_handoff),
        required=frozenset(expected_handoff),
    )
    if handoff != expected_handoff:
        raise ValueError("handoff: values must match the code-owned v1 handoff")
    normalized["handoff"] = expected_handoff

    prompt = _mapping(record["prompt_characteristics"], field="prompt_characteristics")
    _check_keys(
        prompt,
        field="prompt_characteristics",
        allowed=frozenset({"mode", "count", "maximum_codepoint_bucket", "total_utf8_byte_bucket"}),
        required=frozenset({"mode", "count", "maximum_codepoint_bucket", "total_utf8_byte_bucket"}),
    )
    normalized["prompt_characteristics"] = {
        "mode": _enum(prompt["mode"], field="prompt_characteristics.mode", allowed=PROMPT_MODES),
        "count": _exact_int(prompt["count"], field="prompt_characteristics.count", minimum=1, maximum=128),
        "maximum_codepoint_bucket": _enum(
            prompt["maximum_codepoint_bucket"],
            field="prompt_characteristics.maximum_codepoint_bucket",
            allowed=frozenset({"1-16", "17-32", "33-64", "65-128", "129-256"}),
        ),
        "total_utf8_byte_bucket": _enum(
            prompt["total_utf8_byte_bucket"],
            field="prompt_characteristics.total_utf8_byte_bucket",
            allowed=frozenset({"1-256", "257-1024", "1025-2048", "2049-4096"}),
        ),
    }

    execution_mode = normalized["execution_mode"]
    if execution_mode == "soft_realtime":
        if record.get("max_sustained_samples") != MAX_SUSTAINED_SAMPLES:
            raise ValueError("max_sustained_samples: soft_realtime requires 1000000")
        normalized["max_sustained_samples"] = MAX_SUSTAINED_SAMPLES
    elif "max_sustained_samples" in record:
        raise ValueError("max_sustained_samples: valid only for soft_realtime")

    if "quality_identity" in record:
        quality = _mapping(record["quality_identity"], field="quality_identity")
        expected_quality_keys = _QUALITY_KEYS - {"threshold"}
        _check_keys(
            quality,
            field="quality_identity",
            allowed=expected_quality_keys,
            required=expected_quality_keys,
        )
        normalized["quality_identity"] = {
            "metric_id": _identifier(quality["metric_id"], field="quality_identity.metric_id"),
            "direction": _enum(
                quality["direction"],
                field="quality_identity.direction",
                allowed=frozenset({"higher_is_better", "lower_is_better"}),
            ),
            "evaluation_dataset_id": _identifier(
                quality["evaluation_dataset_id"], field="quality_identity.evaluation_dataset_id"
            ),
            "evaluation_dataset_sha256": _sha256(
                quality["evaluation_dataset_sha256"],
                field="quality_identity.evaluation_dataset_sha256",
            ),
            "evaluation_protocol_sha256": _sha256(
                quality["evaluation_protocol_sha256"],
                field="quality_identity.evaluation_protocol_sha256",
            ),
            "evaluation_vocabulary_id": _identifier(
                quality["evaluation_vocabulary_id"],
                field="quality_identity.evaluation_vocabulary_id",
            ),
        }

    claimed_fingerprint = _sha256(record["workload_fingerprint"], field="workload_fingerprint")
    normalized["workload_fingerprint"] = claimed_fingerprint
    expected_fingerprint = compute_workload_fingerprint(normalized)
    if claimed_fingerprint != expected_fingerprint:
        raise ValueError("workload_fingerprint: does not match canonical workload record")
    return QualificationWorkloadProfile(_record=normalized)


def _utc_instant(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise ValueError(f"{field}: expected exact RFC3339 UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{field}: invalid Gregorian UTC instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{field}: noncanonical UTC instant")
    return value


def _safe_fact_string(value: Any, *, field: str, maximum_bytes: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field}: expected bounded non-empty string")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{field}: control characters are invalid")
    if "/" in value or "\\" in value or _UUID_RE.search(value):
        raise ValueError(f"{field}: paths and UUIDs are invalid")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValueError(f"{field}: IP addresses are invalid")
    return value


def _probe_status(value: Any, *, field: str) -> str:
    return _enum(value, field=field, allowed=PROBE_STATUSES)


def _core_probe(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    _check_keys(
        record,
        field=field,
        allowed=frozenset({"probe_status", "value"}),
        required=frozenset({"probe_status"}),
    )
    status = _probe_status(record["probe_status"], field=f"{field}.probe_status")
    out: dict[str, Any] = {"probe_status": status}
    if status == "present":
        if "value" not in record:
            raise ValueError(f"{field}.value: required when present")
        out["value"] = _exact_int(record["value"], field=f"{field}.value", minimum=1, maximum=1_048_576)
    elif "value" in record:
        raise ValueError(f"{field}.value: forbidden when probe is not present")
    return out


def _os_probe(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="os")
    allowed = frozenset({"probe_status", "name", "version", "architecture"})
    _check_keys(record, field="os", allowed=allowed, required=frozenset({"probe_status"}))
    status = _probe_status(record["probe_status"], field="os.probe_status")
    out: dict[str, Any] = {"probe_status": status}
    fact_keys = ("name", "version", "architecture")
    if status == "present":
        if any(key not in record for key in fact_keys):
            raise ValueError("os: present requires name, version, and architecture")
        for key in fact_keys:
            out[key] = _safe_fact_string(record[key], field=f"os.{key}", maximum_bytes=128)
    elif any(key in record for key in fact_keys):
        raise ValueError("os: facts are forbidden when probe is not present")
    return out


def _cpu_probe(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="cpu")
    allowed = frozenset({"probe_status", "model", "logical_cores", "physical_cores"})
    _check_keys(record, field="cpu", allowed=allowed, required=frozenset({"probe_status"}))
    status = _probe_status(record["probe_status"], field="cpu.probe_status")
    out: dict[str, Any] = {"probe_status": status}
    fact_keys = ("model", "logical_cores", "physical_cores")
    if status == "present":
        if any(key not in record for key in fact_keys):
            raise ValueError("cpu: present requires model and both core probes")
        out["model"] = _safe_fact_string(record["model"], field="cpu.model", maximum_bytes=256)
        out["logical_cores"] = _core_probe(record["logical_cores"], field="cpu.logical_cores")
        out["physical_cores"] = _core_probe(record["physical_cores"], field="cpu.physical_cores")
    elif any(key in record for key in fact_keys):
        raise ValueError("cpu: facts are forbidden when probe is not present")
    return out


def _memory_probe(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    _check_keys(
        record,
        field=field,
        allowed=frozenset({"probe_status", "value_bytes"}),
        required=frozenset({"probe_status"}),
    )
    status = _probe_status(record["probe_status"], field=f"{field}.probe_status")
    out: dict[str, Any] = {"probe_status": status}
    if status == "present":
        if "value_bytes" not in record:
            raise ValueError(f"{field}.value_bytes: required when present")
        out["value_bytes"] = _exact_int(
            record["value_bytes"], field=f"{field}.value_bytes", minimum=1, maximum=9_223_372_036_854_775_807
        )
    elif "value_bytes" in record:
        raise ValueError(f"{field}.value_bytes: forbidden when probe is not present")
    return out


def _accelerator_probe(value: Any, *, index: int) -> dict[str, Any]:
    field = f"accelerators[{index}]"
    record = _mapping(value, field=field)
    fact_keys = frozenset({"kind", "vendor", "model", "device_count", "memory"})
    _check_keys(
        record,
        field=field,
        allowed=frozenset({"accelerator_id", "probe_status"}) | fact_keys,
        required=frozenset({"accelerator_id", "probe_status"}),
    )
    status = _probe_status(record["probe_status"], field=f"{field}.probe_status")
    out: dict[str, Any] = {
        "accelerator_id": _identifier(record["accelerator_id"], field=f"{field}.accelerator_id", short=True),
        "probe_status": status,
    }
    if status == "present":
        if any(key not in record for key in fact_keys):
            raise ValueError(f"{field}: present requires all accelerator facts")
        out.update(
            {
                "kind": _identifier(record["kind"], field=f"{field}.kind", short=True),
                "vendor": _safe_fact_string(record["vendor"], field=f"{field}.vendor", maximum_bytes=128),
                "model": _safe_fact_string(record["model"], field=f"{field}.model", maximum_bytes=256),
                "device_count": _exact_int(
                    record["device_count"], field=f"{field}.device_count", minimum=1, maximum=256
                ),
                "memory": _memory_probe(record["memory"], field=f"{field}.memory"),
            }
        )
    elif any(key in record for key in fact_keys):
        raise ValueError(f"{field}: facts are forbidden when probe is not present")
    return out


def _runtime_probe(value: Any, *, index: int) -> dict[str, Any]:
    field = f"runtimes[{index}]"
    record = _mapping(value, field=field)
    fact_keys = frozenset({"version", "provider_ids"})
    _check_keys(
        record,
        field=field,
        allowed=frozenset({"runtime_id", "probe_status"}) | fact_keys,
        required=frozenset({"runtime_id", "probe_status"}),
    )
    status = _probe_status(record["probe_status"], field=f"{field}.probe_status")
    out: dict[str, Any] = {
        "runtime_id": _identifier(record["runtime_id"], field=f"{field}.runtime_id", short=True),
        "probe_status": status,
    }
    if status == "present":
        if any(key not in record for key in fact_keys):
            raise ValueError(f"{field}: present requires version and provider_ids")
        out["version"] = _safe_fact_string(record["version"], field=f"{field}.version", maximum_bytes=128)
        out["provider_ids"] = _sorted_unique_ids(
            record["provider_ids"], field=f"{field}.provider_ids", maximum=32
        )
    elif any(key in record for key in fact_keys):
        raise ValueError(f"{field}: facts are forbidden when probe is not present")
    return out


def _power_probe(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="power_performance_mode")
    _check_keys(
        record,
        field="power_performance_mode",
        allowed=frozenset({"probe_status", "mode"}),
        required=frozenset({"probe_status"}),
    )
    status = _probe_status(record["probe_status"], field="power_performance_mode.probe_status")
    out: dict[str, Any] = {"probe_status": status}
    if status == "present":
        if "mode" not in record:
            raise ValueError("power_performance_mode.mode: required when present")
        out["mode"] = _identifier(record["mode"], field="power_performance_mode.mode", short=True)
    elif "mode" in record:
        raise ValueError("power_performance_mode.mode: forbidden when probe is not present")
    return out


def _probe_issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("probe_issues: expected at most 32 entries")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        field = f"probe_issues[{index}]"
        record = _mapping(item, field=field)
        _check_keys(
            record,
            field=field,
            allowed=frozenset({"probe_id", "status", "code"}),
            required=frozenset({"probe_id", "status", "code"}),
        )
        status = _enum(
            record["status"], field=f"{field}.status", allowed=frozenset({"unsupported", "failed"})
        )
        out.append(
            {
                "probe_id": _identifier(record["probe_id"], field=f"{field}.probe_id", short=True),
                "status": status,
                "code": _identifier(record["code"], field=f"{field}.code", short=True),
            }
        )
    out.sort(key=lambda item: (item["probe_id"].encode("utf-8"), item["code"].encode("utf-8")))
    if len({(item["probe_id"], item["code"]) for item in out}) != len(out):
        raise ValueError("probe_issues: duplicate probe/code pairs are invalid")
    return out


def _environment_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "collector_id": record["collector_id"],
        "collector_version": record["collector_version"],
        "os": record["os"],
        "cpu": record["cpu"],
        "total_memory": record["total_memory"],
        "accelerators": record["accelerators"],
        "runtimes": record["runtimes"],
        "power_performance_mode": record["power_performance_mode"],
    }


def compute_environment_fingerprint(record: Mapping[str, Any]) -> str:
    """Hash only performance-relevant privacy-safe environment facts."""

    identity = _environment_identity(record)
    identity["accelerators"] = sorted(
        copy.deepcopy(identity["accelerators"]),
        key=lambda item: str(item.get("accelerator_id", "")).encode("utf-8"),
    )
    identity["runtimes"] = sorted(
        copy.deepcopy(identity["runtimes"]),
        key=lambda item: str(item.get("runtime_id", "")).encode("utf-8"),
    )
    for runtime in identity["runtimes"]:
        if isinstance(runtime.get("provider_ids"), list):
            runtime["provider_ids"] = sorted(
                runtime["provider_ids"], key=lambda item: str(item).encode("utf-8")
            )
    return canonical_sha256_v1(identity)


def validate_environment_profile(value: Any) -> EnvironmentProfile:
    """Validate one privacy-safe EnvironmentProfile v1 object."""

    record = _mapping(value, field="EnvironmentProfile")
    _check_keys(
        record,
        field="EnvironmentProfile",
        allowed=_ENVIRONMENT_KEYS,
        required=_ENVIRONMENT_KEYS,
    )
    normalized: dict[str, Any] = {
        "schema_version": _exact_int(record["schema_version"], field="schema_version", minimum=1, maximum=1),
        "collector_id": _identifier(record["collector_id"], field="collector_id", short=True),
        "collector_version": _identifier(record["collector_version"], field="collector_version"),
        "collected_at": _utc_instant(record["collected_at"], field="collected_at"),
        "os": _os_probe(record["os"]),
        "cpu": _cpu_probe(record["cpu"]),
        "total_memory": _memory_probe(record["total_memory"], field="total_memory"),
        "power_performance_mode": _power_probe(record["power_performance_mode"]),
        "probe_issues": _probe_issues(record["probe_issues"]),
    }
    accelerators_raw = record["accelerators"]
    if not isinstance(accelerators_raw, list) or len(accelerators_raw) > 32:
        raise ValueError("accelerators: expected at most 32 entries")
    accelerators = [
        _accelerator_probe(item, index=index) for index, item in enumerate(accelerators_raw)
    ]
    accelerators.sort(key=lambda item: item["accelerator_id"].encode("utf-8"))
    if len({item["accelerator_id"] for item in accelerators}) != len(accelerators):
        raise ValueError("accelerators: duplicate accelerator_id")
    normalized["accelerators"] = accelerators

    runtimes_raw = record["runtimes"]
    if not isinstance(runtimes_raw, list) or len(runtimes_raw) > 64:
        raise ValueError("runtimes: expected at most 64 entries")
    runtimes = [_runtime_probe(item, index=index) for index, item in enumerate(runtimes_raw)]
    runtimes.sort(key=lambda item: item["runtime_id"].encode("utf-8"))
    if len({item["runtime_id"] for item in runtimes}) != len(runtimes):
        raise ValueError("runtimes: duplicate runtime_id")
    normalized["runtimes"] = runtimes

    claimed_fingerprint = _sha256(
        record["environment_fingerprint"], field="environment_fingerprint"
    )
    normalized["environment_fingerprint"] = claimed_fingerprint
    if claimed_fingerprint != compute_environment_fingerprint(normalized):
        raise ValueError("environment_fingerprint: does not match canonical environment facts")
    return EnvironmentProfile(_record=normalized)
