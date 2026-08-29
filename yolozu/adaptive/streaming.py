"""Strict, contract-only records for bounded local image streams.

This module intentionally contains no decoder, camera provider, queue, runner, or
stream loop.  It validates the immutable records that those later components must
consume and produce.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from math import gcd
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_decimal_v1, canonical_json_v1, canonical_sha256_v1
from .control_records import load_bounded_json_bytes

__all__ = [
    "FrameResult",
    "StreamContractError",
    "StreamJobSpec",
    "StreamQualificationReport",
    "StreamSelectionDecision",
    "StreamSummary",
    "StreamWorkloadProfile",
    "build_stream_workload_profile",
    "compute_max_consecutive_drops",
    "compute_stream_source_digest",
    "drop_fraction_within_limit",
    "sustained_fps_meets_minimum",
    "validate_frame_result",
    "validate_stream_job_spec",
    "validate_stream_output_artifacts",
    "validate_stream_qualification_report",
    "validate_stream_selection_decision",
    "validate_stream_summary",
    "validate_stream_workload_profile",
]


MAX_UINT64 = 18_446_744_073_709_551_615
MAX_FRAMES = 864_000
MAX_RESULTS_PER_FRAME = 1_000
MAX_TOTAL_RESULTS = 1_000_000
MAX_MASK_ARTIFACTS = 10_000
MAX_OUTPUT_FILES = 10_004
MAX_OUTPUT_BYTES = 4_294_967_296
MAX_MASK_BYTES = 67_108_864
MASK_CHUNK_BYTES = 1_048_576
CALLBACK_MAX_ITEMS = 64
CALLBACK_MAX_BYTES = 67_108_864
MIN_SUSTAINED_DURATION_NS = 600_000_000_000

SOURCE_ADMISSION_POLICY_ID = "rational_monotonic_due_v1"
OUTPUT_CADENCE_ID = "one_frame_result_per_processed_frame_v1"
LATENCY_INTERVAL_ID = "stream_due_to_callback_enqueue_v1"
DECODED_LAYOUT_ID = "uint8_rgb_bgr_strided_v1"
CAMERA_POOL_POLICY_ID = "caller_preallocated_fixed_pool_v1"
MASK_ENCODING_ID = "png_binary_mask_v1"
MEMORY_SCOPE = "whole_stream_job_process_tree_v1"
STRIDE_POLICY_ID = "width_times_3_to_4_v1"
CAMERA_ELIGIBILITY_POLICY_ID = "exactly_one_reenumerate_before_open_v1"
CAMERA_PROVIDER_ALLOWLIST = frozenset({"contract_fixture_camera_v1"})

_FROZEN_TIMEOUTS = {
    "source_probe_timeout_seconds": 10,
    "camera_open_timeout_seconds": 10,
    "camera_first_frame_timeout_seconds": 10,
    "runner_probe_timeout_seconds": 30,
    "runner_load_timeout_seconds": 600,
    "frame_decode_timeout_seconds": 5,
    "frame_predict_timeout_seconds": 30,
    "output_step_timeout_seconds": 30,
    "close_timeout_seconds": 10,
    "cancellation_grace_seconds": 5,
}

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z", re.ASCII)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z",
    re.ASCII,
)
_MASK_PATH_RE = re.compile(
    r"artifacts/masks/[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.png\Z", re.ASCII
)
_CHECKSUMS_NAME = "checksums.json"
_REQUIRED_STREAM_OUTPUTS = frozenset(
    {"provenance.json", "stream_results.jsonl", "stream_summary.json"}
)


class StreamContractError(ValueError):
    """Stable fail-closed stream interface-contract violation."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamContractError(f"{field}: expected object")
    return dict(value)


def _keys(
    value: Mapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str] | None = None,
) -> None:
    required = allowed if required is None else required
    if set(value) - allowed:
        raise StreamContractError(f"{field}: unknown keys")
    if required - set(value):
        raise StreamContractError(f"{field}: missing required keys")


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamContractError(f"{field}: expected integer")
    if value < minimum or value > maximum:
        raise StreamContractError(f"{field}: out of range")
    return value


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise StreamContractError(f"{field}: invalid identifier")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StreamContractError(f"{field}: expected lowercase SHA-256")
    return value


def _enum(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StreamContractError(f"{field}: unsupported value")
    return value


def _decimal(
    value: Any,
    *,
    field: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    positive: bool = False,
) -> str:
    try:
        token = canonical_decimal_v1(
            value,
            field=field,
            positive=positive,
            nonnegative=minimum is not None and minimum >= 0,
        )
    except ValueError as exc:
        raise StreamContractError(str(exc)) from exc
    number = Decimal(token)
    if minimum is not None and number < minimum:
        raise StreamContractError(f"{field}: below minimum")
    if maximum is not None and number > maximum:
        raise StreamContractError(f"{field}: above maximum")
    return token


def _utc(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise StreamContractError(f"{field}: expected exact RFC3339 UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise StreamContractError(f"{field}: invalid UTC instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise StreamContractError(f"{field}: invalid UTC instant")
    return value


def _reduced_ratio(
    numerator: Any,
    denominator: Any,
    *,
    field: str,
    maximum_numerator: int,
    maximum_denominator: int,
) -> tuple[int, int]:
    num = _integer(
        numerator, field=f"{field}_num", minimum=0, maximum=maximum_numerator
    )
    den = _integer(
        denominator, field=f"{field}_den", minimum=1, maximum=maximum_denominator
    )
    if gcd(num, den) != 1:
        raise StreamContractError(f"{field}: ratio must be reduced")
    if num == 0 and den != 1:
        raise StreamContractError(f"{field}: zero is encoded exactly as 0/1")
    return num, den


def _source_rate(numerator: Any, denominator: Any) -> tuple[int, int]:
    num = _integer(
        numerator, field="source_rate_num", minimum=1, maximum=240_000
    )
    den = _integer(
        denominator, field="source_rate_den", minimum=1, maximum=1_001
    )
    if gcd(num, den) != 1 or num * 10 < den or num > 240 * den:
        raise StreamContractError("source rate must be reduced and within 0.1..240")
    return num, den


def _ratio_display(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise StreamContractError("display ratio denominator must be positive")
    with localcontext() as context:
        context.prec = 80
        rounded = (Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_EVEN
        )
    token = format(rounded, "f").rstrip("0").rstrip(".")
    return "0" if token in {"", "-0"} else token


@dataclass(frozen=True)
class _Record:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)


@dataclass(frozen=True)
class StreamJobSpec(_Record):
    local_job_digest: str


@dataclass(frozen=True)
class StreamWorkloadProfile(_Record):
    @property
    def workload_fingerprint(self) -> str:
        return str(self._record["workload_fingerprint"])


@dataclass(frozen=True)
class FrameResult(_Record):
    def canonical_bytes(self) -> bytes:
        return canonical_json_v1(self._record)

    def canonical_line(self) -> bytes:
        return self.canonical_bytes() + b"\n"


@dataclass(frozen=True)
class StreamSummary(_Record):
    pass


@dataclass(frozen=True)
class StreamQualificationReport(_Record):
    pass


@dataclass(frozen=True)
class StreamSelectionDecision(_Record):
    pass


@dataclass(frozen=True)
class _ChecksumEntry:
    path: str
    size_bytes: int
    sha256: str


def _normalized_bbox(value: Any, *, field: str) -> dict[str, str]:
    record = _mapping(value, field=field)
    names = frozenset({"x1", "y1", "x2", "y2"})
    _keys(record, field=field, allowed=names)
    checked = {
        name: _decimal(
            record[name], field=f"{field}.{name}", minimum=Decimal(0), maximum=Decimal(1)
        )
        for name in ("x1", "y1", "x2", "y2")
    }
    if Decimal(checked["x1"]) >= Decimal(checked["x2"]):
        raise StreamContractError(f"{field}: x1 must be less than x2")
    if Decimal(checked["y1"]) >= Decimal(checked["y2"]):
        raise StreamContractError(f"{field}: y1 must be less than y2")
    return checked


def _mask_reference(
    value: Any, *, field: str, width: int, height: int
) -> dict[str, Any] | None:
    if value is None:
        return None
    record = _mapping(value, field=field)
    names = frozenset(
        {"relative_path", "sha256", "size_bytes", "width", "height", "encoding_id"}
    )
    _keys(record, field=field, allowed=names)
    path = record["relative_path"]
    if not isinstance(path, str) or not _MASK_PATH_RE.fullmatch(path):
        raise StreamContractError(f"{field}.relative_path: invalid managed mask path")
    mask_width = _integer(record["width"], field=f"{field}.width", minimum=1, maximum=8_192)
    mask_height = _integer(
        record["height"], field=f"{field}.height", minimum=1, maximum=8_192
    )
    if mask_width != width or mask_height != height:
        raise StreamContractError(f"{field}: mask dimensions must match the frame")
    if record["encoding_id"] != MASK_ENCODING_ID:
        raise StreamContractError(f"{field}.encoding_id: unsupported mask encoding")
    return {
        "relative_path": path,
        "sha256": _sha256(record["sha256"], field=f"{field}.sha256"),
        "size_bytes": _integer(
            record["size_bytes"], field=f"{field}.size_bytes", minimum=1, maximum=MAX_MASK_BYTES
        ),
        "width": mask_width,
        "height": mask_height,
        "encoding_id": MASK_ENCODING_ID,
    }


def validate_frame_result(
    payload: Mapping[str, Any],
    *,
    source_rate_num: int | None = None,
    source_rate_den: int | None = None,
    expected_task: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> FrameResult:
    """Validate one canonical processed-frame row and verify its own digest."""

    record = _mapping(payload, field="frame_result")
    names = frozenset(
        {
            "schema_version",
            "source_frame_index",
            "scheduled_due_offset_num_ns",
            "scheduled_due_offset_den",
            "processing_completed_offset_ns",
            "device_timestamp_ns",
            "task",
            "decoded_width",
            "decoded_height",
            "task_results",
            "frame_result_digest",
        }
    )
    _keys(record, field="frame_result", allowed=names)
    if record["schema_version"] != 1:
        raise StreamContractError("frame_result.schema_version: expected 1")
    frame_index = _integer(
        record["source_frame_index"],
        field="frame_result.source_frame_index",
        minimum=0,
        maximum=MAX_FRAMES - 1,
    )
    due_num, due_den = _reduced_ratio(
        record["scheduled_due_offset_num_ns"],
        record["scheduled_due_offset_den"],
        field="scheduled_due_offset",
        maximum_numerator=MAX_UINT64,
        maximum_denominator=240_000,
    )
    if (source_rate_num is None) != (source_rate_den is None):
        raise StreamContractError("source rate context requires numerator and denominator")
    if source_rate_num is not None and source_rate_den is not None:
        rate_num, rate_den = _source_rate(source_rate_num, source_rate_den)
        expected_num = frame_index * rate_den * 1_000_000_000
        expected_den = rate_num
        divisor = gcd(expected_num, expected_den)
        expected = (expected_num // divisor, expected_den // divisor)
        if (due_num, due_den) != expected:
            raise StreamContractError("frame_result: scheduled due offset does not match source rate")
    completed = _integer(
        record["processing_completed_offset_ns"],
        field="frame_result.processing_completed_offset_ns",
        minimum=0,
        maximum=MAX_UINT64,
    )
    if completed * due_den < due_num:
        raise StreamContractError("frame_result: completion precedes the governed due time")
    device_timestamp = record["device_timestamp_ns"]
    if device_timestamp is not None:
        device_timestamp = _integer(
            device_timestamp,
            field="frame_result.device_timestamp_ns",
            minimum=0,
            maximum=MAX_UINT64,
        )
    task = _enum(
        record["task"],
        field="frame_result.task",
        allowed=frozenset({"object_detection", "instance_segmentation"}),
    )
    if expected_task is not None and task != expected_task:
        raise StreamContractError("frame_result: task does not match the stream")
    width = _integer(
        record["decoded_width"], field="frame_result.decoded_width", minimum=1, maximum=8_192
    )
    height = _integer(
        record["decoded_height"],
        field="frame_result.decoded_height",
        minimum=1,
        maximum=8_192,
    )
    if width * height > 33_177_600:
        raise StreamContractError("frame_result: decoded pixel cap exceeded")
    if expected_width is not None and width != expected_width:
        raise StreamContractError("frame_result: decoded width mismatch")
    if expected_height is not None and height != expected_height:
        raise StreamContractError("frame_result: decoded height mismatch")
    raw_results = record["task_results"]
    if not isinstance(raw_results, list) or len(raw_results) > MAX_RESULTS_PER_FRAME:
        raise StreamContractError("frame_result.task_results: result cap exceeded")
    results: list[dict[str, Any]] = []
    mask_paths: set[str] = set()
    for index, raw in enumerate(raw_results):
        field = f"frame_result.task_results[{index}]"
        item = _mapping(raw, field=field)
        item_names = frozenset({"class_id", "score", "bbox", "mask"})
        _keys(item, field=field, allowed=item_names)
        mask = _mask_reference(item["mask"], field=f"{field}.mask", width=width, height=height)
        if task == "object_detection" and mask is not None:
            raise StreamContractError(f"{field}.mask: detection results require null")
        if task == "instance_segmentation" and mask is None:
            raise StreamContractError(f"{field}.mask: segmentation results require a mask")
        if mask is not None and mask["relative_path"] in mask_paths:
            raise StreamContractError("frame_result: duplicate mask reference")
        if mask is not None:
            mask_paths.add(mask["relative_path"])
        results.append(
            {
                "class_id": _integer(
                    item["class_id"], field=f"{field}.class_id", minimum=0, maximum=2_147_483_647
                ),
                "score": _decimal(
                    item["score"], field=f"{field}.score", minimum=Decimal(0), maximum=Decimal(1)
                ),
                "bbox": _normalized_bbox(item["bbox"], field=f"{field}.bbox"),
                "mask": mask,
            }
        )
    checked = {
        "schema_version": 1,
        "source_frame_index": frame_index,
        "scheduled_due_offset_num_ns": due_num,
        "scheduled_due_offset_den": due_den,
        "processing_completed_offset_ns": completed,
        "device_timestamp_ns": device_timestamp,
        "task": task,
        "decoded_width": width,
        "decoded_height": height,
        "task_results": results,
        "frame_result_digest": _sha256(
            record["frame_result_digest"], field="frame_result.frame_result_digest"
        ),
    }
    expected_digest = canonical_sha256_v1(
        checked, own_digest_field="frame_result_digest"
    )
    if checked["frame_result_digest"] != expected_digest:
        raise StreamContractError("frame_result: digest mismatch")
    if len(canonical_json_v1(checked)) > CALLBACK_MAX_BYTES:
        raise StreamContractError("frame_result: canonical byte cap exceeded")
    return FrameResult(checked)


def _source(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="source")
    names = frozenset(
        {
            "source_kind",
            "width",
            "height",
            "source_rate_num",
            "source_rate_den",
            "container",
            "codec",
            "pixel_format",
            "provider_id",
            "capability_format",
        }
    )
    _keys(record, field="source", allowed=names)
    kind = _enum(
        record["source_kind"],
        field="source.source_kind",
        allowed=frozenset({"local_mp4", "local_camera"}),
    )
    width = _integer(record["width"], field="source.width", minimum=1, maximum=8_192)
    height = _integer(
        record["height"], field="source.height", minimum=1, maximum=8_192
    )
    if width * height > 33_177_600:
        raise StreamContractError("source: decoded pixel cap exceeded")
    rate_num, rate_den = _source_rate(
        record["source_rate_num"], record["source_rate_den"]
    )
    if kind == "local_mp4":
        if (
            record["container"] != "mp4"
            or record["codec"] != "h264_avc"
            or record["pixel_format"] != "yuv420p8"
            or record["provider_id"] is not None
            or record["capability_format"] is not None
        ):
            raise StreamContractError("source: invalid bounded MP4 declaration")
        provider_id = capability_format = None
    else:
        if record["container"] is not None or record["codec"] is not None:
            raise StreamContractError("source: camera cannot declare a container or codec")
        pixel_format = _enum(
            record["pixel_format"],
            field="source.pixel_format",
            allowed=frozenset({"rgb24", "bgr24"}),
        )
        provider_id = _identifier(record["provider_id"], field="source.provider_id")
        if provider_id not in CAMERA_PROVIDER_ALLOWLIST:
            raise StreamContractError("source.provider_id: provider is not code-owned allowlisted")
        capability_format = _enum(
            record["capability_format"],
            field="source.capability_format",
            allowed=frozenset({"rgb24", "bgr24"}),
        )
        if pixel_format != capability_format:
            raise StreamContractError("source: camera format mismatch")
    return {
        "source_kind": kind,
        "width": width,
        "height": height,
        "source_rate_num": rate_num,
        "source_rate_den": rate_den,
        "container": "mp4" if kind == "local_mp4" else None,
        "codec": "h264_avc" if kind == "local_mp4" else None,
        "pixel_format": "yuv420p8" if kind == "local_mp4" else pixel_format,
        "provider_id": provider_id,
        "capability_format": capability_format,
    }


def _stream_source_preimage(
    value: Any, *, expected_source: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record = _mapping(value, field="stream_source_preimage")
    kind = _enum(
        record.get("source_kind"),
        field="stream_source_preimage.source_kind",
        allowed=frozenset({"local_mp4", "local_camera"}),
    )
    if kind == "local_mp4":
        names = frozenset({"source_kind", "byte_length", "source_sha256"})
        _keys(record, field="stream_source_preimage", allowed=names)
        checked = {
            "source_kind": "local_mp4",
            "byte_length": _integer(
                record["byte_length"],
                field="stream_source_preimage.byte_length",
                minimum=1,
                maximum=8_589_934_592,
            ),
            "source_sha256": _sha256(
                record["source_sha256"],
                field="stream_source_preimage.source_sha256",
            ),
        }
    else:
        names = frozenset(
            {
                "source_kind",
                "provider_id",
                "capability_format",
                "width",
                "height",
                "source_rate_num",
                "source_rate_den",
                "eligible_device_count",
            }
        )
        _keys(record, field="stream_source_preimage", allowed=names)
        provider = _identifier(
            record["provider_id"], field="stream_source_preimage.provider_id"
        )
        if provider not in CAMERA_PROVIDER_ALLOWLIST:
            raise StreamContractError(
                "stream_source_preimage.provider_id: provider is not allowlisted"
            )
        rate_num, rate_den = _source_rate(
            record["source_rate_num"], record["source_rate_den"]
        )
        checked = {
            "source_kind": "local_camera",
            "provider_id": provider,
            "capability_format": _enum(
                record["capability_format"],
                field="stream_source_preimage.capability_format",
                allowed=frozenset({"rgb24", "bgr24"}),
            ),
            "width": _integer(
                record["width"],
                field="stream_source_preimage.width",
                minimum=1,
                maximum=8_192,
            ),
            "height": _integer(
                record["height"],
                field="stream_source_preimage.height",
                minimum=1,
                maximum=8_192,
            ),
            "source_rate_num": rate_num,
            "source_rate_den": rate_den,
            "eligible_device_count": _integer(
                record["eligible_device_count"],
                field="stream_source_preimage.eligible_device_count",
                minimum=1,
                maximum=1,
            ),
        }
        if checked["width"] * checked["height"] > 33_177_600:
            raise StreamContractError("stream_source_preimage: decoded pixel cap exceeded")
    if expected_source is not None:
        source = _source(expected_source)
        if kind != source["source_kind"]:
            raise StreamContractError("stream_source_preimage: source kind mismatch")
        if kind == "local_camera":
            projection = {
                key: source[key]
                for key in (
                    "source_kind",
                    "provider_id",
                    "capability_format",
                    "width",
                    "height",
                    "source_rate_num",
                    "source_rate_den",
                )
            }
            observed_capability = {
                key: item
                for key, item in checked.items()
                if key != "eligible_device_count"
            }
            if observed_capability != projection:
                raise StreamContractError(
                    "stream_source_preimage: camera capability mismatch"
                )
    return checked


def compute_stream_source_digest(
    preimage: Mapping[str, Any], *, expected_source: Mapping[str, Any] | None = None
) -> str:
    """Derive the local-only digest from one typed, privacy-bounded preimage."""

    return canonical_sha256_v1(
        _stream_source_preimage(preimage, expected_source=expected_source)
    )


def _decoder_policy(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="decoder_policy")
    names = frozenset(
        {
            "policy_id",
            "policy_version",
            "backend_id",
            "backend_version",
            "enforcement_id",
            "allowed_profiles",
            "allowed_levels",
            "pixel_format",
            "max_reference_frames",
            "max_dpb_bytes",
            "max_gop_frames",
            "max_sample_bytes",
            "max_nal_unit_bytes",
            "probe_input_bytes",
            "probe_box_count",
            "probe_nesting_depth",
            "policy_digest",
        }
    )
    _keys(record, field="decoder_policy", allowed=names)
    profiles = ["constrained_baseline", "main", "high"]
    levels = ["3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0", "5.1"]
    fixed: dict[str, Any] = {
        "policy_id": "bounded_h264_decoder_v1",
        "policy_version": 1,
        "backend_id": _identifier(record["backend_id"], field="decoder_policy.backend_id"),
        "backend_version": _identifier(
            record["backend_version"], field="decoder_policy.backend_version"
        ),
        "enforcement_id": "bounded_process_tree_hard_limit_v1",
        "allowed_profiles": profiles,
        "allowed_levels": levels,
        "pixel_format": "yuv420p8",
        "max_reference_frames": 16,
        "max_dpb_bytes": 536_870_912,
        "max_gop_frames": 600,
        "max_sample_bytes": 67_108_864,
        "max_nal_unit_bytes": 33_554_432,
        "probe_input_bytes": 67_108_864,
        "probe_box_count": 100_000,
        "probe_nesting_depth": 32,
        "policy_digest": _sha256(
            record["policy_digest"], field="decoder_policy.policy_digest"
        ),
    }
    for key, expected in fixed.items():
        if key in {"backend_id", "backend_version", "policy_digest"}:
            continue
        if record[key] != expected:
            raise StreamContractError(f"decoder_policy.{key}: frozen v1 value mismatch")
    if fixed["policy_digest"] != canonical_sha256_v1(
        fixed, own_digest_field="policy_digest"
    ):
        raise StreamContractError("decoder_policy: digest mismatch")
    return fixed


def _memory_collector(value: Any) -> dict[str, str]:
    record = _mapping(value, field="memory_collector")
    names = frozenset({"collector_id", "collector_version", "source", "scope"})
    _keys(record, field="memory_collector", allowed=names)
    scope = record["scope"]
    if scope != MEMORY_SCOPE:
        raise StreamContractError("memory_collector.scope: incomplete scope")
    return {
        "collector_id": _identifier(
            record["collector_id"], field="memory_collector.collector_id"
        ),
        "collector_version": _identifier(
            record["collector_version"], field="memory_collector.collector_version"
        ),
        "source": _identifier(record["source"], field="memory_collector.source"),
        "scope": MEMORY_SCOPE,
    }


def _quality_requirement(value: Any, *, task: str) -> dict[str, Any] | None:
    if value is None:
        return None
    record = _mapping(value, field="quality_requirement")
    names = frozenset(
        {
            "metric_id",
            "direction",
            "threshold",
            "evaluation_dataset_id",
            "evaluation_dataset_sha256",
            "evaluation_protocol_sha256",
            "evaluation_vocabulary_id",
            "task",
        }
    )
    _keys(record, field="quality_requirement", allowed=names)
    if record["task"] != task:
        raise StreamContractError("quality_requirement.task: task mismatch")
    return {
        "metric_id": _identifier(record["metric_id"], field="quality_requirement.metric_id"),
        "direction": _enum(
            record["direction"],
            field="quality_requirement.direction",
            allowed=frozenset({"higher_is_better", "lower_is_better"}),
        ),
        "threshold": _decimal(record["threshold"], field="quality_requirement.threshold"),
        "evaluation_dataset_id": _identifier(
            record["evaluation_dataset_id"],
            field="quality_requirement.evaluation_dataset_id",
        ),
        "evaluation_dataset_sha256": _sha256(
            record["evaluation_dataset_sha256"],
            field="quality_requirement.evaluation_dataset_sha256",
        ),
        "evaluation_protocol_sha256": _sha256(
            record["evaluation_protocol_sha256"],
            field="quality_requirement.evaluation_protocol_sha256",
        ),
        "evaluation_vocabulary_id": _identifier(
            record["evaluation_vocabulary_id"],
            field="quality_requirement.evaluation_vocabulary_id",
        ),
        "task": task,
    }


def validate_stream_job_spec(payload: Mapping[str, Any]) -> StreamJobSpec:
    record = _mapping(payload, field="stream_job_spec")
    names = frozenset(
        {
            "schema_version",
            "task",
            "source",
            "required_duration_seconds",
            "warmup_frame_count",
            "min_sustained_fps",
            "max_p95_latency_ms",
            "queue_capacity_frames",
            "max_queued_decoded_bytes",
            "drop_policy",
            "max_drop_num",
            "max_drop_den",
            "max_consecutive_drops",
            "max_duration_seconds",
            "max_frames",
            "job_timeout_seconds",
            "max_results_per_frame",
            "max_total_results",
            "max_mask_artifacts",
            "max_output_files",
            "max_output_bytes",
            "max_mask_bytes",
            "mask_chunk_bytes",
            "callback_max_items",
            "callback_max_bytes",
            "max_decoder_rss_bytes",
            "max_stream_job_peak_rss_bytes",
            "max_accelerator_process_tree_peak_bytes",
            "camera_pool_capacity_frames",
            "camera_pool_bytes",
            "stride_policy_id",
            "decoded_stride_min_bytes",
            "decoded_stride_max_bytes",
            "source_admission_policy_id",
            "source_admission_policy_version",
            "output_cadence_id",
            "latency_interval_id",
            "decoded_layout_id",
            "camera_pool_policy_id",
            "camera_pool_policy_version",
            "camera_eligibility_policy_id",
            "mask_encoding_id",
            "mask_encoding_version",
            "decoder_policy",
            "memory_collector",
            "quality_requirement",
            "network_policy",
            *_FROZEN_TIMEOUTS,
        }
    )
    _keys(record, field="stream_job_spec", allowed=names)
    if record["schema_version"] != 1:
        raise StreamContractError("stream_job_spec.schema_version: expected 1")
    task = _enum(
        record["task"],
        field="stream_job_spec.task",
        allowed=frozenset({"object_detection", "instance_segmentation"}),
    )
    source = _source(record["source"])
    required_duration = _integer(
        record["required_duration_seconds"],
        field="required_duration_seconds",
        minimum=1,
        maximum=3_600,
    )
    warmup_frame_count = _integer(
        record["warmup_frame_count"],
        field="warmup_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    minimum_fps = _decimal(
        record["min_sustained_fps"],
        field="min_sustained_fps",
        minimum=Decimal("0.1"),
        maximum=Decimal(240),
    )
    maximum_p95 = _decimal(
        record["max_p95_latency_ms"], field="max_p95_latency_ms", positive=True
    )
    queue_capacity = _integer(
        record["queue_capacity_frames"],
        field="queue_capacity_frames",
        minimum=1,
        maximum=64,
    )
    queued_bytes = _integer(
        record["max_queued_decoded_bytes"],
        field="max_queued_decoded_bytes",
        minimum=1,
        maximum=536_870_912,
    )
    worst_case = source["width"] * 4 * source["height"]
    if worst_case > queued_bytes:
        raise StreamContractError("max_queued_decoded_bytes: one frame does not fit")
    camera_pool_capacity = record["camera_pool_capacity_frames"]
    camera_pool_bytes = record["camera_pool_bytes"]
    if source["source_kind"] == "local_mp4":
        if camera_pool_capacity is not None or camera_pool_bytes is not None:
            raise StreamContractError("MP4 source cannot declare a camera ingress pool")
    else:
        camera_pool_capacity = _integer(
            camera_pool_capacity,
            field="camera_pool_capacity_frames",
            minimum=1,
            maximum=64,
        )
        camera_pool_bytes = _integer(
            camera_pool_bytes,
            field="camera_pool_bytes",
            minimum=1,
            maximum=MAX_UINT64,
        )
        if camera_pool_capacity != queue_capacity:
            raise StreamContractError("camera pool count must equal queue capacity")
        if camera_pool_bytes != camera_pool_capacity * worst_case:
            raise StreamContractError("camera pool bytes must reserve every worst-case frame")
    drop_policy = _enum(
        record["drop_policy"],
        field="drop_policy",
        allowed=frozenset({"block", "drop_oldest"}),
    )
    drop_num, drop_den = _reduced_ratio(
        record["max_drop_num"],
        record["max_drop_den"],
        field="max_drop",
        maximum_numerator=1_000_000,
        maximum_denominator=1_000_000,
    )
    if drop_num > drop_den:
        raise StreamContractError("max_drop: numerator exceeds denominator")
    max_consecutive = _integer(
        record["max_consecutive_drops"],
        field="max_consecutive_drops",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    if drop_policy == "block" and ((drop_num, drop_den) != (0, 1) or max_consecutive != 0):
        raise StreamContractError("block policy requires zero drop gates")
    max_duration = _integer(
        record["max_duration_seconds"],
        field="max_duration_seconds",
        minimum=1,
        maximum=3_600,
    )
    timeout = _integer(
        record["job_timeout_seconds"],
        field="job_timeout_seconds",
        minimum=1,
        maximum=3_600,
    )
    warmup_duration_ceiling = (
        warmup_frame_count * source["source_rate_den"]
        + source["source_rate_num"]
        - 1
    ) // source["source_rate_num"]
    if (
        required_duration + warmup_duration_ceiling > max_duration
        or max_duration > timeout
    ):
        raise StreamContractError("stream duration exceeds its enclosing deadline")
    max_frames = _integer(
        record["max_frames"], field="max_frames", minimum=1, maximum=MAX_FRAMES
    )
    required_frames = (
        source["source_rate_num"] * required_duration
        + source["source_rate_den"]
        - 1
    ) // source["source_rate_den"]
    if max_frames < warmup_frame_count + required_frames:
        raise StreamContractError("max_frames cannot cover required duration without sampling")
    max_results = _integer(
        record["max_results_per_frame"],
        field="max_results_per_frame",
        minimum=1,
        maximum=MAX_RESULTS_PER_FRAME,
    )
    max_total_results = _integer(
        record["max_total_results"],
        field="max_total_results",
        minimum=1,
        maximum=MAX_TOTAL_RESULTS,
    )
    max_masks = _integer(
        record["max_mask_artifacts"],
        field="max_mask_artifacts",
        minimum=0,
        maximum=MAX_MASK_ARTIFACTS,
    )
    if task == "instance_segmentation" and max_masks == 0:
        raise StreamContractError("instance segmentation requires a positive mask cap")
    max_files = _integer(
        record["max_output_files"],
        field="max_output_files",
        minimum=4,
        maximum=MAX_OUTPUT_FILES,
    )
    if max_masks + 4 > max_files:
        raise StreamContractError("max_output_files cannot cover declared mask artifacts")
    optional_accelerator = record["max_accelerator_process_tree_peak_bytes"]
    if optional_accelerator is not None:
        optional_accelerator = _integer(
            optional_accelerator,
            field="max_accelerator_process_tree_peak_bytes",
            minimum=1,
            maximum=MAX_UINT64,
        )
    frozen = {
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_mask_bytes": MAX_MASK_BYTES,
        "mask_chunk_bytes": MASK_CHUNK_BYTES,
        "callback_max_items": CALLBACK_MAX_ITEMS,
        "callback_max_bytes": CALLBACK_MAX_BYTES,
        "source_admission_policy_id": SOURCE_ADMISSION_POLICY_ID,
        "source_admission_policy_version": 1,
        "output_cadence_id": OUTPUT_CADENCE_ID,
        "latency_interval_id": LATENCY_INTERVAL_ID,
        "decoded_layout_id": DECODED_LAYOUT_ID,
        "camera_pool_policy_id": CAMERA_POOL_POLICY_ID,
        "camera_pool_policy_version": 1,
        "camera_eligibility_policy_id": CAMERA_ELIGIBILITY_POLICY_ID,
        "mask_encoding_id": MASK_ENCODING_ID,
        "mask_encoding_version": 1,
        "network_policy": "deny",
        "stride_policy_id": STRIDE_POLICY_ID,
        **_FROZEN_TIMEOUTS,
    }
    for key, expected in frozen.items():
        if record[key] != expected:
            raise StreamContractError(f"stream_job_spec.{key}: frozen v1 value mismatch")
    checked: dict[str, Any] = {
        "schema_version": 1,
        "task": task,
        "source": source,
        "required_duration_seconds": required_duration,
        "warmup_frame_count": warmup_frame_count,
        "min_sustained_fps": minimum_fps,
        "max_p95_latency_ms": maximum_p95,
        "queue_capacity_frames": queue_capacity,
        "max_queued_decoded_bytes": queued_bytes,
        "drop_policy": drop_policy,
        "max_drop_num": drop_num,
        "max_drop_den": drop_den,
        "max_consecutive_drops": max_consecutive,
        "max_duration_seconds": max_duration,
        "max_frames": max_frames,
        "job_timeout_seconds": timeout,
        "max_results_per_frame": max_results,
        "max_total_results": max_total_results,
        "max_mask_artifacts": max_masks,
        "max_output_files": max_files,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_mask_bytes": MAX_MASK_BYTES,
        "mask_chunk_bytes": MASK_CHUNK_BYTES,
        "callback_max_items": CALLBACK_MAX_ITEMS,
        "callback_max_bytes": CALLBACK_MAX_BYTES,
        "max_decoder_rss_bytes": _integer(
            record["max_decoder_rss_bytes"],
            field="max_decoder_rss_bytes",
            minimum=67_108_864,
            maximum=2_147_483_648,
        ),
        "max_stream_job_peak_rss_bytes": _integer(
            record["max_stream_job_peak_rss_bytes"],
            field="max_stream_job_peak_rss_bytes",
            minimum=1,
            maximum=MAX_UINT64,
        ),
        "max_accelerator_process_tree_peak_bytes": optional_accelerator,
        "camera_pool_capacity_frames": camera_pool_capacity,
        "camera_pool_bytes": camera_pool_bytes,
        "stride_policy_id": STRIDE_POLICY_ID,
        "decoded_stride_min_bytes": source["width"] * 3,
        "decoded_stride_max_bytes": source["width"] * 4,
        **{key: frozen[key] for key in frozen if key != "network_policy"},
        "decoder_policy": _decoder_policy(record["decoder_policy"]),
        "memory_collector": _memory_collector(record["memory_collector"]),
        "quality_requirement": _quality_requirement(
            record["quality_requirement"], task=task
        ),
        "network_policy": "deny",
    }
    for key in ("decoded_stride_min_bytes", "decoded_stride_max_bytes"):
        if record[key] != checked[key]:
            raise StreamContractError(f"stream_job_spec.{key}: stride policy mismatch")
    if camera_pool_bytes is not None and camera_pool_bytes > checked["max_stream_job_peak_rss_bytes"]:
        raise StreamContractError("camera pool cannot fit the whole-job memory gate")
    return StreamJobSpec(checked, canonical_sha256_v1(checked))


def build_stream_workload_profile(
    job: StreamJobSpec | Mapping[str, Any],
    *,
    collector_id: str,
    collector_version: str,
) -> StreamWorkloadProfile:
    validated_job = job if isinstance(job, StreamJobSpec) else validate_stream_job_spec(job)
    record: dict[str, Any] = {
        "schema_version": 1,
        "collector_id": _identifier(collector_id, field="collector_id"),
        "collector_version": _identifier(collector_version, field="collector_version"),
        "stream_job": validated_job.to_dict(),
        "workload_fingerprint": "0" * 64,
    }
    record["workload_fingerprint"] = canonical_sha256_v1(
        record, own_digest_field="workload_fingerprint"
    )
    return StreamWorkloadProfile(record)


def validate_stream_workload_profile(
    payload: Mapping[str, Any],
) -> StreamWorkloadProfile:
    record = _mapping(payload, field="stream_workload_profile")
    names = frozenset(
        {"schema_version", "collector_id", "collector_version", "stream_job", "workload_fingerprint"}
    )
    _keys(record, field="stream_workload_profile", allowed=names)
    if record["schema_version"] != 1:
        raise StreamContractError("stream_workload_profile.schema_version: expected 1")
    checked = {
        "schema_version": 1,
        "collector_id": _identifier(record["collector_id"], field="collector_id"),
        "collector_version": _identifier(
            record["collector_version"], field="collector_version"
        ),
        "stream_job": validate_stream_job_spec(record["stream_job"]).to_dict(),
        "workload_fingerprint": _sha256(
            record["workload_fingerprint"], field="workload_fingerprint"
        ),
    }
    expected = canonical_sha256_v1(checked, own_digest_field="workload_fingerprint")
    if checked["workload_fingerprint"] != expected:
        raise StreamContractError("stream_workload_profile: fingerprint mismatch")
    return StreamWorkloadProfile(checked)


def compute_max_consecutive_drops(indices: list[int] | tuple[int, ...]) -> int:
    """Return the longest adjacent source-index run after strict validation."""

    previous = -2
    current = maximum = 0
    for position, value in enumerate(indices):
        index = _integer(
            value,
            field=f"dropped_indices[{position}]",
            minimum=0,
            maximum=MAX_FRAMES - 1,
        )
        if index <= previous:
            raise StreamContractError("dropped_indices: expected strict ascending order")
        current = current + 1 if index == previous + 1 else 1
        maximum = max(maximum, current)
        previous = index
    return maximum


def drop_fraction_within_limit(
    *,
    dropped_count: int,
    processed_count: int,
    maximum_numerator: int,
    maximum_denominator: int,
) -> bool:
    dropped = _integer(
        dropped_count, field="dropped_count", minimum=0, maximum=MAX_FRAMES
    )
    processed = _integer(
        processed_count, field="processed_count", minimum=0, maximum=MAX_FRAMES
    )
    numerator, denominator = _reduced_ratio(
        maximum_numerator,
        maximum_denominator,
        field="maximum_drop",
        maximum_numerator=1_000_000,
        maximum_denominator=1_000_000,
    )
    if numerator > denominator:
        raise StreamContractError("maximum_drop: numerator exceeds denominator")
    accounted = dropped + processed
    if accounted == 0:
        raise StreamContractError("drop fraction is unknown for zero accounted frames")
    return dropped * denominator <= numerator * accounted


def sustained_fps_meets_minimum(
    *, processed_count: int, duration_ns: int, minimum_fps: str
) -> bool:
    processed = _integer(
        processed_count, field="processed_count", minimum=1, maximum=MAX_FRAMES
    )
    duration = _integer(
        duration_ns, field="duration_ns", minimum=1, maximum=MAX_UINT64
    )
    token = _decimal(
        minimum_fps,
        field="minimum_fps",
        minimum=Decimal("0.1"),
        maximum=Decimal(240),
    )
    decimal = Decimal(token)
    sign, digits, exponent = decimal.as_tuple()
    if sign:
        raise StreamContractError("minimum_fps: expected nonnegative value")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    scale = 10 ** max(0, -exponent)
    if exponent > 0:
        coefficient *= 10**exponent
    return processed * 1_000_000_000 * scale >= coefficient * duration


_TERMINATION_REASONS = frozenset(
    {
        "normal_eof",
        "duration_cap",
        "frame_cap",
        "result_cap",
        "file_cap",
        "output_byte_cap",
        "source_probe_timeout",
        "camera_open_timeout",
        "camera_frame_timeout",
        "runner_probe_timeout",
        "runner_load_timeout",
        "decode_timeout",
        "predict_timeout",
        "output_timeout",
        "close_timeout",
        "cancelled",
        "cancellation_forced",
        "consumer_backpressure",
        "decoder_backend_unsupported",
        "decoder_sample_limit",
        "decoder_dpb_limit",
        "decoder_memory_limit",
        "camera_binding_ambiguous",
        "invalid_source",
        "runner_failure",
    }
)
_QUALIFIED_TERMINATIONS = frozenset({"normal_eof", "duration_cap", "frame_cap"})


def validate_stream_summary(payload: Mapping[str, Any]) -> StreamSummary:
    record = _mapping(payload, field="stream_summary")
    names = frozenset(
        {
            "schema_version",
            "status",
            "task",
            "source_kind",
            "scheduled_frame_count",
            "processed_frame_count",
            "dropped_frame_count",
            "failed_unaccounted_frame_count",
            "max_consecutive_drops",
            "frame_queue_count_high_watermark",
            "frame_queue_decoded_bytes_high_watermark",
            "callback_item_high_watermark",
            "callback_bytes_high_watermark",
            "duration_ns",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "drop_fraction_display",
            "result_count",
            "mask_artifact_count",
            "output_file_count",
            "output_bytes",
            "termination_reason",
            "bundle_spec_digest",
            "evidence_report_digest",
            "summary_digest",
        }
    )
    _keys(record, field="stream_summary", allowed=names)
    if record["schema_version"] != 1:
        raise StreamContractError("stream_summary.schema_version: expected 1")
    status = _enum(
        record["status"],
        field="stream_summary.status",
        allowed=frozenset({"completed", "failed"}),
    )
    task = _enum(
        record["task"],
        field="stream_summary.task",
        allowed=frozenset({"object_detection", "instance_segmentation"}),
    )
    source_kind = _enum(
        record["source_kind"],
        field="stream_summary.source_kind",
        allowed=frozenset({"local_mp4", "local_camera"}),
    )
    scheduled = _integer(
        record["scheduled_frame_count"],
        field="scheduled_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    processed = _integer(
        record["processed_frame_count"],
        field="processed_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    dropped = _integer(
        record["dropped_frame_count"],
        field="dropped_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    failed = _integer(
        record["failed_unaccounted_frame_count"],
        field="failed_unaccounted_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    if processed + dropped + failed != scheduled:
        raise StreamContractError("stream_summary: scheduled frames are not fully accounted")
    maximum_consecutive = _integer(
        record["max_consecutive_drops"],
        field="max_consecutive_drops",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    if maximum_consecutive > dropped or (dropped == 0 and maximum_consecutive != 0):
        raise StreamContractError("stream_summary: impossible consecutive-drop count")
    if failed and status != "failed":
        raise StreamContractError("stream_summary: failed-unaccounted frames require failed status")
    p_tokens: list[str | None] = []
    for name in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms"):
        value = record[name]
        p_tokens.append(
            None
            if value is None
            else _decimal(value, field=name, minimum=Decimal(0))
        )
    if processed == 0 and any(value is not None for value in p_tokens):
        raise StreamContractError("stream_summary: zero processed frames require unknown latency")
    if processed > 0:
        if any(value is None for value in p_tokens):
            raise StreamContractError("stream_summary: processed frames require latency percentiles")
        decimals = [Decimal(str(value)) for value in p_tokens]
        if not decimals[0] <= decimals[1] <= decimals[2]:
            raise StreamContractError("stream_summary: latency percentiles are unordered")
    accounted = processed + dropped
    display = record["drop_fraction_display"]
    expected_display = None if accounted == 0 else _ratio_display(dropped, accounted)
    if display != expected_display:
        raise StreamContractError("stream_summary.drop_fraction_display: not exact half-even display")
    checked = {
        "schema_version": 1,
        "status": status,
        "task": task,
        "source_kind": source_kind,
        "scheduled_frame_count": scheduled,
        "processed_frame_count": processed,
        "dropped_frame_count": dropped,
        "failed_unaccounted_frame_count": failed,
        "max_consecutive_drops": maximum_consecutive,
        "frame_queue_count_high_watermark": _integer(
            record["frame_queue_count_high_watermark"],
            field="frame_queue_count_high_watermark",
            minimum=0,
            maximum=64,
        ),
        "frame_queue_decoded_bytes_high_watermark": _integer(
            record["frame_queue_decoded_bytes_high_watermark"],
            field="frame_queue_decoded_bytes_high_watermark",
            minimum=0,
            maximum=536_870_912,
        ),
        "callback_item_high_watermark": _integer(
            record["callback_item_high_watermark"],
            field="callback_item_high_watermark",
            minimum=0,
            maximum=CALLBACK_MAX_ITEMS,
        ),
        "callback_bytes_high_watermark": _integer(
            record["callback_bytes_high_watermark"],
            field="callback_bytes_high_watermark",
            minimum=0,
            maximum=CALLBACK_MAX_BYTES,
        ),
        "duration_ns": _integer(
            record["duration_ns"], field="duration_ns", minimum=0, maximum=MAX_UINT64
        ),
        "p50_latency_ms": p_tokens[0],
        "p95_latency_ms": p_tokens[1],
        "p99_latency_ms": p_tokens[2],
        "drop_fraction_display": expected_display,
        "result_count": _integer(
            record["result_count"], field="result_count", minimum=0, maximum=MAX_TOTAL_RESULTS
        ),
        "mask_artifact_count": _integer(
            record["mask_artifact_count"],
            field="mask_artifact_count",
            minimum=0,
            maximum=MAX_MASK_ARTIFACTS,
        ),
        "output_file_count": _integer(
            record["output_file_count"],
            field="output_file_count",
            minimum=0,
            maximum=MAX_OUTPUT_FILES,
        ),
        "output_bytes": _integer(
            record["output_bytes"], field="output_bytes", minimum=0, maximum=MAX_OUTPUT_BYTES
        ),
        "termination_reason": _enum(
            record["termination_reason"],
            field="termination_reason",
            allowed=_TERMINATION_REASONS,
        ),
        "bundle_spec_digest": _sha256(
            record["bundle_spec_digest"], field="bundle_spec_digest"
        ),
        "evidence_report_digest": (
            None
            if record["evidence_report_digest"] is None
            else _sha256(record["evidence_report_digest"], field="evidence_report_digest")
        ),
        "summary_digest": _sha256(record["summary_digest"], field="summary_digest"),
    }
    if checked["summary_digest"] != canonical_sha256_v1(
        checked, own_digest_field="summary_digest"
    ):
        raise StreamContractError("stream_summary: digest mismatch")
    return StreamSummary(checked)


def _bounded_text_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        raise StreamContractError(f"{field}: expected at most 32 entries")
    checked: list[str] = []
    total = 0
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise StreamContractError(f"{field}[{index}]: expected nonempty text")
        size = len(item.encode("utf-8", errors="strict"))
        if size > 512:
            raise StreamContractError(f"{field}[{index}]: text limit exceeded")
        total += size
        checked.append(item)
    if total > 8_192:
        raise StreamContractError(f"{field}: aggregate text limit exceeded")
    return checked


def _sustained_section(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="sustained_section")
    names = frozenset(
        {
            "start_source_frame_index",
            "end_source_frame_index_exclusive",
            "scheduled_frame_count",
            "duration_ns",
            "processed_frame_count",
            "dropped_frame_count",
            "failed_unaccounted_frame_count",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "sustained_fps_display",
        }
    )
    _keys(record, field="sustained_section", allowed=names)
    start_index = _integer(
        record["start_source_frame_index"],
        field="sustained_section.start_source_frame_index",
        minimum=0,
        maximum=MAX_FRAMES - 1,
    )
    end_index = _integer(
        record["end_source_frame_index_exclusive"],
        field="sustained_section.end_source_frame_index_exclusive",
        minimum=1,
        maximum=MAX_FRAMES,
    )
    scheduled = _integer(
        record["scheduled_frame_count"],
        field="sustained_section.scheduled_frame_count",
        minimum=1,
        maximum=MAX_FRAMES,
    )
    duration = _integer(
        record["duration_ns"],
        field="sustained_section.duration_ns",
        minimum=1,
        maximum=MAX_UINT64,
    )
    processed = _integer(
        record["processed_frame_count"],
        field="sustained_section.processed_frame_count",
        minimum=1,
        maximum=MAX_FRAMES,
    )
    dropped = _integer(
        record["dropped_frame_count"],
        field="sustained_section.dropped_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    failed = _integer(
        record["failed_unaccounted_frame_count"],
        field="sustained_section.failed_unaccounted_frame_count",
        minimum=0,
        maximum=MAX_FRAMES,
    )
    if end_index != start_index + scheduled or processed + dropped + failed != scheduled:
        raise StreamContractError("sustained_section: source indices are not fully accounted")
    percentiles = [
        _decimal(record[name], field=f"sustained_section.{name}", minimum=Decimal(0))
        for name in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms")
    ]
    if not Decimal(percentiles[0]) <= Decimal(percentiles[1]) <= Decimal(percentiles[2]):
        raise StreamContractError("sustained_section: latency percentiles are unordered")
    display = _ratio_display(processed * 1_000_000_000, duration)
    if record["sustained_fps_display"] != display:
        raise StreamContractError("sustained_section: FPS display mismatch")
    return {
        "start_source_frame_index": start_index,
        "end_source_frame_index_exclusive": end_index,
        "scheduled_frame_count": scheduled,
        "duration_ns": duration,
        "processed_frame_count": processed,
        "dropped_frame_count": dropped,
        "failed_unaccounted_frame_count": failed,
        "p50_latency_ms": percentiles[0],
        "p95_latency_ms": percentiles[1],
        "p99_latency_ms": percentiles[2],
        "sustained_fps_display": display,
    }


def _memory_observation(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="memory")
    names = frozenset(
        {
            "collector",
            "coverage_complete",
            "stream_job_peak_rss_bytes",
            "accelerator_process_tree_peak_bytes",
            "thermal_status",
            "power_status",
        }
    )
    _keys(record, field="memory", allowed=names)
    if not isinstance(record["coverage_complete"], bool):
        raise StreamContractError("memory.coverage_complete: expected boolean")
    rss = record["stream_job_peak_rss_bytes"]
    accelerator = record["accelerator_process_tree_peak_bytes"]
    if rss is not None:
        rss = _integer(rss, field="memory.stream_job_peak_rss_bytes", minimum=1, maximum=MAX_UINT64)
    if accelerator is not None:
        accelerator = _integer(
            accelerator,
            field="memory.accelerator_process_tree_peak_bytes",
            minimum=1,
            maximum=MAX_UINT64,
        )
    if not record["coverage_complete"] and (rss is not None or accelerator is not None):
        raise StreamContractError("memory: incomplete coverage cannot publish passing values")
    return {
        "collector": _memory_collector(record["collector"]),
        "coverage_complete": record["coverage_complete"],
        "stream_job_peak_rss_bytes": rss,
        "accelerator_process_tree_peak_bytes": accelerator,
        "thermal_status": _enum(
            record["thermal_status"],
            field="memory.thermal_status",
            allowed=frozenset({"nominal", "throttled", "unknown"}),
        ),
        "power_status": _enum(
            record["power_status"],
            field="memory.power_status",
            allowed=frozenset({"known", "unknown"}),
        ),
    }


def _quality_observation(
    value: Any, *, requirement: dict[str, Any] | None
) -> dict[str, Any] | None:
    if requirement is None:
        if value is not None:
            raise StreamContractError("quality: no quality result was requested")
        return None
    record = _mapping(value, field="quality")
    names = frozenset(set(requirement) | {"status", "measured_value"})
    _keys(record, field="quality", allowed=names)
    for key, expected in requirement.items():
        if record[key] != expected:
            raise StreamContractError(f"quality.{key}: requirement identity mismatch")
    claimed_status = _enum(
        record["status"],
        field="quality.status",
        allowed=frozenset({"passed", "failed", "unknown"}),
    )
    measured = (
        None
        if record["measured_value"] is None
        else _decimal(record["measured_value"], field="quality.measured_value")
    )
    if measured is None:
        derived_status = "unknown"
    else:
        observed = Decimal(measured)
        threshold = Decimal(requirement["threshold"])
        passed = (
            observed >= threshold
            if requirement["direction"] == "higher_is_better"
            else observed <= threshold
        )
        derived_status = "passed" if passed else "failed"
    if claimed_status != derived_status:
        raise StreamContractError("quality.status: diverges from exact measured gate")
    return {**requirement, "status": derived_status, "measured_value": measured}


def validate_stream_qualification_report(
    payload: Mapping[str, Any],
) -> StreamQualificationReport:
    record = _mapping(payload, field="stream_qualification_report")
    names = frozenset(
        {
            "schema_version",
            "report_kind",
            "report_id",
            "report_digest",
            "status",
            "started_at",
            "completed_at",
            "valid_until",
            "bundle_spec_digest",
            "artifact_set_digest",
            "artifact_state_fingerprint",
            "environment_fingerprint",
            "protocol_fingerprint",
            "stream_workload_profile",
            "summary",
            "sustained_section",
            "memory",
            "quality",
            "limitations",
            "failures",
        }
    )
    _keys(record, field="stream_qualification_report", allowed=names)
    if record["schema_version"] != 1 or record["report_kind"] != "stream_qualification":
        raise StreamContractError("stream_qualification_report: wrong kind/version")
    status = _enum(
        record["status"],
        field="stream_qualification_report.status",
        allowed=frozenset({"qualified", "failed", "fixture_only"}),
    )
    started = _utc(record["started_at"], field="started_at")
    completed = _utc(record["completed_at"], field="completed_at")
    valid_until = _utc(record["valid_until"], field="valid_until")
    parsed_started = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    parsed_completed = datetime.strptime(completed, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    parsed_valid = datetime.strptime(valid_until, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    if parsed_started > parsed_completed or not (
        parsed_completed <= parsed_valid
        and (parsed_valid - parsed_completed).total_seconds() <= 90 * 86_400
    ):
        raise StreamContractError("stream_qualification_report: invalid evidence interval")
    workload = validate_stream_workload_profile(record["stream_workload_profile"])
    job = validate_stream_job_spec(workload.to_dict()["stream_job"])
    job_record = job.to_dict()
    summary = validate_stream_summary(record["summary"])
    summary_record = summary.to_dict()
    if (
        summary_record["task"] != job_record["task"]
        or summary_record["source_kind"] != job_record["source"]["source_kind"]
    ):
        raise StreamContractError("stream_qualification_report: summary/workload mismatch")
    sustained = _sustained_section(record["sustained_section"])
    sustained_period_ns = job_record["source"]["source_rate_den"] * 1_000_000_000
    expected_sustained_frames = (
        sustained["duration_ns"] * job_record["source"]["source_rate_num"]
        + sustained_period_ns
        - 1
    ) // sustained_period_ns
    if sustained["scheduled_frame_count"] != expected_sustained_frames:
        raise StreamContractError(
            "stream_qualification_report: sustained count/cadence mismatch"
        )
    memory = _memory_observation(record["memory"])
    if memory["collector"] != job_record["memory_collector"]:
        raise StreamContractError("stream_qualification_report: memory collector mismatch")
    quality = _quality_observation(
        record["quality"], requirement=job_record["quality_requirement"]
    )
    limitations = _bounded_text_list(record["limitations"], field="limitations")
    failures = _bounded_text_list(record["failures"], field="failures")
    if summary_record["scheduled_frame_count"] > job_record["max_frames"]:
        raise StreamContractError("stream_qualification_report: frame cap mismatch")
    if summary_record["duration_ns"] > job_record["max_duration_seconds"] * 1_000_000_000:
        raise StreamContractError("stream_qualification_report: duration cap mismatch")
    if summary_record["frame_queue_count_high_watermark"] > job_record["queue_capacity_frames"]:
        raise StreamContractError("stream_qualification_report: queue-count HWM mismatch")
    if (
        summary_record["frame_queue_decoded_bytes_high_watermark"]
        > job_record["max_queued_decoded_bytes"]
    ):
        raise StreamContractError("stream_qualification_report: queue-byte HWM mismatch")
    if summary_record["callback_item_high_watermark"] > job_record["callback_max_items"]:
        raise StreamContractError("stream_qualification_report: callback-item HWM mismatch")
    if summary_record["callback_bytes_high_watermark"] > job_record["callback_max_bytes"]:
        raise StreamContractError("stream_qualification_report: callback-byte HWM mismatch")
    if summary_record["result_count"] > min(
        job_record["max_total_results"],
        summary_record["processed_frame_count"] * job_record["max_results_per_frame"],
    ):
        raise StreamContractError("stream_qualification_report: result-count cap mismatch")
    if summary_record["mask_artifact_count"] > job_record["max_mask_artifacts"]:
        raise StreamContractError("stream_qualification_report: mask-count cap mismatch")
    if job_record["task"] == "object_detection" and summary_record["mask_artifact_count"]:
        raise StreamContractError("stream_qualification_report: detection cannot publish masks")
    if summary_record["output_file_count"] > job_record["max_output_files"]:
        raise StreamContractError("stream_qualification_report: output-file cap mismatch")
    if summary_record["output_bytes"] > job_record["max_output_bytes"]:
        raise StreamContractError("stream_qualification_report: output-byte cap mismatch")
    if job_record["drop_policy"] == "block" and summary_record["dropped_frame_count"]:
        raise StreamContractError("stream_qualification_report: block policy recorded drops")
    if sustained["duration_ns"] > summary_record["duration_ns"]:
        raise StreamContractError("stream_qualification_report: sustained duration exceeds run")
    if sustained["processed_frame_count"] > summary_record["processed_frame_count"]:
        raise StreamContractError("stream_qualification_report: sustained count exceeds run")
    if sustained["dropped_frame_count"] > summary_record["dropped_frame_count"]:
        raise StreamContractError("stream_qualification_report: sustained drops exceed run")
    if (
        sustained["failed_unaccounted_frame_count"]
        > summary_record["failed_unaccounted_frame_count"]
    ):
        raise StreamContractError("stream_qualification_report: sustained failures exceed run")
    if (
        sustained["start_source_frame_index"] != job_record["warmup_frame_count"]
        or sustained["end_source_frame_index_exclusive"]
        > summary_record["scheduled_frame_count"]
    ):
        raise StreamContractError("stream_qualification_report: sustained source window mismatch")
    if summary_record["evidence_report_digest"] is not None:
        raise StreamContractError("qualification summary cannot claim selected evidence")
    if status == "qualified":
        if failures or summary_record["status"] != "completed":
            raise StreamContractError("qualified stream report cannot contain failures")
        if job_record["required_duration_seconds"] < 600:
            raise StreamContractError("qualified stream report requires a ten-minute job")
        if sustained["duration_ns"] < MIN_SUSTAINED_DURATION_NS:
            raise StreamContractError("qualified stream report has a short sustained section")
        required_frames = (
            job_record["source"]["source_rate_num"]
            * job_record["required_duration_seconds"]
            + job_record["source"]["source_rate_den"]
            - 1
        ) // job_record["source"]["source_rate_den"]
        if (
            summary_record["termination_reason"] not in _QUALIFIED_TERMINATIONS
            or summary_record["duration_ns"]
            < job_record["required_duration_seconds"] * 1_000_000_000
            or summary_record["scheduled_frame_count"]
            < job_record["warmup_frame_count"] + required_frames
            or sustained["scheduled_frame_count"] < required_frames
        ):
            raise StreamContractError("qualified stream report ended early or unsuccessfully")
        if summary_record["termination_reason"] == "duration_cap" and (
            summary_record["duration_ns"]
            != job_record["max_duration_seconds"] * 1_000_000_000
        ):
            raise StreamContractError("qualified duration-cap termination is inconsistent")
        if (
            summary_record["termination_reason"] == "frame_cap"
            and summary_record["scheduled_frame_count"] != job_record["max_frames"]
        ):
            raise StreamContractError("qualified frame-cap termination is inconsistent")
        if sustained["failed_unaccounted_frame_count"]:
            raise StreamContractError("qualified sustained section has failed-unaccounted frames")
        if not sustained_fps_meets_minimum(
            processed_count=sustained["processed_frame_count"],
            duration_ns=sustained["duration_ns"],
            minimum_fps=job_record["min_sustained_fps"],
        ):
            raise StreamContractError("qualified stream report failed sustained FPS")
        if Decimal(sustained["p95_latency_ms"]) > Decimal(
            job_record["max_p95_latency_ms"]
        ):
            raise StreamContractError("qualified stream report failed p95 latency")
        if summary_record["failed_unaccounted_frame_count"]:
            raise StreamContractError("qualified stream report has failed-unaccounted frames")
        if not drop_fraction_within_limit(
            dropped_count=summary_record["dropped_frame_count"],
            processed_count=summary_record["processed_frame_count"],
            maximum_numerator=job_record["max_drop_num"],
            maximum_denominator=job_record["max_drop_den"],
        ):
            raise StreamContractError("qualified stream report failed drop fraction")
        if summary_record["max_consecutive_drops"] > job_record["max_consecutive_drops"]:
            raise StreamContractError("qualified stream report failed consecutive-drop gate")
        if not memory["coverage_complete"] or memory["stream_job_peak_rss_bytes"] is None:
            raise StreamContractError("qualified stream report has unknown whole-job memory")
        if memory["stream_job_peak_rss_bytes"] > job_record["max_stream_job_peak_rss_bytes"]:
            raise StreamContractError("qualified stream report failed RSS gate")
        accelerator_limit = job_record["max_accelerator_process_tree_peak_bytes"]
        if accelerator_limit is not None and (
            memory["accelerator_process_tree_peak_bytes"] is None
            or memory["accelerator_process_tree_peak_bytes"] > accelerator_limit
        ):
            raise StreamContractError("qualified stream report failed accelerator-memory gate")
        if quality is not None and quality["status"] != "passed":
            raise StreamContractError("qualified stream report failed quality gate")
    checked = {
        "schema_version": 1,
        "report_kind": "stream_qualification",
        "report_id": _identifier(record["report_id"], field="report_id"),
        "report_digest": _sha256(record["report_digest"], field="report_digest"),
        "status": status,
        "started_at": started,
        "completed_at": completed,
        "valid_until": valid_until,
        "bundle_spec_digest": _sha256(record["bundle_spec_digest"], field="bundle_spec_digest"),
        "artifact_set_digest": _sha256(record["artifact_set_digest"], field="artifact_set_digest"),
        "artifact_state_fingerprint": _sha256(
            record["artifact_state_fingerprint"], field="artifact_state_fingerprint"
        ),
        "environment_fingerprint": _sha256(
            record["environment_fingerprint"], field="environment_fingerprint"
        ),
        "protocol_fingerprint": _sha256(
            record["protocol_fingerprint"], field="protocol_fingerprint"
        ),
        "stream_workload_profile": workload.to_dict(),
        "summary": summary_record,
        "sustained_section": sustained,
        "memory": memory,
        "quality": quality,
        "limitations": limitations,
        "failures": failures,
    }
    if checked["summary"]["bundle_spec_digest"] != checked["bundle_spec_digest"]:
        raise StreamContractError("stream_qualification_report: bundle identity mismatch")
    if checked["report_digest"] != canonical_sha256_v1(
        checked, own_digest_field="report_digest"
    ):
        raise StreamContractError("stream_qualification_report: digest mismatch")
    return StreamQualificationReport(checked)


_SELECTION_REASONS = frozenset(
    {
        "no_eligible_candidate",
        "catalog_only",
        "maturity_disallowed",
        "evidence_not_qualified",
        "evidence_untrusted",
        "source_rate_mismatch",
        "source_admission_mismatch",
        "decoder_policy_mismatch",
        "decoder_backend_unsupported",
        "decoder_memory_mismatch",
        "stream_memory_collector_mismatch",
        "stream_memory_unknown",
        "output_cadence_mismatch",
        "result_limit_mismatch",
        "mask_encoding_mismatch",
        "camera_pool_mismatch",
        "queue_byte_mismatch",
        "drop_metric_unknown",
        "drop_fraction_above_requirement",
        "sustained_fps_unknown",
        "sustained_fps_below_requirement",
        "consecutive_drops_above_requirement",
        "p95_latency_above_requirement",
        "quality_gate_failed",
        "runtime_unavailable",
        "hardware_unavailable",
        "camera_binding_ambiguous",
    }
)


def _reason_codes(value: Any, *, field: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or len(value) > 32 or (not allow_empty and not value):
        raise StreamContractError(f"{field}: invalid reason count")
    checked = [_enum(item, field=f"{field}[]", allowed=_SELECTION_REASONS) for item in value]
    if checked != sorted(set(checked), key=lambda item: item.encode("utf-8")):
        raise StreamContractError(f"{field}: reasons must be unique byte-sorted")
    return checked


def validate_stream_selection_decision(
    payload: Mapping[str, Any],
) -> StreamSelectionDecision:
    record = _mapping(payload, field="stream_selection_decision")
    names = frozenset(
        {
            "schema_version",
            "decision_kind",
            "decision_id",
            "decision_digest",
            "status",
            "decided_at",
            "local_job_digest",
            "stream_source_preimage",
            "stream_source_digest",
            "artifact_resolver_state_digest",
            "environment_fingerprint",
            "protocol_fingerprint",
            "stream_job_spec",
            "stream_workload_profile",
            "selected_bundle",
            "selected_evidence",
            "selected_artifact_state_fingerprint",
            "support_scope",
            "reason_codes",
            "candidate_evaluations",
        }
    )
    _keys(record, field="stream_selection_decision", allowed=names)
    if record["schema_version"] != 1 or record["decision_kind"] != "local_stream":
        raise StreamContractError("stream_selection_decision: wrong kind/version")
    status = _enum(
        record["status"],
        field="stream_selection_decision.status",
        allowed=frozenset({"selected", "abstained"}),
    )
    job = validate_stream_job_spec(record["stream_job_spec"])
    workload = validate_stream_workload_profile(record["stream_workload_profile"])
    if workload.to_dict()["stream_job"] != job.to_dict():
        raise StreamContractError("stream_selection_decision: job/workload mismatch")
    if record["local_job_digest"] != job.local_job_digest:
        raise StreamContractError("stream_selection_decision: local job digest mismatch")
    source_preimage = _stream_source_preimage(
        record["stream_source_preimage"], expected_source=job.to_dict()["source"]
    )
    source_digest = compute_stream_source_digest(
        source_preimage, expected_source=job.to_dict()["source"]
    )
    if record["stream_source_digest"] != source_digest:
        raise StreamContractError("stream_selection_decision: source digest mismatch")
    selected_bundle = record["selected_bundle"]
    if selected_bundle is not None:
        selected_bundle = _mapping(selected_bundle, field="selected_bundle")
        bundle_names = frozenset({"bundle_id", "bundle_version", "bundle_spec_digest"})
        _keys(selected_bundle, field="selected_bundle", allowed=bundle_names)
        selected_bundle = {
            "bundle_id": _identifier(selected_bundle["bundle_id"], field="selected_bundle.bundle_id"),
            "bundle_version": _identifier(
                selected_bundle["bundle_version"], field="selected_bundle.bundle_version"
            ),
            "bundle_spec_digest": _sha256(
                selected_bundle["bundle_spec_digest"], field="selected_bundle.bundle_spec_digest"
            ),
        }
    selected_evidence = record["selected_evidence"]
    if selected_evidence is not None:
        selected_evidence = _mapping(selected_evidence, field="selected_evidence")
        evidence_names = frozenset({"report_id", "report_digest", "trust_domain"})
        _keys(selected_evidence, field="selected_evidence", allowed=evidence_names)
        selected_evidence = {
            "report_id": _identifier(selected_evidence["report_id"], field="selected_evidence.report_id"),
            "report_digest": _sha256(
                selected_evidence["report_digest"], field="selected_evidence.report_digest"
            ),
            "trust_domain": _enum(
                selected_evidence["trust_domain"],
                field="selected_evidence.trust_domain",
                allowed=frozenset({"yolozu_managed", "site_managed"}),
            ),
        }
    reasons = _reason_codes(
        record["reason_codes"], field="reason_codes", allow_empty=status == "selected"
    )
    raw_evaluations = record["candidate_evaluations"]
    if not isinstance(raw_evaluations, list) or len(raw_evaluations) > 128:
        raise StreamContractError("candidate_evaluations: candidate cap exceeded")
    evaluations: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    selected_count = 0
    selected_evaluation_identity: tuple[str, str] | None = None
    for index, raw in enumerate(raw_evaluations):
        field = f"candidate_evaluations[{index}]"
        item = _mapping(raw, field=field)
        item_names = frozenset({"bundle_id", "bundle_version", "rank_state", "reason_codes"})
        _keys(item, field=field, allowed=item_names)
        identity = (
            _identifier(item["bundle_id"], field=f"{field}.bundle_id"),
            _identifier(item["bundle_version"], field=f"{field}.bundle_version"),
        )
        rank_state = _enum(
            item["rank_state"],
            field=f"{field}.rank_state",
            allowed=frozenset({"selected", "eligible", "excluded"}),
        )
        item_reasons = _reason_codes(
            item["reason_codes"],
            field=f"{field}.reason_codes",
            allow_empty=rank_state != "excluded",
        )
        if rank_state == "selected":
            selected_count += 1
            selected_evaluation_identity = identity
            if item_reasons:
                raise StreamContractError(
                    "candidate_evaluations: selected candidate cannot have rejection reasons"
                )
        identities.append(identity)
        evaluations.append(
            {
                "bundle_id": identity[0],
                "bundle_version": identity[1],
                "rank_state": rank_state,
                "reason_codes": item_reasons,
            }
        )
    if identities != sorted(set(identities), key=lambda item: (item[0].encode(), item[1].encode())):
        raise StreamContractError("candidate_evaluations: expected complete unique byte order")
    selected_artifact = record["selected_artifact_state_fingerprint"]
    if selected_artifact is not None:
        selected_artifact = _sha256(selected_artifact, field="selected_artifact_state_fingerprint")
    if status == "selected":
        if (
            selected_bundle is None
            or selected_evidence is None
            or selected_artifact is None
            or selected_count != 1
            or selected_evaluation_identity
            != (selected_bundle["bundle_id"], selected_bundle["bundle_version"])
            or reasons
        ):
            raise StreamContractError("stream_selection_decision: contradictory selected outcome")
    elif (
        selected_bundle is not None
        or selected_evidence is not None
        or selected_artifact is not None
        or selected_count
        or "no_eligible_candidate" not in reasons
    ):
        raise StreamContractError("stream_selection_decision: contradictory abstained outcome")
    checked = {
        "schema_version": 1,
        "decision_kind": "local_stream",
        "decision_id": _identifier(record["decision_id"], field="decision_id"),
        "decision_digest": _sha256(record["decision_digest"], field="decision_digest"),
        "status": status,
        "decided_at": _utc(record["decided_at"], field="decided_at"),
        "local_job_digest": job.local_job_digest,
        "stream_source_preimage": source_preimage,
        "stream_source_digest": source_digest,
        "artifact_resolver_state_digest": _sha256(
            record["artifact_resolver_state_digest"], field="artifact_resolver_state_digest"
        ),
        "environment_fingerprint": _sha256(
            record["environment_fingerprint"], field="environment_fingerprint"
        ),
        "protocol_fingerprint": _sha256(record["protocol_fingerprint"], field="protocol_fingerprint"),
        "stream_job_spec": job.to_dict(),
        "stream_workload_profile": workload.to_dict(),
        "selected_bundle": selected_bundle,
        "selected_evidence": selected_evidence,
        "selected_artifact_state_fingerprint": selected_artifact,
        "support_scope": _enum(
            record["support_scope"],
            field="support_scope",
            allowed=frozenset({"none", "site_qualified", "public_qualified"}),
        ),
        "reason_codes": reasons,
        "candidate_evaluations": evaluations,
    }
    if status == "abstained" and checked["support_scope"] != "none":
        raise StreamContractError("stream_selection_decision: abstention has support scope")
    if status == "selected" and checked["support_scope"] == "none":
        raise StreamContractError(
            "stream_selection_decision: selected outcome requires qualified support scope"
        )
    if (
        status == "selected"
        and checked["support_scope"] == "public_qualified"
        and selected_evidence["trust_domain"] != "yolozu_managed"
    ):
        raise StreamContractError(
            "stream_selection_decision: public scope requires yolozu-managed evidence"
        )
    if len(canonical_json_v1(checked)) > 1_048_576:
        raise StreamContractError("stream_selection_decision: canonical byte cap exceeded")
    if checked["decision_digest"] != canonical_sha256_v1(
        checked, own_digest_field="decision_digest"
    ):
        raise StreamContractError("stream_selection_decision: digest mismatch")
    return StreamSelectionDecision(checked)


def _canonical_control_object(data: bytes, *, field: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise StreamContractError(f"{field}: expected bytes")
    try:
        value = load_bounded_json_bytes(data, label=field)
        record = _mapping(value, field=field)
        canonical = canonical_json_v1(record)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, StreamContractError):
            raise
        raise StreamContractError(str(exc)) from exc
    if canonical != data:
        raise StreamContractError(f"{field}: expected exact canonical_json_v1 bytes")
    return record


def _parse_stream_checksum_manifest(
    data: bytes, *, job: Mapping[str, Any]
) -> tuple[tuple[_ChecksumEntry, ...], int]:
    manifest = _canonical_control_object(data, field="checksums.json")
    names = frozenset(
        {"schema_version", "files", "expected_paths", "file_count", "total_bytes"}
    )
    _keys(manifest, field="checksums.json", allowed=names)
    if manifest["schema_version"] != 1:
        raise StreamContractError("checksums.json.schema_version: expected 1")
    raw_files = manifest["files"]
    raw_paths = manifest["expected_paths"]
    if not isinstance(raw_files, list) or not isinstance(raw_paths, list):
        raise StreamContractError("checksums.json: files and expected_paths must be arrays")
    if any(
        isinstance(value, Mapping) and value.get("path") == _CHECKSUMS_NAME
        for value in raw_files
    ):
        raise StreamContractError("checksums.json: must not list itself")
    if len(raw_files) + 1 > job["max_output_files"]:
        raise StreamContractError("stream output: output-file cap exceeded")

    entries: list[_ChecksumEntry] = []
    for index, value in enumerate(raw_files):
        field = f"checksums.json.files[{index}]"
        entry = _mapping(value, field=field)
        entry_names = frozenset({"path", "size_bytes", "sha256"})
        _keys(entry, field=field, allowed=entry_names)
        path = entry["path"]
        if not isinstance(path, str):
            raise StreamContractError(f"{field}.path: expected string")
        if path == _CHECKSUMS_NAME:
            raise StreamContractError("checksums.json: must not list itself")
        is_mask = _MASK_PATH_RE.fullmatch(path) is not None
        if path not in _REQUIRED_STREAM_OUTPUTS and not is_mask:
            raise StreamContractError(f"{field}.path: undeclared stream output")
        maximum_size = job["max_mask_bytes"] if is_mask else job["max_output_bytes"]
        entries.append(
            _ChecksumEntry(
                path=path,
                size_bytes=_integer(
                    entry["size_bytes"],
                    field=f"{field}.size_bytes",
                    minimum=0,
                    maximum=maximum_size,
                ),
                sha256=_sha256(entry["sha256"], field=f"{field}.sha256"),
            )
        )
    paths = [entry.path for entry in entries]
    if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
        raise StreamContractError("checksums.json: paths must be UTF-8 byte-sorted")
    if len(paths) != len(set(paths)):
        raise StreamContractError("checksums.json: duplicate path")
    if raw_paths != paths:
        raise StreamContractError(
            "checksums.json: expected_paths must exactly match ordered files"
        )
    if not _REQUIRED_STREAM_OUTPUTS.issubset(paths):
        raise StreamContractError("checksums.json: required stream output is missing")
    mask_count = sum(_MASK_PATH_RE.fullmatch(path) is not None for path in paths)
    if mask_count > job["max_mask_artifacts"]:
        raise StreamContractError("stream output: mask-artifact cap exceeded")
    if (
        _integer(
            manifest["file_count"],
            field="checksums.json.file_count",
            minimum=0,
            maximum=job["max_output_files"],
        )
        != len(entries)
    ):
        raise StreamContractError("checksums.json.file_count: mismatch")
    total_bytes = sum(entry.size_bytes for entry in entries)
    if (
        _integer(
            manifest["total_bytes"],
            field="checksums.json.total_bytes",
            minimum=0,
            maximum=job["max_output_bytes"],
        )
        != total_bytes
    ):
        raise StreamContractError("checksums.json.total_bytes: mismatch")
    if total_bytes + len(data) > job["max_output_bytes"]:
        raise StreamContractError("stream output: output-byte cap exceeded")
    return tuple(entries), total_bytes


def _byte_chunks(value: Iterable[bytes] | bytes, *, field: str) -> Iterable[bytes]:
    if isinstance(value, bytes):
        # A finite declared empty file is valid. Empty chunks yielded by an
        # iterable are rejected by the consumer because they cannot progress.
        return () if not value else (value,)
    if isinstance(value, (str, bytearray, memoryview, Mapping)):
        raise StreamContractError(f"{field}: expected bytes or an iterable of byte chunks")
    try:
        return iter(value)
    except TypeError as exc:
        raise StreamContractError(
            f"{field}: expected bytes or an iterable of byte chunks"
        ) from exc


def _consume_declared_file(
    value: Iterable[bytes] | bytes,
    *,
    entry: _ChecksumEntry,
    collect_limit: int | None,
) -> bytes | None:
    digest = hashlib.sha256()
    size = 0
    collected = bytearray() if collect_limit is not None else None
    for index, chunk in enumerate(_byte_chunks(value, field=entry.path)):
        if not isinstance(chunk, bytes):
            raise StreamContractError(f"{entry.path}[{index}]: expected bytes")
        if not chunk:
            raise StreamContractError(f"{entry.path}[{index}]: empty chunk cannot progress")
        size += len(chunk)
        if size > entry.size_bytes:
            raise StreamContractError(f"{entry.path}: size exceeds checksums.json")
        digest.update(chunk)
        if collected is not None:
            if size > collect_limit:
                raise StreamContractError(f"{entry.path}: control-file byte cap exceeded")
            collected.extend(chunk)
    if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise StreamContractError(
            f"{entry.path}: size or SHA-256 does not match checksums.json"
        )
    return None if collected is None else bytes(collected)


def _frame_lines(
    value: Iterable[bytes] | bytes, *, expected_bytes: int
) -> Iterable[bytes]:
    """Yield complete LF-terminated lines from bounded arbitrary byte chunks."""

    direct_bytes = isinstance(value, bytes)
    chunks = (() if not value else (value,)) if direct_bytes else _byte_chunks(
        value, field="stream_results.jsonl"
    )
    buffered = bytearray()
    total_bytes = 0
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, bytes):
            raise StreamContractError(
                f"stream_results.jsonl[{chunk_index}]: expected bytes"
            )
        if not chunk and not direct_bytes:
            raise StreamContractError(
                f"stream_results.jsonl[{chunk_index}]: empty chunk cannot progress"
            )
        total_bytes += len(chunk)
        if total_bytes > expected_bytes:
            raise StreamContractError(
                "stream_results.jsonl: size exceeds checksums.json"
            )
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            end = len(chunk) if newline < 0 else newline + 1
            segment_size = end - offset
            if len(buffered) + segment_size > CALLBACK_MAX_BYTES + 1:
                raise StreamContractError(
                    "stream_results.jsonl: line exceeds callback byte cap"
                )
            buffered.extend(chunk[offset:end])
            offset = end
            if newline >= 0:
                yield bytes(buffered)
                buffered.clear()
        if total_bytes == expected_bytes and buffered:
            raise StreamContractError(
                "stream_results.jsonl: final line is not LF-terminated"
            )
    if total_bytes != expected_bytes:
        raise StreamContractError(
            "stream_results.jsonl: size does not match checksums.json"
        )
    if buffered:
        raise StreamContractError(
            "stream_results.jsonl: final line is not LF-terminated"
        )


def _canonical_frame_line(raw: bytes, *, line_number: int) -> dict[str, Any]:
    field = f"stream_results.jsonl line {line_number}"
    if not isinstance(raw, bytes):
        raise StreamContractError(f"{field}: expected bytes")
    if len(raw) > CALLBACK_MAX_BYTES + 1:
        raise StreamContractError(f"{field}: line exceeds callback byte cap")
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or b"\n" in raw[:-1]:
        raise StreamContractError(f"{field}: expected one canonical LF-terminated line")
    body = raw[:-1]
    if not body:
        raise StreamContractError(f"{field}: blank line is invalid")
    record = _canonical_control_object(body, field=field)
    return record


def _validate_stream_provenance(
    payload: Mapping[str, Any],
    *,
    job: StreamJobSpec,
    workload: StreamWorkloadProfile,
    decision: StreamSelectionDecision,
    summary: StreamSummary,
) -> dict[str, Any]:
    record = _mapping(payload, field="provenance.json")
    names = frozenset(
        {
            "schema_version",
            "output_kind",
            "local_job_digest",
            "stream_workload_fingerprint",
            "stream_selection_decision_digest",
            "stream_source_digest",
            "bundle_spec_digest",
            "evidence_report_digest",
            "artifact_state_fingerprint",
            "summary_digest",
            "frame_result_count",
        }
    )
    _keys(record, field="provenance.json", allowed=names)
    if record["schema_version"] != 1 or record["output_kind"] != "local_stream":
        raise StreamContractError("provenance.json: wrong kind/version")
    decision_record = decision.to_dict()
    if decision_record["status"] != "selected":
        raise StreamContractError("provenance.json: stream selection did not select a bundle")
    selected_bundle = decision_record["selected_bundle"]
    selected_evidence = decision_record["selected_evidence"]
    expected = {
        "schema_version": 1,
        "output_kind": "local_stream",
        "local_job_digest": job.local_job_digest,
        "stream_workload_fingerprint": workload.workload_fingerprint,
        "stream_selection_decision_digest": decision_record["decision_digest"],
        "stream_source_digest": decision_record["stream_source_digest"],
        "bundle_spec_digest": selected_bundle["bundle_spec_digest"],
        "evidence_report_digest": selected_evidence["report_digest"],
        "artifact_state_fingerprint": decision_record[
            "selected_artifact_state_fingerprint"
        ],
        "summary_digest": summary.to_dict()["summary_digest"],
        "frame_result_count": summary.to_dict()["processed_frame_count"],
    }
    if record != expected:
        raise StreamContractError("provenance.json: identity or count mismatch")
    return expected


def _bind_output_summary_to_job(
    summary: Mapping[str, Any], *, job: Mapping[str, Any]
) -> None:
    if (
        summary["task"] != job["task"]
        or summary["source_kind"] != job["source"]["source_kind"]
    ):
        raise StreamContractError("stream output: summary/job identity mismatch")
    bounded_fields = (
        ("scheduled_frame_count", "max_frames"),
        ("frame_queue_count_high_watermark", "queue_capacity_frames"),
        (
            "frame_queue_decoded_bytes_high_watermark",
            "max_queued_decoded_bytes",
        ),
        ("callback_item_high_watermark", "callback_max_items"),
        ("callback_bytes_high_watermark", "callback_max_bytes"),
        ("result_count", "max_total_results"),
        ("mask_artifact_count", "max_mask_artifacts"),
        ("output_file_count", "max_output_files"),
        ("output_bytes", "max_output_bytes"),
    )
    for summary_field, job_field in bounded_fields:
        if summary[summary_field] > job[job_field]:
            raise StreamContractError(
                f"stream output: {summary_field} exceeds the job cap"
            )
    if summary["duration_ns"] > job["max_duration_seconds"] * 1_000_000_000:
        raise StreamContractError("stream output: duration exceeds the job cap")
    if summary["result_count"] > (
        summary["processed_frame_count"] * job["max_results_per_frame"]
    ):
        raise StreamContractError("stream output: per-frame result cap is inconsistent")
    if job["task"] == "object_detection" and summary["mask_artifact_count"]:
        raise StreamContractError("stream output: detection cannot publish masks")
    if job["drop_policy"] == "block" and summary["dropped_frame_count"]:
        raise StreamContractError("stream output: block policy cannot publish drops")


def validate_stream_output_artifacts(
    declared_regular_outputs: Iterable[tuple[str, Iterable[bytes] | bytes]],
    checksums_json: bytes,
    *,
    job: StreamJobSpec | Mapping[str, Any],
    workload: StreamWorkloadProfile | Mapping[str, Any],
    selection_decision: StreamSelectionDecision | Mapping[str, Any],
    dropped_source_frame_indices: Sequence[int],
) -> StreamSummary:
    """Validate one complete managed stream result without filesystem access.

    ``dropped_source_frame_indices`` is a bounded, internal observation needed to
    recompute drop count and adjacency. It is not copied to summary, provenance,
    qualification evidence, or any other public artifact.
    """

    validated_job = job if isinstance(job, StreamJobSpec) else validate_stream_job_spec(job)
    validated_workload = (
        workload
        if isinstance(workload, StreamWorkloadProfile)
        else validate_stream_workload_profile(workload)
    )
    validated_decision = (
        selection_decision
        if isinstance(selection_decision, StreamSelectionDecision)
        else validate_stream_selection_decision(selection_decision)
    )
    job_record = validated_job.to_dict()
    if validated_workload.to_dict()["stream_job"] != job_record:
        raise StreamContractError("stream output: workload/job mismatch")
    decision_record = validated_decision.to_dict()
    if (
        decision_record["stream_job_spec"] != job_record
        or decision_record["stream_workload_profile"] != validated_workload.to_dict()
    ):
        raise StreamContractError("stream output: selection/job/workload mismatch")
    if decision_record["status"] != "selected":
        raise StreamContractError("stream output: an abstained selection cannot produce output")

    entries, declared_payload_bytes = _parse_stream_checksum_manifest(
        checksums_json, job=job_record
    )
    if isinstance(declared_regular_outputs, Mapping):
        raise StreamContractError(
            "declared_regular_outputs: expected ordered (path, byte chunks) entries"
        )
    outputs: list[tuple[str, Iterable[bytes] | bytes]] = []
    for index, value in enumerate(declared_regular_outputs):
        if index >= job_record["max_output_files"] - 1:
            raise StreamContractError("stream output: output-file cap exceeded")
        if not isinstance(value, tuple) or len(value) != 2:
            raise StreamContractError(
                f"declared_regular_outputs[{index}]: expected (path, byte chunks)"
            )
        path, chunks = value
        if not isinstance(path, str):
            raise StreamContractError(
                f"declared_regular_outputs[{index}].path: expected string"
            )
        outputs.append((path, chunks))
    expected_paths = [entry.path for entry in entries]
    if [path for path, _chunks in outputs] != expected_paths:
        raise StreamContractError(
            "declared regular outputs must exactly match checksums.json ordered paths"
        )
    output_by_path = dict(outputs)
    entry_by_path = {entry.path: entry for entry in entries}

    summary_bytes = _consume_declared_file(
        output_by_path["stream_summary.json"],
        entry=entry_by_path["stream_summary.json"],
        collect_limit=4_194_304,
    )
    assert summary_bytes is not None
    summary = validate_stream_summary(
        _canonical_control_object(summary_bytes, field="stream_summary.json")
    )
    summary_record = summary.to_dict()
    _bind_output_summary_to_job(summary_record, job=job_record)
    if summary_record["output_file_count"] != len(entries) + 1:
        raise StreamContractError("stream output: summary output-file count mismatch")
    if summary_record["output_bytes"] != declared_payload_bytes:
        raise StreamContractError("stream output: summary declared-payload byte mismatch")
    selected_bundle = decision_record["selected_bundle"]
    selected_evidence = decision_record["selected_evidence"]
    if (
        summary_record["bundle_spec_digest"]
        != selected_bundle["bundle_spec_digest"]
        or summary_record["evidence_report_digest"]
        != selected_evidence["report_digest"]
    ):
        raise StreamContractError("stream output: summary selection identity mismatch")

    provenance_bytes = _consume_declared_file(
        output_by_path["provenance.json"],
        entry=entry_by_path["provenance.json"],
        collect_limit=4_194_304,
    )
    assert provenance_bytes is not None
    _validate_stream_provenance(
        _canonical_control_object(provenance_bytes, field="provenance.json"),
        job=validated_job,
        workload=validated_workload,
        decision=validated_decision,
        summary=summary,
    )

    if not isinstance(dropped_source_frame_indices, Sequence) or isinstance(
        dropped_source_frame_indices, (str, bytes, bytearray, memoryview)
    ):
        raise StreamContractError("dropped_source_frame_indices: expected bounded sequence")
    dropped_indices: list[int] = []
    for position, value in enumerate(dropped_source_frame_indices):
        if position >= MAX_FRAMES:
            raise StreamContractError("dropped_source_frame_indices: frame cap exceeded")
        dropped_indices.append(
            _integer(
                value,
                field=f"dropped_source_frame_indices[{position}]",
                minimum=0,
                maximum=MAX_FRAMES - 1,
            )
        )
    derived_consecutive = compute_max_consecutive_drops(dropped_indices)
    if any(index >= summary_record["scheduled_frame_count"] for index in dropped_indices):
        raise StreamContractError("dropped_source_frame_indices: index is outside the run")
    if (
        len(dropped_indices) != summary_record["dropped_frame_count"]
        or derived_consecutive != summary_record["max_consecutive_drops"]
    ):
        raise StreamContractError("stream output: drop aggregate mismatch")

    stream_entry = entry_by_path["stream_results.jsonl"]
    stream_digest = hashlib.sha256()
    stream_bytes = 0
    processed_indices: list[int] = []
    result_count = 0
    mask_references: dict[str, dict[str, Any]] = {}
    previous_index = -1
    for line_number, raw_line in enumerate(
        _frame_lines(
            output_by_path["stream_results.jsonl"],
            expected_bytes=stream_entry.size_bytes,
        ),
        start=1,
    ):
        if line_number > job_record["max_frames"]:
            raise StreamContractError("stream_results.jsonl: frame cap exceeded")
        if not isinstance(raw_line, bytes):
            raise StreamContractError(
                f"stream_results.jsonl line {line_number}: expected bytes"
            )
        stream_bytes += len(raw_line)
        if stream_bytes > stream_entry.size_bytes:
            raise StreamContractError(
                "stream_results.jsonl: size exceeds checksums.json"
            )
        stream_digest.update(raw_line)
        frame = validate_frame_result(
            _canonical_frame_line(raw_line, line_number=line_number),
            source_rate_num=job_record["source"]["source_rate_num"],
            source_rate_den=job_record["source"]["source_rate_den"],
            expected_task=job_record["task"],
            expected_width=job_record["source"]["width"],
            expected_height=job_record["source"]["height"],
        )
        if frame.canonical_line() != raw_line:
            raise StreamContractError(
                f"stream_results.jsonl line {line_number}: noncanonical FrameResult"
            )
        frame_record = frame.to_dict()
        source_index = frame_record["source_frame_index"]
        if source_index <= previous_index:
            raise StreamContractError(
                "stream_results.jsonl: source indices must be strict ascending"
            )
        if source_index >= summary_record["scheduled_frame_count"]:
            raise StreamContractError("stream_results.jsonl: source index is outside the run")
        previous_index = source_index
        processed_indices.append(source_index)
        frame_results = frame_record["task_results"]
        if len(frame_results) > job_record["max_results_per_frame"]:
            raise StreamContractError("stream_results.jsonl: per-frame result cap exceeded")
        result_count += len(frame_results)
        if result_count > job_record["max_total_results"]:
            raise StreamContractError("stream_results.jsonl: total result cap exceeded")
        for task_result in frame_results:
            mask = task_result["mask"]
            if mask is None:
                continue
            path = mask["relative_path"]
            if path in mask_references:
                raise StreamContractError(
                    "stream_results.jsonl: duplicate managed mask path"
                )
            if len(mask_references) >= job_record["max_mask_artifacts"]:
                raise StreamContractError("stream output: mask-artifact cap exceeded")
            mask_references[path] = mask
    if stream_bytes != stream_entry.size_bytes or stream_digest.hexdigest() != stream_entry.sha256:
        raise StreamContractError(
            "stream_results.jsonl: size or SHA-256 does not match checksums.json"
        )
    if len(processed_indices) != summary_record["processed_frame_count"]:
        raise StreamContractError("stream output: processed-frame count mismatch")
    if result_count != summary_record["result_count"]:
        raise StreamContractError("stream output: task-result count mismatch")
    if len(mask_references) != summary_record["mask_artifact_count"]:
        raise StreamContractError("stream output: mask-artifact count mismatch")

    processed_cursor = dropped_cursor = 0
    while processed_cursor < len(processed_indices) and dropped_cursor < len(dropped_indices):
        processed_index = processed_indices[processed_cursor]
        dropped_index = dropped_indices[dropped_cursor]
        if processed_index == dropped_index:
            raise StreamContractError("stream output: processed and dropped indices overlap")
        if processed_index < dropped_index:
            processed_cursor += 1
        else:
            dropped_cursor += 1
    derived_failed = summary_record["scheduled_frame_count"] - (
        len(processed_indices) + len(dropped_indices)
    )
    if derived_failed != summary_record["failed_unaccounted_frame_count"]:
        raise StreamContractError("stream output: failed-unaccounted count mismatch")

    declared_masks = {
        entry.path: entry
        for entry in entries
        if _MASK_PATH_RE.fullmatch(entry.path) is not None
    }
    if set(declared_masks) != set(mask_references):
        raise StreamContractError(
            "stream output: declared masks must exactly match FrameResult references"
        )
    for path, mask in mask_references.items():
        entry = declared_masks[path]
        if entry.size_bytes != mask["size_bytes"] or entry.sha256 != mask["sha256"]:
            raise StreamContractError(
                f"{path}: FrameResult mask reference does not match checksums.json"
            )
        _consume_declared_file(
            output_by_path[path], entry=entry, collect_limit=None
        )
    return summary
