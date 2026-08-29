"""Strict, tracker-free session-scoped tracking output interface contract v1."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from yolozu.adaptive.canonical import (
    canonical_decimal_v1,
    canonical_json_v1,
    canonical_sha256_v1,
)
from yolozu.adaptive.control_records import (
    MAX_CONTROL_RECORD_BYTES,
    load_bounded_json_bytes,
)
from yolozu.adaptive.streaming import (
    FrameResult,
    StreamJobSpec,
    StreamSummary,
    validate_frame_result,
    validate_stream_job_spec,
    validate_stream_summary,
)

__all__ = [
    "JS_SAFE_TRACK_ID_MAX",
    "MAX_ACTIVE_TRACKS",
    "MAX_JOB_ROW_STATE_BUDGET",
    "MAX_SESSION_UNIQUE_TRACKS",
    "MAX_TRACKING_MASK_ARTIFACTS",
    "MAX_TRACKING_OUTPUT_FILES",
    "MAX_TRACKING_ROWS_PER_FRAME",
    "TrackingContractError",
    "TrackingOutputInterface",
    "TrackingStateMachine",
    "TrackingValidationSummary",
    "validate_tracking_output_artifacts",
    "validate_tracking_output_interface",
    "validate_tracking_output_provenance",
    "validate_tracking_output_record",
    "validate_tracking_output_streams",
]


JS_SAFE_TRACK_ID_MAX = 9_007_199_254_740_991
MAX_SOURCE_FRAMES = 864_000
MAX_TRACKING_ROWS_PER_FRAME = 1_000
MAX_ACTIVE_TRACKS = 1_000
MAX_SESSION_UNIQUE_TRACKS = 1_000_000
MAX_JOB_ROW_STATE_BUDGET = 1_000_000
MAX_TRACK_AGE_FRAMES = 240
MAX_TRACKING_MASK_ARTIFACTS = 10_000
MAX_TRACKING_OUTPUT_FILES = 10_004
MAX_OUTPUT_BYTES = 4_294_967_296
MAX_UINT64 = 18_446_744_073_709_551_615
MAX_FRAME_RESULT_LINE_BYTES = MAX_CONTROL_RECORD_BYTES
MAX_TRACKING_RECORD_LINE_BYTES = MAX_CONTROL_RECORD_BYTES

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MASK_PATH_RE = re.compile(
    r"artifacts/masks/[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.png\Z", re.ASCII
)
_STATES = frozenset({"observed", "predicted", "lost", "ended"})
_TERMINATION_REASONS = frozenset(
    {"reset", "eof", "cancelled", "terminal_failure"}
)
_CHECKSUMS_NAME = "checksums.json"
_REQUIRED_DURABLE_OUTPUTS = frozenset(
    {
        "detector_frame_results.jsonl",
        "provenance.json",
        "stream_summary.json",
        "tracking_results.jsonl",
    }
)


class TrackingContractError(ValueError):
    """Stable fail-closed tracking interface-contract violation."""


@dataclass(frozen=True)
class TrackingOutputInterface:
    """Validated immutable limits and lifecycle semantics for one tracking job."""

    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)

    @property
    def digest(self) -> str:
        return str(self._record["tracking_output_interface_digest"])


@dataclass(frozen=True)
class TrackingValidationSummary:
    """Content-free aggregate returned after strict validation of both JSONL files."""

    detector_frame_result_count: int
    detector_task_result_count: int
    tracking_row_count: int
    session_termination_count: int
    session_count: int
    job_row_state_budget_used: int
    detector_bytes: int
    tracking_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "tracking_validation_summary",
            "detector_frame_result_count": self.detector_frame_result_count,
            "detector_task_result_count": self.detector_task_result_count,
            "tracking_row_count": self.tracking_row_count,
            "session_termination_count": self.session_termination_count,
            "session_count": self.session_count,
            "job_row_state_budget_used": self.job_row_state_budget_used,
            "detector_bytes": self.detector_bytes,
            "tracking_bytes": self.tracking_bytes,
            "contains_frame_or_identity_data": False,
        }


@dataclass(frozen=True)
class _ChecksumEntry:
    path: str
    size_bytes: int
    sha256: str


class _DigestingChunks:
    """One-shot bounded byte iterator with an exact expected digest."""

    def __init__(
        self,
        chunks: Iterable[bytes] | bytes,
        *,
        entry: _ChecksumEntry,
    ) -> None:
        self._chunks = chunks
        self._entry = entry
        self._started = False
        self._completed = False
        self._size = 0
        self._digest = hashlib.sha256()

    def __iter__(self) -> Iterable[bytes]:
        if self._started:
            raise TrackingContractError(
                f"{self._entry.path}: declared output can be consumed only once"
            )
        self._started = True
        chunks: Iterable[bytes]
        if isinstance(self._chunks, bytes):
            chunks = () if not self._chunks else (self._chunks,)
        else:
            chunks = self._chunks
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TrackingContractError(
                    f"{self._entry.path}: output chunks must be bytes"
                )
            if not chunk:
                raise TrackingContractError(
                    f"{self._entry.path}: zero-length non-progress chunk"
                )
            self._size += len(chunk)
            if self._size > self._entry.size_bytes:
                raise TrackingContractError(
                    f"{self._entry.path}: size exceeds checksums.json declaration"
                )
            self._digest.update(chunk)
            yield chunk
        self._completed = True

    def verify(self) -> None:
        if not self._completed:
            raise TrackingContractError(
                f"{self._entry.path}: declared output was not completely consumed"
            )
        if (
            self._size != self._entry.size_bytes
            or self._digest.hexdigest() != self._entry.sha256
        ):
            raise TrackingContractError(
                f"{self._entry.path}: size or SHA-256 does not match checksums.json"
            )


@dataclass(frozen=True)
class _TrackState:
    state: str
    consecutive_prediction_frames: int
    consecutive_lost_frames: int


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrackingContractError(f"{field}: expected object")
    return dict(value)


def _check_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TrackingContractError(f"{field}: unknown keys")
    missing = sorted(required - set(value))
    if missing:
        raise TrackingContractError(f"{field}: missing required keys")


def _exact_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingContractError(f"{field}: expected integer")
    if value < minimum or value > maximum:
        raise TrackingContractError(f"{field}: out of range")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TrackingContractError(f"{field}: expected lowercase SHA-256")
    return value


def _confidence(value: Any, *, field: str) -> str:
    try:
        token = canonical_decimal_v1(value, field=field, nonnegative=True)
    except ValueError as exc:
        raise TrackingContractError(str(exc)) from exc
    if Decimal(token) > 1:
        raise TrackingContractError(f"{field}: expected 0..1")
    return token


def _coordinate(value: Any, *, field: str) -> str:
    token = _confidence(value, field=field)
    return token


def _bbox(value: Any, *, field: str) -> dict[str, str]:
    bbox = _mapping(value, field=field)
    keys = frozenset({"x1", "y1", "x2", "y2"})
    _check_keys(bbox, field=field, allowed=keys, required=keys)
    checked = {
        name: _coordinate(bbox[name], field=f"{field}.{name}")
        for name in ("x1", "y1", "x2", "y2")
    }
    if not (
        Decimal(checked["x1"]) < Decimal(checked["x2"])
        and Decimal(checked["y1"]) < Decimal(checked["y2"])
    ):
        raise TrackingContractError(f"{field}: expected non-empty normalized xyxy")
    return checked


def _mask(
    value: Any,
    *,
    field: str,
    task: str,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> dict[str, Any] | None:
    if value is None:
        if task == "instance_segmentation":
            raise TrackingContractError(f"{field}: instance segmentation requires mask")
        return None
    if task != "instance_segmentation":
        raise TrackingContractError(f"{field}: object detection forbids mask")
    mask = _mapping(value, field=field)
    keys = frozenset(
        {"relative_path", "sha256", "size_bytes", "width", "height", "encoding_id"}
    )
    _check_keys(mask, field=field, allowed=keys, required=keys)
    path = mask["relative_path"]
    if (
        not isinstance(path, str)
        or not _MASK_PATH_RE.fullmatch(path)
        or ".." in path.split("/")
        or "//" in path
    ):
        raise TrackingContractError(f"{field}.relative_path: invalid managed path")
    if mask["encoding_id"] != "png_binary_mask_v1":
        raise TrackingContractError(f"{field}.encoding_id: unsupported value")
    checked = {
        "relative_path": path,
        "sha256": _sha256(mask["sha256"], field=f"{field}.sha256"),
        "size_bytes": _exact_int(
            mask["size_bytes"], field=f"{field}.size_bytes", minimum=1, maximum=67_108_864
        ),
        "width": _exact_int(
            mask["width"], field=f"{field}.width", minimum=1, maximum=8_192
        ),
        "height": _exact_int(
            mask["height"], field=f"{field}.height", minimum=1, maximum=8_192
        ),
        "encoding_id": "png_binary_mask_v1",
    }
    if expected_width is not None and checked["width"] != expected_width:
        raise TrackingContractError(f"{field}.width: source frame mismatch")
    if expected_height is not None and checked["height"] != expected_height:
        raise TrackingContractError(f"{field}.height: source frame mismatch")
    return checked


def _task_result_copy(value: Any, *, field: str, task: str) -> dict[str, Any]:
    result = _mapping(value, field=field)
    keys = frozenset({"class_id", "score", "bbox", "mask"})
    _check_keys(result, field=field, allowed=keys, required=keys)
    return {
        "class_id": _exact_int(
            result["class_id"], field=f"{field}.class_id", minimum=0, maximum=2_147_483_647
        ),
        "score": _confidence(result["score"], field=f"{field}.score"),
        "bbox": _bbox(result["bbox"], field=f"{field}.bbox"),
        "mask": _mask(result["mask"], field=f"{field}.mask", task=task),
    }


def _track_estimate(
    value: Any,
    *,
    field: str,
    task: str,
    expected_source: str,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    estimate = _mapping(value, field=field)
    keys = frozenset({"source_semantics", "confidence", "bbox", "mask"})
    _check_keys(estimate, field=field, allowed=keys, required=keys)
    if estimate["source_semantics"] != expected_source:
        raise TrackingContractError(f"{field}.source_semantics: state mismatch")
    return {
        "source_semantics": expected_source,
        "confidence": _confidence(
            estimate["confidence"], field=f"{field}.confidence"
        ),
        "bbox": _bbox(estimate["bbox"], field=f"{field}.bbox"),
        "mask": _mask(
            estimate["mask"],
            field=f"{field}.mask",
            task=task,
            expected_width=frame_width,
            expected_height=frame_height,
        ),
    }


def validate_tracking_output_interface(
    payload: Mapping[str, Any],
) -> TrackingOutputInterface:
    """Validate one immutable tracker-free output policy and its canonical digest."""

    record = _mapping(payload, field="tracking_output_interface")
    keys = frozenset(
        {
            "schema_version",
            "kind",
            "task",
            "detector_interval",
            "max_results_per_frame",
            "max_prediction_frames",
            "max_lost_frames",
            "max_active_tracks",
            "max_session_unique_tracks",
            "max_job_row_state_budget",
            "identity_scope",
            "biometric_inference",
            "cross_session_identity_linking",
            "persistent_identity_database",
            "tracking_output_interface_digest",
        }
    )
    _check_keys(record, field="tracking_output_interface", allowed=keys, required=keys)
    if record["schema_version"] != 1 or record["kind"] != "tracking_output_interface":
        raise TrackingContractError("tracking_output_interface: unsupported version or kind")
    task = record["task"]
    if task not in {"object_detection", "instance_segmentation"}:
        raise TrackingContractError("tracking_output_interface.task: unsupported value")
    checked: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tracking_output_interface",
        "task": task,
        "detector_interval": _exact_int(
            record["detector_interval"],
            field="detector_interval",
            minimum=1,
            maximum=MAX_SOURCE_FRAMES,
        ),
        "max_results_per_frame": _exact_int(
            record["max_results_per_frame"],
            field="max_results_per_frame",
            minimum=1,
            maximum=MAX_TRACKING_ROWS_PER_FRAME,
        ),
        "max_prediction_frames": _exact_int(
            record["max_prediction_frames"],
            field="max_prediction_frames",
            minimum=0,
            maximum=MAX_TRACK_AGE_FRAMES,
        ),
        "max_lost_frames": _exact_int(
            record["max_lost_frames"],
            field="max_lost_frames",
            minimum=0,
            maximum=MAX_TRACK_AGE_FRAMES,
        ),
    }
    fixed = {
        "max_active_tracks": MAX_ACTIVE_TRACKS,
        "max_session_unique_tracks": MAX_SESSION_UNIQUE_TRACKS,
        "max_job_row_state_budget": MAX_JOB_ROW_STATE_BUDGET,
        "identity_scope": "session_only",
        "biometric_inference": False,
        "cross_session_identity_linking": False,
        "persistent_identity_database": False,
    }
    for field, expected in fixed.items():
        if record[field] != expected or type(record[field]) is not type(expected):
            raise TrackingContractError(f"{field}: expected fixed privacy or limit value")
        checked[field] = expected
    checked["tracking_output_interface_digest"] = _sha256(
        record["tracking_output_interface_digest"],
        field="tracking_output_interface_digest",
    )
    if canonical_sha256_v1(
        checked, own_digest_field="tracking_output_interface_digest"
    ) != checked["tracking_output_interface_digest"]:
        raise TrackingContractError("tracking_output_interface_digest: mismatch")
    return TrackingOutputInterface(checked)


def _validate_observation_ref(
    value: Any, *, frame: Mapping[str, Any], field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _mapping(value, field=field)
    keys = frozenset({"source_frame_result_digest", "source_result_index"})
    _check_keys(reference, field=field, allowed=keys, required=keys)
    digest = _sha256(
        reference["source_frame_result_digest"],
        field=f"{field}.source_frame_result_digest",
    )
    if digest != frame["frame_result_digest"]:
        raise TrackingContractError(f"{field}: cross-frame or digest mismatch")
    task_results = frame["task_results"]
    index = _exact_int(
        reference["source_result_index"],
        field=f"{field}.source_result_index",
        minimum=0,
        maximum=max(0, len(task_results) - 1),
    )
    if not task_results or index >= len(task_results):
        raise TrackingContractError(f"{field}.source_result_index: dangling reference")
    return {
        "source_frame_result_digest": digest,
        "source_result_index": index,
    }, copy.deepcopy(task_results[index])


def validate_tracking_output_record(
    payload: Mapping[str, Any],
    *,
    interface: TrackingOutputInterface | Mapping[str, Any],
    frame_result: FrameResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one row or termination record structurally and normalize it."""

    if not isinstance(interface, TrackingOutputInterface):
        interface = validate_tracking_output_interface(interface)
    policy = interface.to_dict()
    record = _mapping(payload, field="tracking_output_record")
    kind = record.get("kind")
    if kind == "tracking_session_termination":
        keys = frozenset(
            {
                "schema_version",
                "kind",
                "tracking_output_interface_digest",
                "session_index",
                "last_source_frame_index",
                "last_session_frame_index",
                "termination_reason",
                "termination_offset_ns",
                "active_track_count",
                "lost_track_count",
            }
        )
        _check_keys(record, field="tracking_session_termination", allowed=keys, required=keys)
        if record["schema_version"] != 1:
            raise TrackingContractError("tracking_session_termination: unsupported version")
        if record["tracking_output_interface_digest"] != interface.digest:
            raise TrackingContractError("tracking_session_termination: interface mismatch")
        source_index = record["last_source_frame_index"]
        session_frame_index = record["last_session_frame_index"]
        if (source_index is None) != (session_frame_index is None):
            raise TrackingContractError("tracking_session_termination: incomplete last-frame pair")
        if source_index is not None:
            source_index = _exact_int(
                source_index,
                field="last_source_frame_index",
                minimum=0,
                maximum=MAX_SOURCE_FRAMES - 1,
            )
            session_frame_index = _exact_int(
                session_frame_index,
                field="last_session_frame_index",
                minimum=0,
                maximum=MAX_SOURCE_FRAMES - 1,
            )
        reason = record["termination_reason"]
        if reason not in _TERMINATION_REASONS:
            raise TrackingContractError("termination_reason: unsupported value")
        active_count = _exact_int(
            record["active_track_count"],
            field="active_track_count",
            minimum=0,
            maximum=MAX_ACTIVE_TRACKS,
        )
        lost_count = _exact_int(
            record["lost_track_count"],
            field="lost_track_count",
            minimum=0,
            maximum=MAX_ACTIVE_TRACKS,
        )
        if active_count + lost_count > MAX_ACTIVE_TRACKS:
            raise TrackingContractError(
                "tracking_session_termination: active plus lost count exceeds cap"
            )
        return {
            "schema_version": 1,
            "kind": "tracking_session_termination",
            "tracking_output_interface_digest": interface.digest,
            "session_index": _exact_int(
                record["session_index"],
                field="session_index",
                minimum=0,
                maximum=MAX_SOURCE_FRAMES - 1,
            ),
            "last_source_frame_index": source_index,
            "last_session_frame_index": session_frame_index,
            "termination_reason": reason,
            "termination_offset_ns": _exact_int(
                record["termination_offset_ns"],
                field="termination_offset_ns",
                minimum=0,
                maximum=MAX_UINT64,
            ),
            "active_track_count": active_count,
            "lost_track_count": lost_count,
        }
    if kind != "tracking_result":
        raise TrackingContractError("tracking_output_record.kind: unsupported value")
    if frame_result is None:
        raise TrackingContractError("tracking_result: source FrameResult is required")
    if not isinstance(frame_result, FrameResult):
        frame_result = validate_frame_result(frame_result, expected_task=policy["task"])
    frame = frame_result.to_dict()
    common = frozenset(
        {
            "schema_version",
            "kind",
            "tracking_output_interface_digest",
            "source_frame_index",
            "session_index",
            "session_frame_index",
            "track_id",
            "state",
            "row_source",
            "source_scheduled_due_offset_num_ns",
            "source_scheduled_due_offset_den",
            "tracking_completed_offset_ns",
            "consecutive_prediction_frames",
            "consecutive_lost_frames",
        }
    )
    optional = frozenset({"observation_ref", "observation_copy", "track_estimate"})
    _check_keys(record, field="tracking_result", allowed=common | optional, required=common)
    if record["schema_version"] != 1:
        raise TrackingContractError("tracking_result: unsupported version")
    if record["tracking_output_interface_digest"] != interface.digest:
        raise TrackingContractError("tracking_result: interface mismatch")
    state = record["state"]
    if state not in _STATES:
        raise TrackingContractError("tracking_result.state: unsupported value")
    expected_row_source = {
        "observed": "detector_observation_with_tracker_estimate",
        "predicted": "tracker_prediction",
        "lost": "lifecycle_only",
        "ended": "lifecycle_only",
    }[state]
    if record["row_source"] != expected_row_source:
        raise TrackingContractError("tracking_result.row_source: state mismatch")
    source_index = _exact_int(
        record["source_frame_index"],
        field="source_frame_index",
        minimum=0,
        maximum=MAX_SOURCE_FRAMES - 1,
    )
    if source_index != frame["source_frame_index"]:
        raise TrackingContractError("source_frame_index: source FrameResult mismatch")
    due_num = _exact_int(
        record["source_scheduled_due_offset_num_ns"],
        field="source_scheduled_due_offset_num_ns",
        minimum=0,
        maximum=MAX_UINT64,
    )
    due_den = _exact_int(
        record["source_scheduled_due_offset_den"],
        field="source_scheduled_due_offset_den",
        minimum=1,
        maximum=240_000,
    )
    if (
        due_num != frame["scheduled_due_offset_num_ns"]
        or due_den != frame["scheduled_due_offset_den"]
    ):
        raise TrackingContractError("tracking_result: source due-time mismatch")
    completed = _exact_int(
        record["tracking_completed_offset_ns"],
        field="tracking_completed_offset_ns",
        minimum=0,
        maximum=MAX_UINT64,
    )
    if completed < frame["processing_completed_offset_ns"] or completed * due_den < due_num:
        raise TrackingContractError("tracking_completed_offset_ns: precedes source completion or due")
    prediction_age = _exact_int(
        record["consecutive_prediction_frames"],
        field="consecutive_prediction_frames",
        minimum=0,
        maximum=MAX_TRACK_AGE_FRAMES,
    )
    lost_age = _exact_int(
        record["consecutive_lost_frames"],
        field="consecutive_lost_frames",
        minimum=0,
        maximum=MAX_TRACK_AGE_FRAMES,
    )
    checked: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tracking_result",
        "tracking_output_interface_digest": interface.digest,
        "source_frame_index": source_index,
        "session_index": _exact_int(
            record["session_index"], field="session_index", minimum=0, maximum=MAX_SOURCE_FRAMES - 1
        ),
        "session_frame_index": _exact_int(
            record["session_frame_index"],
            field="session_frame_index",
            minimum=0,
            maximum=MAX_SOURCE_FRAMES - 1,
        ),
        "track_id": _exact_int(
            record["track_id"],
            field="track_id",
            minimum=1,
            maximum=JS_SAFE_TRACK_ID_MAX,
        ),
        "state": state,
        "row_source": expected_row_source,
        "source_scheduled_due_offset_num_ns": due_num,
        "source_scheduled_due_offset_den": due_den,
        "tracking_completed_offset_ns": completed,
        "consecutive_prediction_frames": prediction_age,
        "consecutive_lost_frames": lost_age,
    }
    if state == "observed":
        if prediction_age or lost_age:
            raise TrackingContractError("observed: lifecycle ages must be zero")
        if "observation_ref" not in record or "track_estimate" not in record:
            raise TrackingContractError("observed: reference and estimate are required")
        observation_ref, source_result = _validate_observation_ref(
            record["observation_ref"], frame=frame, field="observation_ref"
        )
        checked["observation_ref"] = observation_ref
        if "observation_copy" in record:
            observation_copy = _task_result_copy(
                record["observation_copy"], field="observation_copy", task=policy["task"]
            )
            if observation_copy != source_result:
                raise TrackingContractError("observation_copy: referenced result mismatch")
            checked["observation_copy"] = observation_copy
        checked["track_estimate"] = _track_estimate(
            record["track_estimate"],
            field="track_estimate",
            task=policy["task"],
            expected_source="observation_adjusted",
            frame_width=frame["decoded_width"],
            frame_height=frame["decoded_height"],
        )
    elif state == "predicted":
        if prediction_age < 1 or lost_age:
            raise TrackingContractError("predicted: invalid lifecycle ages")
        if prediction_age > policy["max_prediction_frames"]:
            raise TrackingContractError("predicted: configured age limit exceeded")
        if any(name in record for name in ("observation_ref", "observation_copy")):
            raise TrackingContractError("predicted: detector links and copies are forbidden")
        if "track_estimate" not in record:
            raise TrackingContractError("predicted: track_estimate is required")
        checked["track_estimate"] = _track_estimate(
            record["track_estimate"],
            field="track_estimate",
            task=policy["task"],
            expected_source="tracker_prediction",
            frame_width=frame["decoded_width"],
            frame_height=frame["decoded_height"],
        )
    else:
        if state == "lost":
            if lost_age < 1 or prediction_age:
                raise TrackingContractError("lost: invalid lifecycle ages")
            if lost_age > policy["max_lost_frames"]:
                raise TrackingContractError("lost: configured age limit exceeded")
        elif prediction_age or lost_age:
            raise TrackingContractError("ended: lifecycle ages must be zero")
        if any(
            name in record
            for name in ("observation_ref", "observation_copy", "track_estimate")
        ):
            raise TrackingContractError(f"{state}: observation, geometry, mask, score, and estimate are forbidden")
    return checked


class TrackingStateMachine:
    """Validate complete frame batches before mutating bounded session state."""

    def __init__(self, interface: TrackingOutputInterface | Mapping[str, Any]) -> None:
        if not isinstance(interface, TrackingOutputInterface):
            interface = validate_tracking_output_interface(interface)
        self.interface = interface
        self._policy = interface.to_dict()
        self._session_index = 0
        self._next_session_frame_index = 0
        self._last_source_frame_index: int | None = None
        self._last_session_source_frame_index: int | None = None
        self._last_session_frame_index: int | None = None
        self._last_due_num: int | None = None
        self._last_due_den: int | None = None
        self._last_frame_processing_completed_offset_ns = 0
        self._last_tracking_completed_offset_ns = 0
        self._active: dict[int, _TrackState] = {}
        self._seen_ids: set[int] = set()
        self._job_budget_used = 0
        self._final = False
        self._session_termination_count = 0
        self._tracking_row_count = 0

    @property
    def job_budget_used(self) -> int:
        return self._job_budget_used

    @property
    def session_index(self) -> int:
        return self._session_index

    @property
    def session_termination_count(self) -> int:
        return self._session_termination_count

    @property
    def tracking_row_count(self) -> int:
        return self._tracking_row_count

    @property
    def final(self) -> bool:
        return self._final

    def validate_frame_batch(
        self,
        frame_result: FrameResult | Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if self._final:
            raise TrackingContractError("tracking state: job already terminated")
        if not isinstance(frame_result, FrameResult):
            frame_result = validate_frame_result(
                frame_result, expected_task=self._policy["task"]
            )
        frame = frame_result.to_dict()
        source_index = frame["source_frame_index"]
        if self._last_source_frame_index is not None and source_index <= self._last_source_frame_index:
            raise TrackingContractError("source_frame_index: duplicate or regressing")
        due_num = frame["scheduled_due_offset_num_ns"]
        due_den = frame["scheduled_due_offset_den"]
        if (
            self._last_due_num is not None
            and self._last_due_den is not None
            and due_num * self._last_due_den <= self._last_due_num * due_den
        ):
            raise TrackingContractError("source scheduled due offset: duplicate or regressing")
        if (
            frame["processing_completed_offset_ns"]
            < self._last_frame_processing_completed_offset_ns
        ):
            raise TrackingContractError("source processing completion offset: regressing")
        if self._next_session_frame_index >= MAX_SOURCE_FRAMES:
            raise TrackingContractError("session_frame_index: session frame cap exceeded")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise TrackingContractError("tracking frame batch: expected bounded row sequence")
        if len(rows) > self._policy["max_results_per_frame"]:
            raise TrackingContractError("tracking frame batch: per-frame row limit exceeded")

        normalized = tuple(
            validate_tracking_output_record(
                row, interface=self.interface, frame_result=frame_result
            )
            for row in rows
        )
        track_ids = [row["track_id"] for row in normalized]
        if track_ids != sorted(track_ids) or len(track_ids) != len(set(track_ids)):
            raise TrackingContractError("tracking frame batch: rows must have unique ascending track_id")
        for row in normalized:
            if row["session_index"] != self._session_index:
                raise TrackingContractError("session_index: reset sequence mismatch")
            if row["session_frame_index"] != self._next_session_frame_index:
                raise TrackingContractError("session_frame_index: expected exact next index")
            if row["tracking_completed_offset_ns"] < self._last_tracking_completed_offset_ns:
                raise TrackingContractError("tracking_completed_offset_ns: regressing")
        if normalized:
            completions = {row["tracking_completed_offset_ns"] for row in normalized}
            if len(completions) != 1:
                raise TrackingContractError("tracking frame batch: completion timestamp mismatch")

        rows_by_id = {row["track_id"]: row for row in normalized}
        missing_active = sorted(set(self._active) - set(rows_by_id))
        if missing_active:
            raise TrackingContractError("tracking frame batch: active track omitted")
        duplicate_references: set[tuple[str, int]] = set()
        new_ids: list[int] = []
        next_active = dict(self._active)
        for row in normalized:
            track_id = row["track_id"]
            previous = self._active.get(track_id)
            if previous is None:
                if track_id in self._seen_ids:
                    raise TrackingContractError("track_id: ended ID cannot be reused in session")
                if row["state"] != "observed":
                    raise TrackingContractError("track state: a new ID must start observed")
                new_ids.append(track_id)
            if row["state"] == "observed":
                if self._next_session_frame_index % self._policy["detector_interval"] != 0:
                    raise TrackingContractError("observed: detector cadence is not due")
                reference = row["observation_ref"]
                key = (
                    reference["source_frame_result_digest"],
                    reference["source_result_index"],
                )
                if key in duplicate_references:
                    raise TrackingContractError("observation_ref: duplicate source-result link")
                duplicate_references.add(key)
                next_state = _TrackState("observed", 0, 0)
            elif row["state"] == "predicted":
                if previous is None or previous.state == "lost":
                    raise TrackingContractError("predicted: forbidden state transition")
                expected_age = (
                    previous.consecutive_prediction_frames + 1
                    if previous.state == "predicted"
                    else 1
                )
                if (
                    row["consecutive_prediction_frames"] != expected_age
                    or expected_age > self._policy["max_prediction_frames"]
                ):
                    raise TrackingContractError("predicted: age exceeds bound or is not consecutive")
                next_state = _TrackState("predicted", expected_age, 0)
            elif row["state"] == "lost":
                if previous is None:
                    raise TrackingContractError("lost: forbidden initial state")
                expected_age = (
                    previous.consecutive_lost_frames + 1
                    if previous.state == "lost"
                    else 1
                )
                if (
                    row["consecutive_lost_frames"] != expected_age
                    or expected_age > self._policy["max_lost_frames"]
                ):
                    raise TrackingContractError("lost: age exceeds bound or is not consecutive")
                next_state = _TrackState("lost", 0, expected_age)
            else:
                if previous is None:
                    raise TrackingContractError("ended: forbidden initial state")
                next_active.pop(track_id, None)
                continue
            next_active[track_id] = next_state

        if len(next_active) > MAX_ACTIVE_TRACKS:
            raise TrackingContractError("tracking state: active track limit exceeded")
        next_unique_count = len(self._seen_ids) + len(new_ids)
        if next_unique_count > MAX_SESSION_UNIQUE_TRACKS:
            raise TrackingContractError("tracking state: session unique-ID limit exceeded")
        budget_increment = len(normalized) + len(new_ids)
        if self._job_budget_used + budget_increment > MAX_JOB_ROW_STATE_BUDGET:
            raise TrackingContractError("tracking state: nonresetting job row-state budget exceeded")

        # All row, transition, reference, and cap checks above are complete. Mutate once.
        self._active = next_active
        self._seen_ids.update(new_ids)
        self._job_budget_used += budget_increment
        self._tracking_row_count += len(normalized)
        self._last_source_frame_index = source_index
        self._last_session_source_frame_index = source_index
        self._last_session_frame_index = self._next_session_frame_index
        self._last_due_num = due_num
        self._last_due_den = due_den
        self._last_frame_processing_completed_offset_ns = frame[
            "processing_completed_offset_ns"
        ]
        self._next_session_frame_index += 1
        if normalized:
            self._last_tracking_completed_offset_ns = normalized[0][
                "tracking_completed_offset_ns"
            ]
        else:
            self._last_tracking_completed_offset_ns = max(
                self._last_tracking_completed_offset_ns,
                frame["processing_completed_offset_ns"],
            )
        return normalized

    def terminate_session(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._final:
            raise TrackingContractError("tracking state: duplicate terminal record")
        record = validate_tracking_output_record(payload, interface=self.interface)
        if record["kind"] != "tracking_session_termination":
            raise TrackingContractError("tracking state: expected session termination")
        if record["session_index"] != self._session_index:
            raise TrackingContractError("tracking session termination: session mismatch")
        if record["last_source_frame_index"] != self._last_session_source_frame_index:
            raise TrackingContractError("tracking session termination: last source frame mismatch")
        if record["last_session_frame_index"] != self._last_session_frame_index:
            raise TrackingContractError("tracking session termination: last session frame mismatch")
        active_count = sum(state.state != "lost" for state in self._active.values())
        lost_count = sum(state.state == "lost" for state in self._active.values())
        if (
            record["active_track_count"] != active_count
            or record["lost_track_count"] != lost_count
        ):
            raise TrackingContractError("tracking session termination: active/lost count mismatch")
        if record["termination_offset_ns"] < self._last_tracking_completed_offset_ns:
            raise TrackingContractError("termination_offset_ns: precedes the last frame")

        reason = record["termination_reason"]
        if reason == "reset" and self._session_index >= MAX_SOURCE_FRAMES - 1:
            raise TrackingContractError("session_index: reset would exceed session cap")

        self._session_termination_count += 1
        self._active = {}
        self._seen_ids = set()
        self._last_session_source_frame_index = None
        self._last_session_frame_index = None
        self._next_session_frame_index = 0
        self._last_tracking_completed_offset_ns = record["termination_offset_ns"]
        if reason == "reset":
            self._session_index += 1
        else:
            self._final = True
        return record


def _canonical_jsonl_record(
    raw: bytes, *, label: str, maximum_bytes: int
) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise TrackingContractError(f"{label}: expected bytes")
    if len(raw) > maximum_bytes + 1:
        raise TrackingContractError(f"{label}: line exceeds its byte limit")
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or b"\n" in raw[:-1]:
        raise TrackingContractError(f"{label}: expected one canonical LF-terminated line")
    body = raw[:-1]
    if not body:
        raise TrackingContractError(f"{label}: blank line is invalid")
    try:
        value = load_bounded_json_bytes(body, label=label)
        record = _mapping(value, field=label)
        canonical = canonical_json_v1(record)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TrackingContractError):
            raise
        raise TrackingContractError(str(exc)) from exc
    if canonical != body:
        raise TrackingContractError(f"{label}: line is not canonical_json_v1")
    return record


def _register_mask_reference(
    value: Mapping[str, Any] | None,
    *,
    references: dict[str, dict[str, Any]],
    maximum: int,
) -> None:
    if value is None:
        return
    mask = copy.deepcopy(dict(value))
    path = str(mask["relative_path"])
    previous = references.get(path)
    if previous is not None:
        if previous != mask:
            raise TrackingContractError(
                f"{path}: one managed mask path has conflicting metadata"
            )
        return
    if len(references) >= maximum:
        raise TrackingContractError("tracking output: shared mask-artifact cap exceeded")
    references[path] = mask


def _validated_stream_job(
    value: StreamJobSpec | Mapping[str, Any],
) -> StreamJobSpec:
    try:
        return value if isinstance(value, StreamJobSpec) else validate_stream_job_spec(value)
    except ValueError as exc:
        raise TrackingContractError(f"stream_job: {exc}") from exc


def _validated_stream_summary(
    value: StreamSummary | Mapping[str, Any],
) -> StreamSummary:
    try:
        return value if isinstance(value, StreamSummary) else validate_stream_summary(value)
    except ValueError as exc:
        raise TrackingContractError(f"stream_summary.json: {exc}") from exc


def _canonical_control_record(data: bytes, *, field: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TrackingContractError(f"{field}: expected bytes")
    try:
        value = load_bounded_json_bytes(data, label=field)
        record = _mapping(value, field=field)
        canonical = canonical_json_v1(record)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TrackingContractError):
            raise
        raise TrackingContractError(str(exc)) from exc
    if canonical != data:
        raise TrackingContractError(f"{field}: expected exact canonical_json_v1 bytes")
    return record


def _consume_canonical_control(
    chunks: _DigestingChunks, *, field: str
) -> dict[str, Any]:
    data = bytearray()
    for chunk in chunks:
        if len(data) + len(chunk) > MAX_CONTROL_RECORD_BYTES:
            raise TrackingContractError(f"{field}: control-file byte cap exceeded")
        data.extend(chunk)
    chunks.verify()
    return _canonical_control_record(bytes(data), field=field)


def validate_tracking_output_provenance(
    payload: Mapping[str, Any],
    *,
    interface: TrackingOutputInterface | Mapping[str, Any],
    stream_job: StreamJobSpec | Mapping[str, Any],
    stream_summary: StreamSummary | Mapping[str, Any],
    detector_frame_results_sha256: str,
    tracking_results_sha256: str,
) -> dict[str, Any]:
    """Validate the minimal aggregate-only identity record for retained output."""

    if not isinstance(interface, TrackingOutputInterface):
        interface = validate_tracking_output_interface(interface)
    job = _validated_stream_job(stream_job)
    summary = _validated_stream_summary(stream_summary)
    record = _mapping(payload, field="provenance.json")
    names = frozenset(
        {
            "schema_version",
            "kind",
            "tracking_output_interface_digest",
            "stream_job_digest",
            "summary_digest",
            "detector_frame_results_sha256",
            "tracking_results_sha256",
            "identity_scope",
            "contains_frame_or_identity_data",
        }
    )
    _check_keys(record, field="provenance.json", allowed=names, required=names)
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise TrackingContractError("provenance.json.schema_version: expected integer 1")
    if record["kind"] != "tracking_output_provenance":
        raise TrackingContractError("provenance.json.kind: unsupported value")
    if record["identity_scope"] != "session_only":
        raise TrackingContractError("provenance.json.identity_scope: expected session_only")
    if record["contains_frame_or_identity_data"] is not False:
        raise TrackingContractError(
            "provenance.json.contains_frame_or_identity_data: expected false"
        )
    summary_record = summary.to_dict()
    expected = {
        "schema_version": 1,
        "kind": "tracking_output_provenance",
        "tracking_output_interface_digest": interface.digest,
        "stream_job_digest": job.local_job_digest,
        "summary_digest": summary_record["summary_digest"],
        "detector_frame_results_sha256": _sha256(
            detector_frame_results_sha256,
            field="detector_frame_results_sha256",
        ),
        "tracking_results_sha256": _sha256(
            tracking_results_sha256,
            field="tracking_results_sha256",
        ),
        "identity_scope": "session_only",
        "contains_frame_or_identity_data": False,
    }
    if record != expected:
        raise TrackingContractError("provenance.json: identity or privacy binding mismatch")
    return expected


def _parse_tracking_checksum_manifest(
    data: bytes,
    *,
    stream_job: StreamJobSpec,
) -> tuple[_ChecksumEntry, ...]:
    manifest = _canonical_control_record(data, field="checksums.json")
    job = stream_job.to_dict()
    keys = frozenset(
        {"schema_version", "files", "expected_paths", "file_count", "total_bytes"}
    )
    _check_keys(manifest, field="checksums.json", allowed=keys, required=keys)
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise TrackingContractError("checksums.json.schema_version: expected integer 1")
    raw_files = manifest["files"]
    raw_paths = manifest["expected_paths"]
    if not isinstance(raw_files, list) or not isinstance(raw_paths, list):
        raise TrackingContractError("checksums.json: files and expected_paths must be arrays")
    if len(raw_files) + 1 > job["max_output_files"]:
        raise TrackingContractError("tracking output: shared output-file cap exceeded")

    entries: list[_ChecksumEntry] = []
    for index, raw_entry in enumerate(raw_files):
        entry = _mapping(raw_entry, field=f"checksums.json.files[{index}]")
        entry_keys = frozenset({"path", "size_bytes", "sha256"})
        _check_keys(
            entry,
            field=f"checksums.json.files[{index}]",
            allowed=entry_keys,
            required=entry_keys,
        )
        path = entry["path"]
        if not isinstance(path, str):
            raise TrackingContractError(
                f"checksums.json.files[{index}].path: expected string"
            )
        if path == _CHECKSUMS_NAME:
            raise TrackingContractError("checksums.json: must not list itself")
        if path not in _REQUIRED_DURABLE_OUTPUTS and not _MASK_PATH_RE.fullmatch(path):
            raise TrackingContractError(
                f"checksums.json.files[{index}].path: undeclared tracking output"
            )
        maximum_size = (
            job["max_mask_bytes"]
            if _MASK_PATH_RE.fullmatch(path)
            else job["max_output_bytes"]
        )
        entries.append(
            _ChecksumEntry(
                path=path,
                size_bytes=_exact_int(
                    entry["size_bytes"],
                    field=f"checksums.json.files[{index}].size_bytes",
                    minimum=0,
                    maximum=maximum_size,
                ),
                sha256=_sha256(
                    entry["sha256"],
                    field=f"checksums.json.files[{index}].sha256",
                ),
            )
        )

    paths = [entry.path for entry in entries]
    if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
        raise TrackingContractError("checksums.json: paths must be UTF-8 byte-sorted")
    if len(paths) != len(set(paths)):
        raise TrackingContractError("checksums.json: duplicate path")
    if raw_paths != paths:
        raise TrackingContractError(
            "checksums.json: expected_paths must exactly match ordered files"
        )
    if not _REQUIRED_DURABLE_OUTPUTS.issubset(paths):
        raise TrackingContractError("checksums.json: required durable output is missing")
    mask_count = sum(path not in _REQUIRED_DURABLE_OUTPUTS for path in paths)
    if mask_count > job["max_mask_artifacts"]:
        raise TrackingContractError("tracking output: shared mask-artifact cap exceeded")
    if (
        _exact_int(
            manifest["file_count"],
            field="checksums.json.file_count",
            minimum=0,
            maximum=job["max_output_files"],
        )
        != len(entries)
    ):
        raise TrackingContractError("checksums.json.file_count: mismatch")
    total_bytes = sum(entry.size_bytes for entry in entries)
    if (
        _exact_int(
            manifest["total_bytes"],
            field="checksums.json.total_bytes",
            minimum=0,
            maximum=job["max_output_bytes"],
        )
        != total_bytes
    ):
        raise TrackingContractError("checksums.json.total_bytes: mismatch")
    if total_bytes + len(data) > job["max_output_bytes"]:
        raise TrackingContractError("tracking output: shared output-byte cap exceeded")
    return tuple(entries)


def validate_tracking_output_streams(
    detector_frame_results: Iterable[bytes],
    tracking_results: Iterable[bytes],
    *,
    interface: TrackingOutputInterface | Mapping[str, Any],
    expected_frame_result_count: int,
    expected_task_result_count: int,
    shared_output_bytes: int = 0,
    max_total_results: int | None = None,
    stream_job: StreamJobSpec | Mapping[str, Any] | None = None,
    scheduled_frame_count: int | None = None,
    _mask_references: dict[str, dict[str, Any]] | None = None,
    _max_mask_artifacts: int = MAX_TRACKING_MASK_ARTIFACTS,
) -> TrackingValidationSummary:
    """Validate retained detector and tracking JSONL streams in bounded lockstep.

    Each complete canonical FrameResult is validated before any row may link to it.
    The function keeps only one frame and at most 1,000 tracking rows at a time.
    """

    if not isinstance(interface, TrackingOutputInterface):
        interface = validate_tracking_output_interface(interface)
    policy = interface.to_dict()
    job = None if stream_job is None else _validated_stream_job(stream_job)
    job_record = None if job is None else job.to_dict()
    if job_record is not None and job_record["task"] != policy["task"]:
        raise TrackingContractError("tracking output: stream job/interface task mismatch")
    if max_total_results is not None and job_record is not None:
        raise TrackingContractError(
            "max_total_results: do not duplicate the StreamJobSpec cap"
        )
    configured_max_results = _exact_int(
        (
            job_record["max_total_results"]
            if job_record is not None
            else 1_000_000 if max_total_results is None else max_total_results
        ),
        field="max_total_results",
        minimum=1,
        maximum=1_000_000,
    )
    configured_max_frames = (
        MAX_SOURCE_FRAMES if job_record is None else job_record["max_frames"]
    )
    configured_max_detector_results_per_frame = (
        MAX_TRACKING_ROWS_PER_FRAME
        if job_record is None
        else job_record["max_results_per_frame"]
    )
    scheduled = (
        None
        if scheduled_frame_count is None
        else _exact_int(
            scheduled_frame_count,
            field="scheduled_frame_count",
            minimum=0,
            maximum=configured_max_frames,
        )
    )
    expected_frames = _exact_int(
        expected_frame_result_count,
        field="expected_frame_result_count",
        minimum=0,
        maximum=configured_max_frames,
    )
    expected_results = _exact_int(
        expected_task_result_count,
        field="expected_task_result_count",
        minimum=0,
        maximum=configured_max_results,
    )
    shared = _exact_int(
        shared_output_bytes,
        field="shared_output_bytes",
        minimum=0,
        maximum=MAX_OUTPUT_BYTES,
    )
    machine = TrackingStateMachine(interface)
    tracking_iterator = iter(tracking_results)
    tracking_line_number = 0
    lookahead: tuple[dict[str, Any], int] | None = None
    detector_bytes = 0
    tracking_bytes = 0
    frame_count = 0
    task_result_count = 0

    def next_tracking() -> tuple[dict[str, Any], int] | None:
        nonlocal tracking_bytes, tracking_line_number
        try:
            raw = next(tracking_iterator)
        except StopIteration:
            return None
        tracking_line_number += 1
        tracking_bytes += len(raw)
        if shared + detector_bytes + tracking_bytes > MAX_OUTPUT_BYTES:
            raise TrackingContractError("tracking output: shared 4-GiB byte cap exceeded")
        return (
            _canonical_jsonl_record(
                raw,
                label=f"tracking_results.jsonl:{tracking_line_number}",
                maximum_bytes=MAX_TRACKING_RECORD_LINE_BYTES,
            ),
            tracking_line_number,
        )

    for detector_line_number, raw_frame in enumerate(detector_frame_results, start=1):
        detector_bytes += len(raw_frame)
        if shared + detector_bytes + tracking_bytes > MAX_OUTPUT_BYTES:
            raise TrackingContractError("tracking output: shared 4-GiB byte cap exceeded")
        frame_payload = _canonical_jsonl_record(
            raw_frame,
            label=f"detector_frame_results.jsonl:{detector_line_number}",
            maximum_bytes=MAX_FRAME_RESULT_LINE_BYTES,
        )
        try:
            frame = validate_frame_result(
                frame_payload,
                source_rate_num=(
                    None
                    if job_record is None
                    else job_record["source"]["source_rate_num"]
                ),
                source_rate_den=(
                    None
                    if job_record is None
                    else job_record["source"]["source_rate_den"]
                ),
                expected_task=policy["task"],
                expected_width=(
                    None if job_record is None else job_record["source"]["width"]
                ),
                expected_height=(
                    None if job_record is None else job_record["source"]["height"]
                ),
            )
        except ValueError as exc:
            raise TrackingContractError(
                f"detector_frame_results.jsonl:{detector_line_number}: {exc}"
            ) from exc
        if frame.canonical_line() != raw_frame:
            raise TrackingContractError(
                f"detector_frame_results.jsonl:{detector_line_number}: canonical bytes mismatch"
            )
        frame_count += 1
        if frame_count > configured_max_frames:
            raise TrackingContractError("detector_frame_results.jsonl: frame line cap exceeded")
        frame_record = frame.to_dict()
        if (
            scheduled is not None
            and frame_record["source_frame_index"] >= scheduled
        ):
            raise TrackingContractError(
                "detector_frame_results.jsonl: source frame is outside the StreamSummary run"
            )
        if (
            len(frame_record["task_results"])
            > configured_max_detector_results_per_frame
        ):
            raise TrackingContractError(
                "detector_frame_results.jsonl: per-frame detector result cap exceeded"
            )
        task_result_count += len(frame_record["task_results"])
        if task_result_count > configured_max_results:
            raise TrackingContractError("detector_frame_results.jsonl: task-result cap exceeded")
        if _mask_references is not None:
            for result in frame_record["task_results"]:
                _register_mask_reference(
                    result["mask"],
                    references=_mask_references,
                    maximum=_max_mask_artifacts,
                )

        if lookahead is None:
            lookahead = next_tracking()
        rows: list[dict[str, Any]] = []
        while lookahead is not None and lookahead[0].get("kind") == "tracking_result":
            candidate = lookahead[0]
            candidate_index = candidate.get("source_frame_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index < frame_record["source_frame_index"]
            ):
                raise TrackingContractError("tracking_results.jsonl: dangling or regressing source frame")
            if candidate_index != frame_record["source_frame_index"]:
                break
            rows.append(candidate)
            if len(rows) > policy["max_results_per_frame"]:
                raise TrackingContractError("tracking_results.jsonl: per-frame row cap exceeded")
            lookahead = next_tracking()
        normalized_rows = machine.validate_frame_batch(frame, rows)
        if _mask_references is not None:
            for row in normalized_rows:
                observation = row.get("observation_copy")
                if observation is not None:
                    _register_mask_reference(
                        observation["mask"],
                        references=_mask_references,
                        maximum=_max_mask_artifacts,
                    )
                estimate = row.get("track_estimate")
                if estimate is not None:
                    _register_mask_reference(
                        estimate["mask"],
                        references=_mask_references,
                        maximum=_max_mask_artifacts,
                    )

        if lookahead is not None and lookahead[0].get("kind") == "tracking_session_termination":
            termination = lookahead[0]
            last_source = termination.get("last_source_frame_index")
            if last_source == frame_record["source_frame_index"]:
                machine.terminate_session(termination)
                lookahead = next_tracking()
            elif last_source is None or (
                isinstance(last_source, int)
                and not isinstance(last_source, bool)
                and last_source < frame_record["source_frame_index"]
            ):
                raise TrackingContractError("tracking_results.jsonl: misplaced termination record")

    if lookahead is None:
        lookahead = next_tracking()
    if lookahead is not None and lookahead[0].get("kind") == "tracking_session_termination":
        machine.terminate_session(lookahead[0])
        lookahead = next_tracking()
    if lookahead is not None:
        raise TrackingContractError("tracking_results.jsonl: dangling row or extra termination")
    if not machine.final:
        raise TrackingContractError("tracking_results.jsonl: missing EOF/cancel/failure termination")
    if frame_count != expected_frames:
        raise TrackingContractError(
            "detector_frame_results.jsonl: retained processed-frame count mismatch"
        )
    if task_result_count != expected_results:
        raise TrackingContractError(
            "detector_frame_results.jsonl: retained task-result count mismatch"
        )

    return TrackingValidationSummary(
        detector_frame_result_count=frame_count,
        detector_task_result_count=task_result_count,
        tracking_row_count=machine.tracking_row_count,
        session_termination_count=machine.session_termination_count,
        session_count=machine.session_index + 1,
        job_row_state_budget_used=machine.job_budget_used,
        detector_bytes=detector_bytes,
        tracking_bytes=tracking_bytes,
    )


def validate_tracking_output_artifacts(
    declared_regular_outputs: Iterable[
        tuple[str, Iterable[bytes] | bytes]
    ],
    checksums_json: bytes,
    *,
    interface: TrackingOutputInterface | Mapping[str, Any],
    stream_job: StreamJobSpec | Mapping[str, Any],
) -> TrackingValidationSummary:
    """Validate one complete declared tracking output at the byte boundary.

    The caller supplies each regular output exactly once in UTF-8 byte-sorted
    path order. JSONL values are iterables of complete LF-terminated lines;
    other files may use arbitrary byte chunks. ``checksums.json`` is supplied
    separately because it must never contain a self-entry. Counts are derived
    from the retained canonical StreamSummary; limits and source identity come
    from ``stream_job``. This function does not open paths, write output, or
    provide filesystem durability.
    """

    if not isinstance(interface, TrackingOutputInterface):
        interface = validate_tracking_output_interface(interface)
    job = _validated_stream_job(stream_job)
    job_record = job.to_dict()
    if job_record["task"] != interface.to_dict()["task"]:
        raise TrackingContractError("tracking output: stream job/interface task mismatch")
    entries = _parse_tracking_checksum_manifest(
        checksums_json,
        stream_job=job,
    )
    expected_paths = [entry.path for entry in entries]
    entry_by_path = {entry.path: entry for entry in entries}

    outputs: list[tuple[str, Iterable[bytes] | bytes]] = []
    for index, raw_output in enumerate(declared_regular_outputs):
        if index >= job_record["max_output_files"] - 1:
            raise TrackingContractError("tracking output: shared output-file cap exceeded")
        if not isinstance(raw_output, tuple) or len(raw_output) != 2:
            raise TrackingContractError(
                f"declared_regular_outputs[{index}]: expected (path, byte chunks)"
            )
        path, chunks = raw_output
        if not isinstance(path, str):
            raise TrackingContractError(
                f"declared_regular_outputs[{index}].path: expected string"
            )
        outputs.append((path, chunks))
    actual_paths = [path for path, _chunks in outputs]
    if actual_paths != expected_paths:
        raise TrackingContractError(
            "declared regular outputs must exactly match checksums.json ordered paths"
        )

    digesters = {
        path: _DigestingChunks(chunks, entry=entry_by_path[path])
        for path, chunks in outputs
    }
    stream_paths = {
        "detector_frame_results.jsonl",
        "tracking_results.jsonl",
    }
    summary_record = _validated_stream_summary(
        _consume_canonical_control(
            digesters["stream_summary.json"], field="stream_summary.json"
        )
    ).to_dict()
    if (
        summary_record["task"] != job_record["task"]
        or summary_record["source_kind"] != job_record["source"]["source_kind"]
    ):
        raise TrackingContractError("stream_summary.json: StreamJobSpec identity mismatch")
    if summary_record["scheduled_frame_count"] > job_record["max_frames"]:
        raise TrackingContractError("stream_summary.json: frame cap exceeded")
    if summary_record["result_count"] > job_record["max_total_results"]:
        raise TrackingContractError("stream_summary.json: total result cap exceeded")
    if summary_record["result_count"] > (
        summary_record["processed_frame_count"]
        * job_record["max_results_per_frame"]
    ):
        raise TrackingContractError(
            "stream_summary.json: per-frame detector result cap is inconsistent"
        )
    if summary_record["mask_artifact_count"] > job_record["max_mask_artifacts"]:
        raise TrackingContractError("stream_summary.json: mask-artifact cap exceeded")
    if summary_record["output_file_count"] != len(entries) + 1:
        raise TrackingContractError("stream_summary.json: output-file count mismatch")
    declared_payload_bytes = sum(entry.size_bytes for entry in entries)
    if summary_record["output_bytes"] != declared_payload_bytes:
        raise TrackingContractError("stream_summary.json: declared-payload byte mismatch")

    validated_summary = _validated_stream_summary(summary_record)
    provenance_record = _consume_canonical_control(
        digesters["provenance.json"], field="provenance.json"
    )
    validate_tracking_output_provenance(
        provenance_record,
        interface=interface,
        stream_job=job,
        stream_summary=validated_summary,
        detector_frame_results_sha256=entry_by_path[
            "detector_frame_results.jsonl"
        ].sha256,
        tracking_results_sha256=entry_by_path["tracking_results.jsonl"].sha256,
    )

    nonstream_declared_bytes = len(checksums_json) + sum(
        entry.size_bytes for entry in entries if entry.path not in stream_paths
    )
    mask_references: dict[str, dict[str, Any]] = {}
    summary = validate_tracking_output_streams(
        digesters["detector_frame_results.jsonl"],
        digesters["tracking_results.jsonl"],
        interface=interface,
        expected_frame_result_count=summary_record["processed_frame_count"],
        expected_task_result_count=summary_record["result_count"],
        shared_output_bytes=nonstream_declared_bytes,
        stream_job=job,
        scheduled_frame_count=summary_record["scheduled_frame_count"],
        _mask_references=mask_references,
        _max_mask_artifacts=job_record["max_mask_artifacts"],
    )
    digesters["detector_frame_results.jsonl"].verify()
    digesters["tracking_results.jsonl"].verify()

    for entry in entries:
        if entry.path in stream_paths or entry.path in {
            "provenance.json",
            "stream_summary.json",
        }:
            continue
        for _chunk in digesters[entry.path]:
            pass
        digesters[entry.path].verify()

    declared_masks = {
        entry.path: entry
        for entry in entries
        if entry.path not in _REQUIRED_DURABLE_OUTPUTS
    }
    if set(declared_masks) != set(mask_references):
        raise TrackingContractError(
            "tracking output: declared mask files must exactly match mask references"
        )
    for path, mask in mask_references.items():
        entry = declared_masks[path]
        if entry.size_bytes != mask["size_bytes"] or entry.sha256 != mask["sha256"]:
            raise TrackingContractError(
                f"{path}: mask reference does not match checksums.json"
            )
    if len(mask_references) != summary_record["mask_artifact_count"]:
        raise TrackingContractError("stream_summary.json: mask-artifact count mismatch")
    return summary
