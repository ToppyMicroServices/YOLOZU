"""Measured artifact and qualification evidence interface contracts.

The records in this module are strict, privacy-bounded control records.  They
describe local artifact observations and measured qualification results; they
do not make a bundle selectable by themselves.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bundles import AlgorithmBundleSpec, ZERO_DIGEST
from .canonical import canonical_decimal_v1, canonical_sha256_v1
from .control_records import load_bounded_json_bytes

__all__ = [
    "EvidenceActivationProjection",
    "EvidenceActivationRecord",
    "LocalArtifactInventory",
    "QualificationReport",
    "compute_artifact_state_fingerprint",
    "compute_evidence_selection_key",
    "load_evidence_activation_jsonl",
    "load_evidence_activation_jsonl_bytes",
    "project_evidence_activations",
    "validate_evidence_activation_record",
    "validate_local_artifact_inventory",
    "validate_qualification_report",
]


MAX_EVIDENCE_ACTIVATION_RECORDS = 8_192
MAX_EVIDENCE_ACTIVATION_BYTES = 64 * 1024 * 1024
MAX_SIGNED_BYTES = 9_223_372_036_854_775_807
MAX_UINT64 = 18_446_744_073_709_551_615
MAX_SUSTAINED_SAMPLES = 1_000_000
MIN_SUSTAINED_DURATION_NS = 600_000_000_000
MAX_SUSTAINED_SAMPLE_BYTES = 8_000_000
QUALIFICATION_VALIDITY = timedelta(days=90)

LATENCY_INTERVAL_ID = "image_e2e_validated_handoff_v1"
HANDOFF_ID = "image_result_mask_handoff_v1"
HANDOFF_VERSION = 1
LATENCY_PHASES = (
    "decode_pinned_source_bytes",
    "preprocess",
    "predict",
    "postprocess",
    "requested_class_prompt_mapping",
    "strict_result_validation",
    "mask_validation_encoding",
    "result_mask_handoff_completion",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ROLE_RE = re.compile(r"(?:repo_maintainer|release_reviewer|site_operator|automation)\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

_ARTIFACT_ROLES = frozenset(
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
_REPORT_STATUSES = frozenset({"smoke", "qualified", "hold", "failed"})
_TRUST_DOMAINS = frozenset(
    {"yolozu_managed", "site_managed", "operator_asserted", "unknown"}
)


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected object")
    return dict(value)


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


def _list(value: Any, *, field: str, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum} items")
    return list(value)


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum}")
    return value


def _enum(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field}: unsupported value")
    return value


def _token(value: Any, *, field: str, component: bool = False) -> str:
    pattern = _COMPONENT_RE if component else _ID_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field}: invalid identifier")
    return value


def _sha256(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")
    if not allow_zero and value == ZERO_DIGEST:
        raise ValueError(f"{field}: zero sentinel is invalid")
    return value


def _safe_text(value: Any, *, field: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty text")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field}: exceeds {maximum_bytes} UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field}: control characters are invalid")
    return value


def _utc(value: Any, *, field: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected exact RFC3339 UTC second")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{field}: invalid Gregorian UTC instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{field}: noncanonical UTC instant")
    return value, parsed


def _as_of(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("as_of: expected timezone-aware UTC datetime")
        return value.replace(microsecond=0)
    return _utc(value, field="as_of")[1]


def _identity(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset({"id", "version", "source_digest"})
    _keys(record, field=field, allowed=keys, required=keys)
    return {
        "id": _token(record["id"], field=f"{field}.id"),
        "version": _token(record["version"], field=f"{field}.version"),
        "source_digest": _sha256(
            record["source_digest"], field=f"{field}.source_digest"
        ),
    }


@dataclass(frozen=True)
class LocalArtifactInventory:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def artifact_state_fingerprint(self) -> str:
        return str(self._record["artifact_state_fingerprint"])

    @property
    def inventory_digest(self) -> str:
        return str(self._record["inventory_digest"])


def _artifact_state_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bundle_spec_digest": record["bundle_spec_digest"],
        "artifact_set_digest": record["artifact_set_digest"],
        "observations": [
            {
                "artifact_id": item["artifact_id"],
                "order": item["order"],
                "presence_status": item["presence_status"],
                "path_type_status": item["path_type_status"],
                "read_status": item["read_status"],
                "observed_size_bytes": item["observed_size_bytes"],
                "observed_sha256": item["observed_sha256"],
            }
            for item in record["observations"]
        ],
    }


def compute_artifact_state_fingerprint(record: Mapping[str, Any]) -> str:
    """Hash stable local artifact state without paths or audit-time fields."""

    return canonical_sha256_v1(_artifact_state_projection(record))


def validate_local_artifact_inventory(
    value: Mapping[str, Any], bundle: AlgorithmBundleSpec
) -> LocalArtifactInventory:
    """Validate one complete trusted-preflight artifact inventory."""

    record = _mapping(value, field="LocalArtifactInventory")
    keys = frozenset(
        {
            "schema_version",
            "inventory_id",
            "bundle_spec_digest",
            "artifact_set_digest",
            "observations",
            "artifact_state_fingerprint",
            "inventory_digest",
        }
    )
    _keys(record, field="LocalArtifactInventory", allowed=keys, required=keys)
    bundle_record = bundle.to_dict()
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("LocalArtifactInventory.schema_version: expected 1")
    if record["bundle_spec_digest"] != bundle.spec_digest:
        raise ValueError("bundle_spec_digest: does not match bundle")
    if record["artifact_set_digest"] != bundle.artifact_set_digest:
        raise ValueError("artifact_set_digest: does not match bundle")

    expected_artifacts = bundle_record["artifacts"]
    observations = _list(
        record["observations"],
        field="observations",
        minimum=len(expected_artifacts),
        maximum=len(expected_artifacts),
    )
    normalized_observations: list[dict[str, Any]] = []
    for index, (raw, expected) in enumerate(zip(observations, expected_artifacts)):
        field = f"observations[{index}]"
        item = _mapping(raw, field=field)
        item_keys = frozenset(
            {
                "artifact_id",
                "role",
                "order",
                "expected_size_bytes",
                "expected_sha256",
                "presence_status",
                "path_type_status",
                "read_status",
                "observed_size_bytes",
                "observed_sha256",
                "verified_at",
                "error_status",
            }
        )
        _keys(item, field=field, allowed=item_keys, required=item_keys)
        for name, expected_value in (
            ("artifact_id", expected["artifact_id"]),
            ("role", expected["role"]),
            ("order", expected["order"]),
            ("expected_size_bytes", expected["expected_size_bytes"]),
            ("expected_sha256", expected["sha256"]),
        ):
            if item[name] != expected_value or (
                isinstance(expected_value, int) and type(item[name]) is not int
            ):
                raise ValueError(f"{field}.{name}: contradicts bundle artifact")

        presence = _enum(
            item["presence_status"],
            field=f"{field}.presence_status",
            allowed=frozenset({"present", "missing", "unknown"}),
        )
        path_type = _enum(
            item["path_type_status"],
            field=f"{field}.path_type_status",
            allowed=frozenset({"regular_file", "not_regular_file", "unknown"}),
        )
        read_status = _enum(
            item["read_status"],
            field=f"{field}.read_status",
            allowed=frozenset({"readable", "unreadable", "not_applicable", "unknown"}),
        )
        error = _enum(
            item["error_status"],
            field=f"{field}.error_status",
            allowed=frozenset(
                {
                    "none",
                    "missing",
                    "not_regular_file",
                    "unreadable",
                    "size_mismatch",
                    "sha256_mismatch",
                    "observation_failed",
                }
            ),
        )
        observed_size = item["observed_size_bytes"]
        observed_hash = item["observed_sha256"]
        if observed_size is not None:
            observed_size = _integer(
                observed_size,
                field=f"{field}.observed_size_bytes",
                minimum=0,
                maximum=MAX_SIGNED_BYTES,
            )
        if observed_hash is not None:
            observed_hash = _sha256(
                observed_hash, field=f"{field}.observed_sha256", allow_zero=True
            )
        verified_at = _utc(item["verified_at"], field=f"{field}.verified_at")[0]

        if presence == "present":
            if path_type == "unknown":
                raise ValueError(f"{field}: present artifact requires known path type")
        else:
            if presence == "missing" and error != "missing":
                raise ValueError(f"{field}: missing artifact requires missing error")
            if path_type == "regular_file" or read_status == "readable":
                raise ValueError(f"{field}: absent/unknown artifact cannot be readable file")
        if path_type == "not_regular_file" and error != "not_regular_file":
            raise ValueError(f"{field}: non-regular path requires matching error")
        if read_status == "readable":
            if presence != "present" or path_type != "regular_file":
                raise ValueError(f"{field}: readable requires a present regular file")
            if observed_size is None or observed_hash is None:
                raise ValueError(f"{field}: readable requires observed size and SHA-256")
            expected_error = "none"
            if observed_size != expected["expected_size_bytes"]:
                expected_error = "size_mismatch"
            elif observed_hash != expected["sha256"]:
                expected_error = "sha256_mismatch"
            if error != expected_error:
                raise ValueError(f"{field}: error status contradicts observed content")
        else:
            if observed_hash is not None:
                raise ValueError(f"{field}: unread artifact cannot have observed SHA-256")
            if read_status == "unreadable" and error != "unreadable":
                raise ValueError(f"{field}: unreadable status requires matching error")
            if error == "none":
                raise ValueError(f"{field}: non-readable observation cannot pass")

        normalized_observations.append(
            {
                "artifact_id": expected["artifact_id"],
                "role": _enum(
                    item["role"], field=f"{field}.role", allowed=_ARTIFACT_ROLES
                ),
                "order": expected["order"],
                "expected_size_bytes": expected["expected_size_bytes"],
                "expected_sha256": expected["sha256"],
                "presence_status": presence,
                "path_type_status": path_type,
                "read_status": read_status,
                "observed_size_bytes": observed_size,
                "observed_sha256": observed_hash,
                "verified_at": verified_at,
                "error_status": error,
            }
        )
        if error != "none":
            raise ValueError(f"{field}: every artifact member must pass trusted preflight")

    normalized = {
        "schema_version": 1,
        "inventory_id": _token(
            record["inventory_id"], field="inventory_id", component=True
        ),
        "bundle_spec_digest": bundle.spec_digest,
        "artifact_set_digest": bundle.artifact_set_digest,
        "observations": normalized_observations,
        "artifact_state_fingerprint": _sha256(
            record["artifact_state_fingerprint"], field="artifact_state_fingerprint"
        ),
        "inventory_digest": _sha256(record["inventory_digest"], field="inventory_digest"),
    }
    if normalized["artifact_state_fingerprint"] != compute_artifact_state_fingerprint(
        normalized
    ):
        raise ValueError("artifact_state_fingerprint: digest mismatch")
    if normalized["inventory_digest"] != canonical_sha256_v1(
        normalized, own_digest_field="inventory_digest"
    ):
        raise ValueError("inventory_digest: digest mismatch")
    return LocalArtifactInventory(normalized)


@dataclass(frozen=True)
class QualificationReport:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def report_id(self) -> str:
        return str(self._record["report_id"])

    @property
    def report_digest(self) -> str:
        return str(self._record["report_digest"])


def _latency_interval(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="latency_interval")
    keys = frozenset(
        {
            "interval_id",
            "handoff_id",
            "handoff_version",
            "included_phases",
            "publication_boundary",
        }
    )
    _keys(record, field="latency_interval", allowed=keys, required=keys)
    phases = _list(
        record["included_phases"],
        field="latency_interval.included_phases",
        minimum=len(LATENCY_PHASES),
        maximum=len(LATENCY_PHASES),
    )
    if tuple(phases) != LATENCY_PHASES:
        raise ValueError("latency_interval: phase set/order is not comparable v1")
    if record["interval_id"] != LATENCY_INTERVAL_ID:
        raise ValueError("latency_interval.interval_id: unsupported interval")
    if record["handoff_id"] != HANDOFF_ID or record["handoff_version"] != HANDOFF_VERSION:
        raise ValueError("latency_interval: result/mask handoff mismatch")
    if record["publication_boundary"] != "managed_output_transaction_after_interval":
        raise ValueError("latency_interval: output publication must remain outside")
    return {
        "interval_id": LATENCY_INTERVAL_ID,
        "handoff_id": HANDOFF_ID,
        "handoff_version": HANDOFF_VERSION,
        "included_phases": list(LATENCY_PHASES),
        "publication_boundary": "managed_output_transaction_after_interval",
    }


def _memory_measurement(value: Any, *, field: str, accelerator: bool) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset(
        {
            "status",
            "value_bytes",
            "collector_id",
            "collector_version",
            "collector_source_digest",
            "scope",
            "covered_processes",
            "covered_devices",
        }
    )
    _keys(record, field=field, allowed=keys, required=keys)
    status = _enum(
        record["status"],
        field=f"{field}.status",
        allowed=frozenset({"known", "unknown", "not_applicable"}),
    )
    collector_id = _token(record["collector_id"], field=f"{field}.collector_id")
    collector_version = _token(
        record["collector_version"], field=f"{field}.collector_version"
    )
    collector_digest = _sha256(
        record["collector_source_digest"], field=f"{field}.collector_source_digest"
    )
    scope_allowed = (
        frozenset({"runner_process_tree_all_declared_devices", "unknown", "not_applicable"})
        if accelerator
        else frozenset({"runner_process_tree", "isolated_cgroup_or_container", "unknown"})
    )
    scope = _enum(record["scope"], field=f"{field}.scope", allowed=scope_allowed)
    covered_processes = _enum(
        record["covered_processes"],
        field=f"{field}.covered_processes",
        allowed=frozenset({"all", "unknown", "not_applicable"}),
    )
    covered_devices = _enum(
        record["covered_devices"],
        field=f"{field}.covered_devices",
        allowed=frozenset({"all_declared", "unknown", "not_applicable"}),
    )
    raw_value = record["value_bytes"]
    measured: int | None
    if status == "known":
        measured = _integer(
            raw_value, field=f"{field}.value_bytes", minimum=1, maximum=MAX_SIGNED_BYTES
        )
        if covered_processes != "all":
            raise ValueError(f"{field}: known value requires all runner processes")
        if accelerator:
            if scope != "runner_process_tree_all_declared_devices" or covered_devices != "all_declared":
                raise ValueError(f"{field}: known accelerator value requires all devices")
        elif scope not in {"runner_process_tree", "isolated_cgroup_or_container"}:
            raise ValueError(f"{field}: known RSS requires complete tree/group scope")
        elif covered_devices != "not_applicable":
            raise ValueError(f"{field}: RSS device coverage must be not_applicable")
    else:
        if raw_value is not None:
            raise ValueError(f"{field}.value_bytes: unknown/not-applicable must be null")
        measured = None
        if status == "not_applicable" and not accelerator:
            raise ValueError(f"{field}: RSS is never not_applicable")
        if status == "not_applicable" and (
            scope != "not_applicable"
            or covered_processes != "not_applicable"
            or covered_devices != "not_applicable"
        ):
            raise ValueError(f"{field}: not_applicable fields must agree")
        if status == "unknown" and (
            scope != "unknown"
            or covered_processes != "unknown"
            or covered_devices != ("unknown" if accelerator else "not_applicable")
        ):
            raise ValueError(f"{field}: unknown coverage fields must agree")
    return {
        "status": status,
        "value_bytes": measured,
        "collector_id": collector_id,
        "collector_version": collector_version,
        "collector_source_digest": collector_digest,
        "scope": scope,
        "covered_processes": covered_processes,
        "covered_devices": covered_devices,
    }


def _repeat(value: Any, *, index: int) -> dict[str, Any]:
    field = f"repeats[{index - 1}]"
    record = _mapping(value, field=field)
    common = frozenset({"repeat_index", "status", "failure_code"})
    measured = frozenset(
        {
            "sample_count",
            "duration_ns",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "throughput_processed_count",
            "throughput_duration_ns",
            "runner_tree_peak_rss",
            "accelerator_process_tree_peak",
        }
    )
    _keys(record, field=field, allowed=common | measured, required=common | measured)
    if record["repeat_index"] != index or type(record["repeat_index"]) is not int:
        raise ValueError(f"{field}.repeat_index: expected contiguous 1..3")
    status = _enum(
        record["status"], field=f"{field}.status", allowed=frozenset({"completed", "failed"})
    )
    failure = record["failure_code"]
    if status == "completed":
        if failure is not None:
            raise ValueError(f"{field}.failure_code: completed repeat requires null")
        sample_count = _integer(
            record["sample_count"], field=f"{field}.sample_count", minimum=200, maximum=200
        )
        duration = _integer(
            record["duration_ns"], field=f"{field}.duration_ns", minimum=1, maximum=MAX_UINT64
        )
        processed = _integer(
            record["throughput_processed_count"],
            field=f"{field}.throughput_processed_count",
            minimum=200,
            maximum=200,
        )
        throughput_duration = _integer(
            record["throughput_duration_ns"],
            field=f"{field}.throughput_duration_ns",
            minimum=1,
            maximum=MAX_UINT64,
        )
        if throughput_duration != duration or processed != sample_count:
            raise ValueError(f"{field}: throughput ratio must cover the exact timed samples")
        percentiles = [
            canonical_decimal_v1(record[name], field=f"{field}.{name}", nonnegative=True)
            for name in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms")
        ]
        if not Decimal(percentiles[0]) <= Decimal(percentiles[1]) <= Decimal(percentiles[2]):
            raise ValueError(f"{field}: latency percentiles are inconsistent")
    else:
        failure = _token(failure, field=f"{field}.failure_code", component=True)
        if any(record[name] is not None for name in measured - {"runner_tree_peak_rss", "accelerator_process_tree_peak"}):
            raise ValueError(f"{field}: failed repeat measurements must be null")
        sample_count = duration = processed = throughput_duration = None
        percentiles = [None, None, None]
    rss = _memory_measurement(
        record["runner_tree_peak_rss"], field=f"{field}.runner_tree_peak_rss", accelerator=False
    )
    accelerator_memory = _memory_measurement(
        record["accelerator_process_tree_peak"],
        field=f"{field}.accelerator_process_tree_peak",
        accelerator=True,
    )
    if status == "failed" and (rss["status"] == "known" or accelerator_memory["status"] == "known"):
        raise ValueError(f"{field}: failed repeat cannot publish passing peak memory")
    return {
        "repeat_index": index,
        "status": status,
        "failure_code": failure,
        "sample_count": sample_count,
        "duration_ns": duration,
        "p50_latency_ms": percentiles[0],
        "p95_latency_ms": percentiles[1],
        "p99_latency_ms": percentiles[2],
        "throughput_processed_count": processed,
        "throughput_duration_ns": throughput_duration,
        "runner_tree_peak_rss": rss,
        "accelerator_process_tree_peak": accelerator_memory,
    }


def _ratio_less(left_count: int, left_duration: int, right_count: int, right_duration: int) -> bool:
    return left_count * right_duration < right_count * left_duration


def _conservative(value: Any, repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    record = _mapping(value, field="conservative_aggregates")
    keys = frozenset(
        {
            "repeat_throughput_source_index",
            "repeat_throughput_processed_count",
            "repeat_throughput_duration_ns",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "runner_tree_peak_rss",
            "accelerator_process_tree_peak",
        }
    )
    _keys(record, field="conservative_aggregates", allowed=keys, required=keys)
    if any(item["status"] != "completed" for item in repeats):
        raise ValueError("conservative_aggregates: require three completed repeats")
    worst = 0
    for index in range(1, len(repeats)):
        if _ratio_less(
            repeats[index]["throughput_processed_count"],
            repeats[index]["throughput_duration_ns"],
            repeats[worst]["throughput_processed_count"],
            repeats[worst]["throughput_duration_ns"],
        ):
            worst = index
    expected = repeats[worst]
    if (
        record["repeat_throughput_source_index"] != worst + 1
        or record["repeat_throughput_processed_count"] != expected["throughput_processed_count"]
        or record["repeat_throughput_duration_ns"] != expected["throughput_duration_ns"]
    ):
        raise ValueError("conservative_aggregates: repeat throughput is not the minimum exact ratio")
    out: dict[str, Any] = {
        "repeat_throughput_source_index": worst + 1,
        "repeat_throughput_processed_count": expected["throughput_processed_count"],
        "repeat_throughput_duration_ns": expected["throughput_duration_ns"],
    }
    for name in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms"):
        claimed = canonical_decimal_v1(record[name], field=f"conservative_aggregates.{name}", nonnegative=True)
        maximum = max((item[name] for item in repeats), key=Decimal)
        if Decimal(claimed) != Decimal(maximum):
            raise ValueError(f"conservative_aggregates.{name}: expected cross-repeat maximum")
        out[name] = claimed
    for name, accelerator in (
        ("runner_tree_peak_rss", False),
        ("accelerator_process_tree_peak", True),
    ):
        claimed = _memory_measurement(
            record[name], field=f"conservative_aggregates.{name}", accelerator=accelerator
        )
        measurements = [item[name] for item in repeats]
        known = [item for item in measurements if item["status"] == "known"]
        if len(known) != len(measurements):
            if claimed["status"] == "known":
                raise ValueError(f"conservative_aggregates.{name}: partial known values must yield unknown")
        else:
            maximum = max(item["value_bytes"] for item in known)
            if claimed["status"] != "known" or claimed["value_bytes"] != maximum:
                raise ValueError(f"conservative_aggregates.{name}: expected maximum known value")
        out[name] = claimed
    return out


def _cold_start(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="cold_start")
    keys = frozenset(
        {"status", "cold_start_ms", "failure_code", "fresh_runner", "os_cache_state", "interval_id"}
    )
    _keys(record, field="cold_start", allowed=keys, required=keys)
    status = _enum(record["status"], field="cold_start.status", allowed=frozenset({"known", "failed"}))
    if record["fresh_runner"] is not True or record["os_cache_state"] != "uncontrolled":
        raise ValueError("cold_start: requires a fresh runner and uncontrolled OS cache claim")
    if record["interval_id"] != LATENCY_INTERVAL_ID:
        raise ValueError("cold_start.interval_id: latency interval mismatch")
    if status == "known":
        value_ms = canonical_decimal_v1(record["cold_start_ms"], field="cold_start.cold_start_ms", nonnegative=True)
        if record["failure_code"] is not None:
            raise ValueError("cold_start.failure_code: known result requires null")
        failure = None
    else:
        if record["cold_start_ms"] is not None:
            raise ValueError("cold_start.cold_start_ms: failed result requires null")
        value_ms = None
        failure = _token(record["failure_code"], field="cold_start.failure_code", component=True)
    return {
        "status": status,
        "cold_start_ms": value_ms,
        "failure_code": failure,
        "fresh_runner": True,
        "os_cache_state": "uncontrolled",
        "interval_id": LATENCY_INTERVAL_ID,
    }


def _warmup(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="warmup")
    keys = frozenset({"status", "iteration_count", "failure_code"})
    _keys(record, field="warmup", allowed=keys, required=keys)
    status = _enum(
        record["status"],
        field="warmup.status",
        allowed=frozenset({"completed", "failed"}),
    )
    if status == "completed":
        iterations = _integer(
            record["iteration_count"],
            field="warmup.iteration_count",
            minimum=20,
            maximum=20,
        )
        if record["failure_code"] is not None:
            raise ValueError("warmup.failure_code: completed warm-up requires null")
        failure = None
    else:
        if record["iteration_count"] is not None:
            raise ValueError("warmup.iteration_count: failed warm-up requires null")
        iterations = None
        failure = _token(
            record["failure_code"], field="warmup.failure_code", component=True
        )
    return {
        "status": status,
        "iteration_count": iterations,
        "failure_code": failure,
    }


def _lifetime_memory(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="lifetime_memory")
    keys = frozenset(
        {
            "interval_scope",
            "runner_tree_peak_rss",
            "accelerator_process_tree_peak",
        }
    )
    _keys(record, field="lifetime_memory", allowed=keys, required=keys)
    if record["interval_scope"] != "fresh_runner_creation_through_close":
        raise ValueError("lifetime_memory: incomplete collection interval")
    return {
        "interval_scope": "fresh_runner_creation_through_close",
        "runner_tree_peak_rss": _memory_measurement(
            record["runner_tree_peak_rss"],
            field="lifetime_memory.runner_tree_peak_rss",
            accelerator=False,
        ),
        "accelerator_process_tree_peak": _memory_measurement(
            record["accelerator_process_tree_peak"],
            field="lifetime_memory.accelerator_process_tree_peak",
            accelerator=True,
        ),
    }


def _observation(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset({"status", "value"})
    _keys(record, field=field, allowed=keys, required=keys)
    status = _enum(record["status"], field=f"{field}.status", allowed=frozenset({"known", "unknown"}))
    if status == "known":
        observed = _token(record["value"], field=f"{field}.value", component=True)
    else:
        if record["value"] is not None:
            raise ValueError(f"{field}.value: unknown observation requires null")
        observed = None
    return {"status": status, "value": observed}


def _sustained(value: Any, *, execution_mode: str) -> dict[str, Any]:
    record = _mapping(value, field="sustained_section")
    if execution_mode == "batch":
        keys = frozenset({"status", "reason"})
        _keys(record, field="sustained_section", allowed=keys, required=keys)
        if record != {"status": "not_required", "reason": "batch_profile"}:
            raise ValueError("sustained_section: batch requires explicit not_required reason")
        return dict(record)
    keys = frozenset(
        {
            "status",
            "failure_code",
            "schedule_reset_index",
            "duration_ns",
            "processed_count",
            "sample_count",
            "max_sustained_samples",
            "sample_storage_bytes",
            "aggregation_method",
            "p95_latency_ms",
            "p99_latency_ms",
            "throughput_processed_count",
            "throughput_duration_ns",
            "runner_tree_peak_rss",
            "accelerator_process_tree_peak",
            "queue_status",
            "drop_status",
            "power_observation",
            "thermal_observation",
            "warmup_excluded",
            "cold_start_excluded",
            "repeat_samples_excluded",
        }
    )
    _keys(record, field="sustained_section", allowed=keys, required=keys)
    status = _enum(record["status"], field="sustained_section.status", allowed=frozenset({"completed", "failed"}))
    if record["schedule_reset_index"] != 0 or type(record["schedule_reset_index"]) is not int:
        raise ValueError("sustained_section.schedule_reset_index: expected 0")
    max_samples = _integer(
        record["max_sustained_samples"],
        field="sustained_section.max_sustained_samples",
        minimum=MAX_SUSTAINED_SAMPLES,
        maximum=MAX_SUSTAINED_SAMPLES,
    )
    if record["aggregation_method"] != "exact_nearest_rank_all_samples":
        raise ValueError("sustained_section: approximation or sampling is forbidden")
    if any(record[name] is not True for name in ("warmup_excluded", "cold_start_excluded", "repeat_samples_excluded")):
        raise ValueError("sustained_section: warm-up/cold-start/repeat samples must be excluded")
    if record["queue_status"] != "not_applicable" or record["drop_status"] != "not_applicable":
        raise ValueError("sustained_section: static sequential queue/drop is not_applicable")
    if status == "completed":
        if record["failure_code"] is not None:
            raise ValueError("sustained_section.failure_code: completed section requires null")
        duration = _integer(
            record["duration_ns"], field="sustained_section.duration_ns", minimum=MIN_SUSTAINED_DURATION_NS, maximum=MAX_UINT64
        )
        processed = _integer(
            record["processed_count"], field="sustained_section.processed_count", minimum=1, maximum=max_samples
        )
        samples = _integer(
            record["sample_count"], field="sustained_section.sample_count", minimum=1, maximum=max_samples
        )
        if samples != processed:
            raise ValueError("sustained_section: every retained latency must be a successful handoff")
        storage = _integer(
            record["sample_storage_bytes"], field="sustained_section.sample_storage_bytes", minimum=8, maximum=MAX_SUSTAINED_SAMPLE_BYTES
        )
        if storage != samples * 8:
            raise ValueError("sustained_section: sample storage must be exact preallocated uint64 coverage")
        if record["throughput_processed_count"] != processed or record["throughput_duration_ns"] != duration:
            raise ValueError("sustained_section: throughput ratio must use full section")
        p95 = canonical_decimal_v1(record["p95_latency_ms"], field="sustained_section.p95_latency_ms", nonnegative=True)
        p99 = canonical_decimal_v1(record["p99_latency_ms"], field="sustained_section.p99_latency_ms", nonnegative=True)
        if Decimal(p95) > Decimal(p99):
            raise ValueError("sustained_section: p95 exceeds p99")
        failure = None
    else:
        failure = _token(record["failure_code"], field="sustained_section.failure_code", component=True)
        for name in (
            "duration_ns", "processed_count", "sample_count", "sample_storage_bytes",
            "p95_latency_ms", "p99_latency_ms", "throughput_processed_count", "throughput_duration_ns",
        ):
            if record[name] is not None:
                raise ValueError(f"sustained_section.{name}: failed section requires null")
        duration = processed = samples = storage = p95 = p99 = None
    rss = _memory_measurement(record["runner_tree_peak_rss"], field="sustained_section.runner_tree_peak_rss", accelerator=False)
    accelerator_memory = _memory_measurement(
        record["accelerator_process_tree_peak"], field="sustained_section.accelerator_process_tree_peak", accelerator=True
    )
    if status == "failed" and (rss["status"] == "known" or accelerator_memory["status"] == "known"):
        raise ValueError("sustained_section: failed section cannot publish passing memory")
    return {
        "status": status,
        "failure_code": failure,
        "schedule_reset_index": 0,
        "duration_ns": duration,
        "processed_count": processed,
        "sample_count": samples,
        "max_sustained_samples": max_samples,
        "sample_storage_bytes": storage,
        "aggregation_method": "exact_nearest_rank_all_samples",
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "throughput_processed_count": processed,
        "throughput_duration_ns": duration,
        "runner_tree_peak_rss": rss,
        "accelerator_process_tree_peak": accelerator_memory,
        "queue_status": "not_applicable",
        "drop_status": "not_applicable",
        "power_observation": _observation(record["power_observation"], field="sustained_section.power_observation"),
        "thermal_observation": _observation(record["thermal_observation"], field="sustained_section.thermal_observation"),
        "warmup_excluded": True,
        "cold_start_excluded": True,
        "repeat_samples_excluded": True,
    }


def _quality(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="quality")
    status = record.get("status")
    if status == "not_required":
        keys = frozenset({"status", "reason"})
        _keys(record, field="quality", allowed=keys, required=keys)
        if record["reason"] != "request_has_no_quality_requirement":
            raise ValueError("quality.reason: invalid not_required reason")
        return {
            "status": "not_required",
            "reason": "request_has_no_quality_requirement",
        }
    if status == "unknown":
        keys = frozenset({"status", "reason"})
        _keys(record, field="quality", allowed=keys, required=keys)
        return {
            "status": "unknown",
            "reason": _enum(
                record["reason"],
                field="quality.reason",
                allowed=frozenset({"no_matching_evaluator", "vocabulary_mismatch"}),
            ),
        }
    if status != "known":
        raise ValueError("quality.status: unsupported value")
    keys = frozenset(
        {
            "status", "metric_id", "direction", "measured_value", "threshold_context",
            "evaluation_dataset_id", "evaluation_dataset_sha256", "evaluation_protocol_sha256",
            "evaluation_vocabulary_id", "predictions_source",
        }
    )
    _keys(record, field="quality", allowed=keys, required=keys)
    if record["predictions_source"] != "same_qualification_run":
        raise ValueError("quality.predictions_source: must bind the same qualification run")
    return {
        "status": "known",
        "metric_id": _token(record["metric_id"], field="quality.metric_id"),
        "direction": _enum(record["direction"], field="quality.direction", allowed=frozenset({"higher_is_better", "lower_is_better"})),
        "measured_value": canonical_decimal_v1(record["measured_value"], field="quality.measured_value"),
        "threshold_context": canonical_decimal_v1(record["threshold_context"], field="quality.threshold_context"),
        "evaluation_dataset_id": _token(record["evaluation_dataset_id"], field="quality.evaluation_dataset_id"),
        "evaluation_dataset_sha256": _sha256(record["evaluation_dataset_sha256"], field="quality.evaluation_dataset_sha256"),
        "evaluation_protocol_sha256": _sha256(record["evaluation_protocol_sha256"], field="quality.evaluation_protocol_sha256"),
        "evaluation_vocabulary_id": _token(record["evaluation_vocabulary_id"], field="quality.evaluation_vocabulary_id"),
        "predictions_source": "same_qualification_run",
    }


def _bounded_text_list(value: Any, *, field: str, maximum: int) -> list[str]:
    entries = _list(value, field=field, minimum=0, maximum=maximum)
    return [_safe_text(item, field=f"{field}[]", maximum_bytes=512) for item in entries]


def validate_qualification_report(
    value: Mapping[str, Any], *, as_of: str | datetime | None = None
) -> QualificationReport:
    """Validate one measured QualificationReport and its current freshness."""

    record = _mapping(value, field="QualificationReport")
    keys = frozenset(
        {
            "schema_version", "report_id", "report_digest", "collector", "issuer",
            "status", "task", "execution_mode", "bundle_spec_digest", "artifact_set_digest",
            "artifact_state_fingerprint", "environment_fingerprint",
            "qualification_workload_fingerprint", "protocol_fingerprint", "latency_interval",
            "started_at", "completed_at", "valid_until", "repeats", "conservative_aggregates",
            "cold_start", "warmup", "lifetime_memory", "sustained_section", "quality", "resolved_pipeline",
            "source_runtime_provenance", "limitations", "failures",
        }
    )
    _keys(record, field="QualificationReport", allowed=keys, required=keys)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("QualificationReport.schema_version: expected 1")
    current = _as_of(as_of)
    started_text, started = _utc(record["started_at"], field="started_at")
    completed_text, completed = _utc(record["completed_at"], field="completed_at")
    valid_text, valid = _utc(record["valid_until"], field="valid_until")
    if started > completed:
        raise ValueError("QualificationReport: reversed collection interval")
    if completed > current:
        raise ValueError("QualificationReport: future-dated evidence")
    if valid <= completed or valid > completed + QUALIFICATION_VALIDITY:
        raise ValueError("valid_until: must be later than completion and within 90 days")
    if current >= valid:
        raise ValueError("QualificationReport: evidence is expired")

    collector = _identity(record["collector"], field="collector")
    issuer = _identity(record["issuer"], field="issuer")
    execution_mode = _enum(record["execution_mode"], field="execution_mode", allowed=frozenset({"batch", "soft_realtime"}))
    status = _enum(record["status"], field="status", allowed=_REPORT_STATUSES)
    repeats = [_repeat(item, index=index + 1) for index, item in enumerate(_list(record["repeats"], field="repeats", minimum=3, maximum=3))]
    conservative: dict[str, Any] | None
    if record["conservative_aggregates"] is None:
        conservative = None
    else:
        conservative = _conservative(record["conservative_aggregates"], repeats)
    cold = _cold_start(record["cold_start"])
    warmup = _warmup(record["warmup"])
    lifetime_memory = _lifetime_memory(record["lifetime_memory"])
    sustained = _sustained(record["sustained_section"], execution_mode=execution_mode)
    if status == "qualified":
        if conservative is None or any(item["status"] != "completed" for item in repeats):
            raise ValueError("qualified report requires exactly three completed repeats and conservative aggregates")
        if cold["status"] != "known":
            raise ValueError("qualified report requires known cold start")
        if warmup["status"] != "completed":
            raise ValueError("qualified report requires exactly 20 successful warm-up iterations")
        if execution_mode == "soft_realtime" and sustained["status"] != "completed":
            raise ValueError("qualified soft_realtime report requires complete ten-minute section")
    elif status == "smoke" and conservative is not None:
        raise ValueError("smoke report cannot carry qualified conservative aggregates")

    resolved = _mapping(record["resolved_pipeline"], field="resolved_pipeline")
    resolved_keys = frozenset({"decoder", "model_input", "preprocess", "postprocess"})
    _keys(resolved, field="resolved_pipeline", allowed=resolved_keys, required=resolved_keys)
    provenance = _mapping(record["source_runtime_provenance"], field="source_runtime_provenance")
    provenance_keys = frozenset(
        {"model_source_id", "model_revision", "runtime_id", "runtime_version", "provider_id", "provider_version"}
    )
    _keys(provenance, field="source_runtime_provenance", allowed=provenance_keys, required=provenance_keys)
    normalized = {
        "schema_version": 1,
        "report_id": _token(record["report_id"], field="report_id", component=True),
        "report_digest": _sha256(record["report_digest"], field="report_digest"),
        "collector": collector,
        "issuer": issuer,
        "status": status,
        "task": _enum(record["task"], field="task", allowed=frozenset({"object_detection", "instance_segmentation"})),
        "execution_mode": execution_mode,
        "bundle_spec_digest": _sha256(record["bundle_spec_digest"], field="bundle_spec_digest"),
        "artifact_set_digest": _sha256(record["artifact_set_digest"], field="artifact_set_digest"),
        "artifact_state_fingerprint": _sha256(record["artifact_state_fingerprint"], field="artifact_state_fingerprint"),
        "environment_fingerprint": _sha256(record["environment_fingerprint"], field="environment_fingerprint"),
        "qualification_workload_fingerprint": _sha256(record["qualification_workload_fingerprint"], field="qualification_workload_fingerprint"),
        "protocol_fingerprint": _sha256(record["protocol_fingerprint"], field="protocol_fingerprint"),
        "latency_interval": _latency_interval(record["latency_interval"]),
        "started_at": started_text,
        "completed_at": completed_text,
        "valid_until": valid_text,
        "repeats": repeats,
        "conservative_aggregates": conservative,
        "cold_start": cold,
        "warmup": warmup,
        "lifetime_memory": lifetime_memory,
        "sustained_section": sustained,
        "quality": _quality(record["quality"]),
        "resolved_pipeline": {name: _identity(resolved[name], field=f"resolved_pipeline.{name}") for name in ("decoder", "model_input", "preprocess", "postprocess")},
        "source_runtime_provenance": {
            name: _safe_text(provenance[name], field=f"source_runtime_provenance.{name}", maximum_bytes=512 if name == "model_source_id" else 128)
            for name in ("model_source_id", "model_revision", "runtime_id", "runtime_version", "provider_id", "provider_version")
        },
        "limitations": _bounded_text_list(record["limitations"], field="limitations", maximum=32),
        "failures": _bounded_text_list(record["failures"], field="failures", maximum=32),
    }
    expected_digest = canonical_sha256_v1(normalized, own_digest_field="report_digest")
    if normalized["report_digest"] != expected_digest:
        raise ValueError("report_digest: digest mismatch")
    return QualificationReport(normalized)


def compute_evidence_selection_key(
    *,
    bundle_spec_digest: str,
    artifact_set_digest: str,
    environment_fingerprint: str,
    qualification_workload_fingerprint: str,
    protocol_fingerprint: str,
) -> str:
    """Compute the canonical immutable qualification selection key."""

    return canonical_sha256_v1(
        {
            "bundle_spec_digest": _sha256(bundle_spec_digest, field="bundle_spec_digest"),
            "artifact_set_digest": _sha256(artifact_set_digest, field="artifact_set_digest"),
            "environment_fingerprint": _sha256(environment_fingerprint, field="environment_fingerprint"),
            "qualification_workload_fingerprint": _sha256(
                qualification_workload_fingerprint, field="qualification_workload_fingerprint"
            ),
            "protocol_fingerprint": _sha256(protocol_fingerprint, field="protocol_fingerprint"),
        }
    )


@dataclass(frozen=True)
class EvidenceActivationRecord:
    _record: dict[str, Any]
    source_trust_domain: str

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def event_digest(self) -> str:
        return str(self._record["event_digest"])


@dataclass(frozen=True)
class EvidenceActivationProjection:
    active_by_selection_key: dict[str, QualificationReport]
    terminal_reason_by_selection_key: dict[str, str]
    head_by_selection_key: dict[str, str]
    events: tuple[EvidenceActivationRecord, ...]


def _review_reference(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="review_reference")
    if record.get("kind") == "public_repository_id":
        keys = frozenset({"kind", "value"})
        _keys(record, field="review_reference", allowed=keys, required=keys)
        return {"kind": "public_repository_id", "value": _token(record["value"], field="review_reference.value")}
    if record.get("kind") == "site_local_status":
        keys = frozenset({"kind", "status"})
        _keys(record, field="review_reference", allowed=keys, required=keys)
        return {"kind": "site_local_status", "status": _enum(record["status"], field="review_reference.status", allowed=frozenset({"present", "not_applicable"}))}
    raise ValueError("review_reference.kind: unsupported value")


def _derived_trust(issuer_claim: str, reviewer_role: str, reference: Mapping[str, Any]) -> str:
    if issuer_claim == "repository_source":
        if reviewer_role not in {"repo_maintainer", "release_reviewer"} or reference["kind"] != "public_repository_id":
            raise ValueError("repository_source: requires repository review role/reference")
        return "yolozu_managed"
    if issuer_claim == "site_source":
        if reviewer_role not in {"site_operator", "release_reviewer"} or reference["kind"] != "site_local_status" or reference["status"] != "present":
            raise ValueError("site_source: requires present site-local review")
        return "site_managed"
    if issuer_claim == "operator_source":
        return "operator_asserted"
    return "unknown"


def validate_evidence_activation_record(
    value: Mapping[str, Any], *, source_trust_domain: str = "operator_asserted"
) -> EvidenceActivationRecord:
    """Validate one append-only evidence activation event."""

    record = _mapping(value, field="EvidenceActivationRecord")
    keys = frozenset(
        {
            "schema_version", "stream_id", "selection_key", "sequence", "previous_event_digest",
            "event_id", "report_id", "report_digest", "state", "replacement_report_id",
            "replacement_report_digest", "activated_at", "valid_until", "reviewer_role_id",
            "review_reference", "issuer_claim", "trust_domain", "reason", "event_digest",
        }
    )
    _keys(record, field="EvidenceActivationRecord", allowed=keys, required=keys)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("EvidenceActivationRecord.schema_version: expected 1")
    selection_key = _sha256(record["selection_key"], field="selection_key")
    if record["stream_id"] != selection_key:
        raise ValueError("stream_id: must equal the canonical selection key")
    sequence = _integer(record["sequence"], field="sequence", minimum=1, maximum=MAX_EVIDENCE_ACTIVATION_RECORDS)
    previous = _sha256(record["previous_event_digest"], field="previous_event_digest", allow_zero=True)
    if (sequence == 1) != (previous == ZERO_DIGEST):
        raise ValueError("previous_event_digest: zero sentinel only for sequence 1")
    state = _enum(record["state"], field="state", allowed=frozenset({"active", "superseded", "revoked"}))
    replacement_id = record["replacement_report_id"]
    replacement_digest = record["replacement_report_digest"]
    if state == "superseded":
        replacement_id = _token(replacement_id, field="replacement_report_id", component=True)
        replacement_digest = _sha256(replacement_digest, field="replacement_report_digest")
    elif replacement_id is not None or replacement_digest is not None:
        raise ValueError("replacement report is valid only for superseded state")
    reviewer = _token(record["reviewer_role_id"], field="reviewer_role_id")
    if _ROLE_RE.fullmatch(reviewer) is None:
        raise ValueError("reviewer_role_id: personal or unsupported role")
    reference = _review_reference(record["review_reference"])
    issuer_claim = _enum(record["issuer_claim"], field="issuer_claim", allowed=frozenset({"repository_source", "site_source", "operator_source", "unknown"}))
    derived = _derived_trust(issuer_claim, reviewer, reference)
    claimed_trust = _enum(record["trust_domain"], field="trust_domain", allowed=_TRUST_DOMAINS)
    if claimed_trust != derived:
        raise ValueError("trust_domain: does not match reviewed source authority")
    supplied_trust = _enum(source_trust_domain, field="source_trust_domain", allowed=_TRUST_DOMAINS)
    if supplied_trust != derived:
        raise ValueError("source_trust_domain: loader authority contradicts event")
    activated_at = _utc(record["activated_at"], field="activated_at")[0]
    valid_until = _utc(record["valid_until"], field="valid_until")[0]
    normalized = {
        "schema_version": 1,
        "stream_id": selection_key,
        "selection_key": selection_key,
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_id": _token(record["event_id"], field="event_id", component=True),
        "report_id": _token(record["report_id"], field="report_id", component=True),
        "report_digest": _sha256(record["report_digest"], field="report_digest"),
        "state": state,
        "replacement_report_id": replacement_id,
        "replacement_report_digest": replacement_digest,
        "activated_at": activated_at,
        "valid_until": valid_until,
        "reviewer_role_id": reviewer,
        "review_reference": reference,
        "issuer_claim": issuer_claim,
        "trust_domain": derived,
        "reason": _safe_text(record["reason"], field="reason", maximum_bytes=512),
        "event_digest": _sha256(record["event_digest"], field="event_digest"),
    }
    if normalized["event_digest"] != canonical_sha256_v1(normalized, own_digest_field="event_digest"):
        raise ValueError("event_digest: digest mismatch")
    return EvidenceActivationRecord(normalized, supplied_trust)


def project_evidence_activations(
    records: Sequence[Mapping[str, Any] | EvidenceActivationRecord],
    reports: Sequence[Mapping[str, Any] | QualificationReport],
    *,
    source_trust_domain: str,
    as_of: str | datetime | None = None,
) -> EvidenceActivationProjection:
    """Project complete per-key streams strictly by sequence, never timestamp."""

    if len(records) > MAX_EVIDENCE_ACTIVATION_RECORDS:
        raise ValueError("evidence activation stream exceeds 8192 records")
    current = _as_of(as_of)
    report_by_identity: dict[tuple[str, str], QualificationReport] = {}
    report_selection_keys: dict[tuple[str, str], str] = {}
    for item in reports:
        report = item if isinstance(item, QualificationReport) else validate_qualification_report(item, as_of=current)
        data = report.to_dict()
        identity = (report.report_id, report.report_digest)
        if identity in report_by_identity:
            raise ValueError("qualification reports contain duplicate ID/digest")
        report_by_identity[identity] = report
        report_selection_keys[identity] = compute_evidence_selection_key(
            bundle_spec_digest=data["bundle_spec_digest"],
            artifact_set_digest=data["artifact_set_digest"],
            environment_fingerprint=data["environment_fingerprint"],
            qualification_workload_fingerprint=data["qualification_workload_fingerprint"],
            protocol_fingerprint=data["protocol_fingerprint"],
        )

    streams: dict[str, list[EvidenceActivationRecord]] = {}
    event_ids: set[str] = set()
    validated_events: list[EvidenceActivationRecord] = []
    for item in records:
        event = item if isinstance(item, EvidenceActivationRecord) else validate_evidence_activation_record(item, source_trust_domain=source_trust_domain)
        data = event.to_dict()
        if data["event_id"] in event_ids:
            raise ValueError("evidence activation stream has duplicate event_id")
        event_ids.add(data["event_id"])
        streams.setdefault(data["selection_key"], []).append(event)
        validated_events.append(event)

    active_by_key: dict[str, QualificationReport] = {}
    terminal_reason: dict[str, str] = {}
    heads: dict[str, str] = {}
    for selection_key, events in streams.items():
        previous = ZERO_DIGEST
        previous_event_time: datetime | None = None
        active: tuple[str, str] | None = None
        active_valid_until: datetime | None = None
        retired: set[tuple[str, str]] = set()
        pending_replacement: tuple[str, str] | None = None
        for expected_sequence, event in enumerate(events, start=1):
            data = event.to_dict()
            if data["sequence"] != expected_sequence:
                raise ValueError("evidence activation stream has sequence gap or duplicate")
            if data["previous_event_digest"] != previous:
                raise ValueError("evidence activation stream has predecessor gap or fork")
            activated = _utc(data["activated_at"], field="activated_at")[1]
            valid_until = _utc(data["valid_until"], field="valid_until")[1]
            if activated > current:
                raise ValueError("evidence activation event is future-dated")
            if previous_event_time is not None and activated < previous_event_time:
                raise ValueError("evidence activation event time reverses sequence order")
            identity = (data["report_id"], data["report_digest"])
            report = report_by_identity.get(identity)
            if report is None:
                raise ValueError("evidence activation references missing qualification report")
            if report_selection_keys[identity] != selection_key:
                raise ValueError("evidence activation selection key does not match report")
            report_data = report.to_dict()
            report_completed = _utc(report_data["completed_at"], field="completed_at")[1]
            report_valid_until = _utc(report_data["valid_until"], field="valid_until")[1]
            if not report_completed <= activated < valid_until <= report_valid_until:
                raise ValueError("evidence activation validity contradicts report/event time")
            if data["state"] == "active":
                if report_data["status"] != "qualified":
                    raise ValueError("only a qualified report can become active")
                if identity in retired:
                    raise ValueError("superseded/revoked report cannot reactivate")
                if active is not None:
                    raise ValueError("multiple active reports for one selection key")
                if pending_replacement is not None and identity != pending_replacement:
                    raise ValueError("active report does not complete pending supersession")
                active = identity
                active_valid_until = valid_until
                pending_replacement = None
                terminal_reason.pop(selection_key, None)
            elif data["state"] == "superseded":
                if active != identity:
                    raise ValueError("superseded event does not target the active report")
                retired.add(identity)
                active = None
                active_valid_until = None
                pending_replacement = (
                    data["replacement_report_id"], data["replacement_report_digest"]
                )
                terminal_reason[selection_key] = "evidence_inactive"
            else:
                if active != identity:
                    raise ValueError("revoked event does not target the active report")
                retired.add(identity)
                active = None
                active_valid_until = None
                pending_replacement = None
                terminal_reason[selection_key] = "evidence_revoked"
            previous = data["event_digest"]
            previous_event_time = activated
        if pending_replacement is not None:
            raise ValueError("evidence activation stream ends with dangling supersession")
        heads[selection_key] = previous
        if active is not None:
            report = report_by_identity[active]
            if active_valid_until is None or current >= active_valid_until:
                raise ValueError("active evidence activation is expired")
            active_by_key[selection_key] = report
        elif selection_key not in terminal_reason:
            raise ValueError("evidence activation stream has invalid zero-active projection")
    return EvidenceActivationProjection(active_by_key, terminal_reason, heads, tuple(validated_events))


def load_evidence_activation_jsonl_bytes(data: bytes) -> list[Any]:
    """Parse the complete bounded public/site evidence activation stream."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > MAX_EVIDENCE_ACTIVATION_BYTES:
        raise ValueError("evidence activation stream exceeds 64 MiB")
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise ValueError("evidence activation stream has a partial suffix")
    lines = data[:-1].split(b"\n")
    if len(lines) > MAX_EVIDENCE_ACTIVATION_RECORDS:
        raise ValueError("evidence activation stream exceeds 8192 records")
    if any(not line.strip() for line in lines):
        raise ValueError("evidence activation stream contains a blank record")
    return [
        load_bounded_json_bytes(line, label=f"evidence activation:{index}")
        for index, line in enumerate(lines, start=1)
    ]


def load_evidence_activation_jsonl(path: Path) -> list[Any]:
    """Read a complete non-symlink evidence activation stream."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("evidence activation: expected regular non-symlink file")
    if candidate.stat().st_size > MAX_EVIDENCE_ACTIVATION_BYTES:
        raise ValueError("evidence activation stream exceeds 64 MiB")
    return load_evidence_activation_jsonl_bytes(candidate.read_bytes())
