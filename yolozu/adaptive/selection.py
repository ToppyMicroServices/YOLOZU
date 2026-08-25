"""Strict selection-decision interface contracts for adaptive routing.

This module validates bounded observations and decision records.  It does not
open control files, choose a bundle, or execute a runner.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .bundles import (
    AlgorithmBundleRegistry,
    AlgorithmBundleSpec,
    build_fixed_class_mapping,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
)
from .canonical import canonical_json_v1, canonical_sha256_v1

__all__ = [
    "CANDIDATE_REASON_CODES",
    "MAX_SELECTION_DECISION_BYTES",
    "ScreeningEligibilityObservation",
    "SelectionDecision",
    "SupportProfileEligibilityObservation",
    "validate_screening_eligibility_observation",
    "validate_selection_decision",
    "validate_support_profile_eligibility_observation",
]


MAX_SELECTION_DECISION_BYTES = 1_048_576
MAX_CANDIDATES = 128
MAX_REASONS = 32
MAX_TRACE_STEPS = 32

TRUST_DOMAINS = frozenset(
    {"yolozu_managed", "site_managed", "operator_asserted", "unknown"}
)
CHANNELS = frozenset({"Candidate", "Experimental", "Stable"})
SUPPORT_SCOPES = frozenset({"public_qualified", "site_qualified", "none"})
RANKING_POLICIES = frozenset(
    {"accuracy_first", "latency_first", "throughput_first", "memory_first"}
)
SCREENING_STATUSES = frozenset(
    {
        "not_applicable",
        "current_pass",
        "current_hold",
        "current_reject",
        "absent",
        "untrusted",
        "conflict",
        "revision_mismatch",
    }
)
SUPPORT_STATUSES = frozenset(
    {"matching_one", "no_match", "absent", "untrusted", "conflict", "not_required_site"}
)

CANDIDATE_REASON_CODES = frozenset(
    {
        "task_mismatch",
        "prompt_mode_mismatch",
        "class_vocabulary_mismatch",
        "evaluation_dataset_mismatch",
        "evaluation_vocabulary_mismatch",
        "bundle_spec_mismatch",
        "environment_mismatch",
        "qualification_workload_mismatch",
        "protocol_mismatch",
        "test_only",
        "bundle_disabled",
        "bundle_revoked",
        "maturity_disallowed",
        "registry_untrusted",
        "lifecycle_untrusted",
        "screening_untrusted",
        "screening_not_current_pass",
        "support_profile_mismatch",
        "support_profile_untrusted",
        "support_profile_conflict",
        "license_not_approved",
        "license_not_allowed",
        "network_required",
        "isolation_required",
        "isolation_unsupported",
        "isolation_image_missing",
        "isolation_policy_mismatch",
        "unsafe_loader_on_host",
        "compute_policy_mismatch",
        "provider_not_allowed",
        "precision_not_allowed",
        "hardware_unavailable",
        "hardware_probe_unknown",
        "runtime_unavailable",
        "runtime_probe_unknown",
        "artifact_size_limit_exceeded",
        "artifact_member_missing",
        "artifact_member_mismatch",
        "artifact_state_mismatch",
        "evidence_not_qualified",
        "evidence_untrusted",
        "evidence_inactive",
        "evidence_revoked",
        "evidence_expired",
        "evidence_superseded",
        "evidence_conflict",
        "evidence_future_dated",
        "requested_metric_unknown",
        "ranking_metric_unknown",
        "cold_start_unknown",
        "cold_start_above_requirement",
        "execution_mode_metric_mismatch",
        "quality_gate_failed",
        "repeat_throughput_gate_failed",
        "sustained_fps_gate_failed",
        "p95_latency_gate_failed",
        "peak_rss_gate_failed",
        "accelerator_memory_gate_failed",
        "catalog_only",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,63}\Z")
_DETAIL_RE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,255}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

_CANDIDATE_SUMMARIES = {
    "selected": "Selected after all required checks passed.",
    "eligible_not_selected": "Eligible, but another candidate ranked first.",
    "excluded": "Excluded because one or more required checks failed.",
}
_DECISION_SUMMARIES = {
    "selected": "Selected one qualified bundle after all required checks passed.",
    "abstained": "No eligible qualified bundle matched all required checks.",
}


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


def _token(
    value: Any,
    *,
    field: str,
    pattern: re.Pattern[str] = _ID_RE,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field}: invalid identifier")
    return value


def _nullable_token(
    value: Any,
    *,
    field: str,
    pattern: re.Pattern[str] = _ID_RE,
) -> str | None:
    if value is None:
        return None
    return _token(value, field=field, pattern=pattern)


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")
    return value


def _nullable_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


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


def _nullable_text(value: Any, *, field: str, maximum_bytes: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty text")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field}: exceeds {maximum_bytes} UTF-8 bytes")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{field}: control characters are invalid")
    return value


@dataclass(frozen=True)
class ScreeningEligibilityObservation:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def observation_digest(self) -> str:
        return str(self._record["observation_digest"])


def _as_bundle(value: AlgorithmBundleSpec | Mapping[str, Any]) -> AlgorithmBundleSpec:
    if isinstance(value, AlgorithmBundleSpec):
        return value
    return validate_algorithm_bundle_spec(value)


def validate_screening_eligibility_observation(
    value: Mapping[str, Any],
    *,
    bundle: AlgorithmBundleSpec | Mapping[str, Any] | None = None,
    source_trust_domain: str | None = None,
) -> ScreeningEligibilityObservation:
    """Validate one bounded screening projection input without opening its stream."""

    record = _mapping(value, field="ScreeningEligibilityObservation")
    keys = frozenset(
        {
            "schema_version",
            "provider_id",
            "provider_version",
            "provenance_class",
            "screening_stream_key",
            "source_revision",
            "status",
            "current_record_id",
            "current_record_digest",
            "projection_head_digest",
            "trust_domain",
            "observation_digest",
        }
    )
    _keys(record, field="ScreeningEligibilityObservation", allowed=keys, required=keys)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("ScreeningEligibilityObservation.schema_version: expected 1")
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "provider_id": _token(record["provider_id"], field="provider_id"),
        "provider_version": _token(record["provider_version"], field="provider_version"),
        "provenance_class": _enum(
            record["provenance_class"],
            field="provenance_class",
            allowed=frozenset({"existing_code_owned", "screened_candidate"}),
        ),
        "screening_stream_key": _nullable_token(
            record["screening_stream_key"], field="screening_stream_key"
        ),
        "source_revision": _nullable_text(
            record["source_revision"], field="source_revision", maximum_bytes=256
        ),
        "status": _enum(record["status"], field="status", allowed=SCREENING_STATUSES),
        "current_record_id": _nullable_token(
            record["current_record_id"], field="current_record_id"
        ),
        "current_record_digest": _nullable_sha256(
            record["current_record_digest"], field="current_record_digest"
        ),
        "projection_head_digest": _nullable_sha256(
            record["projection_head_digest"], field="projection_head_digest"
        ),
        "trust_domain": _enum(
            record["trust_domain"], field="trust_domain", allowed=TRUST_DOMAINS
        ),
        "observation_digest": _sha256(
            record["observation_digest"], field="observation_digest"
        ),
    }
    if source_trust_domain is not None:
        derived = _enum(
            source_trust_domain,
            field="source_trust_domain",
            allowed=TRUST_DOMAINS,
        )
        if normalized["trust_domain"] != derived:
            raise ValueError("trust_domain: does not match loader-derived trust")

    unique_statuses = {"current_pass", "current_hold", "current_reject"}
    current_values = (
        normalized["current_record_id"],
        normalized["current_record_digest"],
        normalized["projection_head_digest"],
    )
    if normalized["status"] in unique_statuses:
        if any(item is None for item in current_values):
            raise ValueError("current screening state requires complete record identity")
    elif any(item is not None for item in current_values):
        raise ValueError("non-unique screening state forbids current record identity")

    if normalized["provenance_class"] == "existing_code_owned":
        if normalized["status"] != "not_applicable":
            raise ValueError("existing_code_owned requires not_applicable screening")
        if normalized["provider_id"] != "no_screening_required" or normalized[
            "provider_version"
        ] != "1":
            raise ValueError("existing_code_owned requires the no-screening provider")
        if normalized["screening_stream_key"] is not None or normalized[
            "source_revision"
        ] is not None:
            raise ValueError("existing_code_owned forbids screening binding")
        if normalized["trust_domain"] != "unknown":
            raise ValueError("not_applicable screening has no trust domain")
    else:
        if normalized["status"] == "not_applicable":
            raise ValueError("screened_candidate cannot use not_applicable")
        if normalized["screening_stream_key"] is None or normalized[
            "source_revision"
        ] is None:
            raise ValueError("screened_candidate requires exact stream and revision binding")
        if normalized["status"] == "current_pass" and normalized[
            "trust_domain"
        ] != "yolozu_managed":
            raise ValueError("current_pass requires yolozu_managed trust")

    checked_bundle: AlgorithmBundleSpec | None = None
    if bundle is not None:
        checked_bundle = _as_bundle(bundle)
        bundle_record = checked_bundle.to_dict()
        if normalized["provenance_class"] != bundle_record["provenance_class"]:
            raise ValueError("provenance_class: bundle mismatch")
        binding = bundle_record.get("screening_binding")
        if binding is not None:
            if normalized["screening_stream_key"] != binding["stream_key"]:
                raise ValueError("screening_stream_key: bundle binding mismatch")
            if normalized["source_revision"] != binding["source_revision"]:
                raise ValueError("source_revision: bundle binding mismatch")
            if normalized["status"] == "current_pass" and (
                normalized["current_record_id"] != binding["pass_record_id"]
                or normalized["current_record_digest"] != binding["pass_record_digest"]
            ):
                raise ValueError("current_pass: onboarding pass identity mismatch")

    expected = canonical_sha256_v1(normalized, own_digest_field="observation_digest")
    if normalized["observation_digest"] != expected:
        raise ValueError("observation_digest: digest mismatch")
    return ScreeningEligibilityObservation(normalized)


@dataclass(frozen=True)
class SupportProfileEligibilityObservation:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def observation_digest(self) -> str:
        return str(self._record["observation_digest"])


def validate_support_profile_eligibility_observation(
    value: Mapping[str, Any],
    *,
    source_trust_domain: str | None = None,
    expected_family_id: str | None = None,
    expected_spec_digest: str | None = None,
    expected_channel: str | None = None,
    expected_environment_fingerprint: str | None = None,
    expected_workload_fingerprint: str | None = None,
    expected_protocol_fingerprint: str | None = None,
    expected_advertised_gates_digest: str | None = None,
    evidence_trust_domain: str | None = None,
    support_scope: str | None = None,
) -> SupportProfileEligibilityObservation:
    """Validate one support-profile projection input without opening its stream."""

    record = _mapping(value, field="SupportProfileEligibilityObservation")
    keys = frozenset(
        {
            "schema_version",
            "provider_id",
            "provider_version",
            "family_id",
            "bundle_spec_digest",
            "channel",
            "lifecycle_assignment_id",
            "lifecycle_assignment_digest",
            "support_profile_index_head_digest",
            "profile_set_record_id",
            "profile_set_record_digest",
            "profile_set_digest",
            "status",
            "profile_id",
            "profile_digest",
            "environment_fingerprint",
            "qualification_workload_fingerprint",
            "protocol_fingerprint",
            "advertised_gates_digest",
            "trust_domain",
            "observation_digest",
        }
    )
    _keys(
        record,
        field="SupportProfileEligibilityObservation",
        allowed=keys,
        required=keys,
    )
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("SupportProfileEligibilityObservation.schema_version: expected 1")
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "provider_id": _token(record["provider_id"], field="provider_id"),
        "provider_version": _token(record["provider_version"], field="provider_version"),
        "family_id": _token(
            record["family_id"], field="family_id", pattern=_COMPONENT_RE
        ),
        "bundle_spec_digest": _sha256(
            record["bundle_spec_digest"], field="bundle_spec_digest"
        ),
        "channel": _enum(record["channel"], field="channel", allowed=CHANNELS),
        "lifecycle_assignment_id": _token(
            record["lifecycle_assignment_id"], field="lifecycle_assignment_id"
        ),
        "lifecycle_assignment_digest": _sha256(
            record["lifecycle_assignment_digest"], field="lifecycle_assignment_digest"
        ),
        "support_profile_index_head_digest": _sha256(
            record["support_profile_index_head_digest"],
            field="support_profile_index_head_digest",
        ),
        "profile_set_record_id": _token(
            record["profile_set_record_id"], field="profile_set_record_id"
        ),
        "profile_set_record_digest": _sha256(
            record["profile_set_record_digest"], field="profile_set_record_digest"
        ),
        "profile_set_digest": _sha256(
            record["profile_set_digest"], field="profile_set_digest"
        ),
        "status": _enum(record["status"], field="status", allowed=SUPPORT_STATUSES),
        "profile_id": _nullable_token(record["profile_id"], field="profile_id"),
        "profile_digest": _nullable_sha256(
            record["profile_digest"], field="profile_digest"
        ),
        "environment_fingerprint": _nullable_sha256(
            record["environment_fingerprint"], field="environment_fingerprint"
        ),
        "qualification_workload_fingerprint": _nullable_sha256(
            record["qualification_workload_fingerprint"],
            field="qualification_workload_fingerprint",
        ),
        "protocol_fingerprint": _nullable_sha256(
            record["protocol_fingerprint"], field="protocol_fingerprint"
        ),
        "advertised_gates_digest": _nullable_sha256(
            record["advertised_gates_digest"], field="advertised_gates_digest"
        ),
        "trust_domain": _enum(
            record["trust_domain"], field="trust_domain", allowed=TRUST_DOMAINS
        ),
        "observation_digest": _sha256(
            record["observation_digest"], field="observation_digest"
        ),
    }
    if source_trust_domain is not None:
        derived = _enum(
            source_trust_domain,
            field="source_trust_domain",
            allowed=TRUST_DOMAINS,
        )
        if normalized["trust_domain"] != derived:
            raise ValueError("trust_domain: does not match loader-derived trust")

    matching_fields = (
        "profile_id",
        "profile_digest",
        "environment_fingerprint",
        "qualification_workload_fingerprint",
        "protocol_fingerprint",
        "advertised_gates_digest",
    )
    if normalized["status"] == "matching_one":
        if any(normalized[field] is None for field in matching_fields):
            raise ValueError("matching_one requires one complete matching profile identity")
        if normalized["trust_domain"] != "yolozu_managed":
            raise ValueError("matching_one requires yolozu_managed trust")
    elif any(normalized[field] is not None for field in matching_fields):
        raise ValueError("nonmatching support status forbids profile match fields")

    expected_fields = {
        "family_id": expected_family_id,
        "bundle_spec_digest": expected_spec_digest,
        "channel": expected_channel,
        "environment_fingerprint": expected_environment_fingerprint,
        "qualification_workload_fingerprint": expected_workload_fingerprint,
        "protocol_fingerprint": expected_protocol_fingerprint,
        "advertised_gates_digest": expected_advertised_gates_digest,
    }
    for field, expected_value in expected_fields.items():
        if expected_value is not None and normalized[field] != expected_value:
            raise ValueError(f"{field}: expected selection binding mismatch")

    if normalized["status"] == "not_required_site":
        if evidence_trust_domain != "site_managed" or support_scope != "site_qualified":
            raise ValueError(
                "not_required_site requires site_managed evidence and site_qualified scope"
            )
    expected = canonical_sha256_v1(normalized, own_digest_field="observation_digest")
    if normalized["observation_digest"] != expected:
        raise ValueError("observation_digest: digest mismatch")
    return SupportProfileEligibilityObservation(normalized)


def _fixed_class_mapping(
    value: Any,
    *,
    bundle: AlgorithmBundleSpec | None,
) -> dict[str, Any]:
    record = _mapping(value, field="class_mapping")
    keys = frozenset(
        {
            "vocabulary_id",
            "vocabulary_digest",
            "bundle_class_count",
            "requested_labels",
            "request_to_bundle_class_index",
            "retained_bundle_to_request",
        }
    )
    _keys(record, field="class_mapping", allowed=keys, required=keys)
    labels_raw = _list(
        record["requested_labels"], field="requested_labels", minimum=1, maximum=128
    )
    labels: list[str] = []
    total_bytes = 0
    for index, item in enumerate(labels_raw):
        if not isinstance(item, str):
            raise ValueError(f"requested_labels[{index}]: expected string")
        label = unicodedata.normalize("NFKC", item).strip()
        if label != item or not 1 <= len(label) <= 256:
            raise ValueError("requested_labels: values must be normalized")
        if any(unicodedata.category(character).startswith("C") for character in label):
            raise ValueError("requested_labels: control characters are invalid")
        total_bytes += len(label.encode("utf-8"))
        labels.append(label)
    if total_bytes > 4096 or len(labels) != len(set(labels)):
        raise ValueError("requested_labels: duplicate or oversized payload")
    bundle_count = _integer(
        record["bundle_class_count"],
        field="bundle_class_count",
        minimum=1,
        maximum=10_000,
    )
    request_to_bundle = [
        _integer(item, field="request_to_bundle_class_index[]", minimum=0, maximum=bundle_count - 1)
        for item in _list(
            record["request_to_bundle_class_index"],
            field="request_to_bundle_class_index",
            minimum=len(labels),
            maximum=len(labels),
        )
    ]
    if len(request_to_bundle) != len(set(request_to_bundle)):
        raise ValueError("request_to_bundle_class_index: duplicate bundle index")
    retained: list[dict[str, int]] = []
    for index, item in enumerate(
        _list(
            record["retained_bundle_to_request"],
            field="retained_bundle_to_request",
            minimum=len(labels),
            maximum=len(labels),
        )
    ):
        entry = _mapping(item, field=f"retained_bundle_to_request[{index}]")
        entry_keys = frozenset({"bundle_class_index", "request_index"})
        _keys(entry, field="retained mapping", allowed=entry_keys, required=entry_keys)
        retained.append(
            {
                "bundle_class_index": _integer(
                    entry["bundle_class_index"],
                    field="bundle_class_index",
                    minimum=0,
                    maximum=bundle_count - 1,
                ),
                "request_index": _integer(
                    entry["request_index"],
                    field="request_index",
                    minimum=0,
                    maximum=len(labels) - 1,
                ),
            }
        )
    expected_retained = [
        {"bundle_class_index": bundle_index, "request_index": request_index}
        for request_index, bundle_index in enumerate(request_to_bundle)
    ]
    expected_retained.sort(key=lambda item: item["bundle_class_index"])
    if retained != expected_retained:
        raise ValueError("retained_bundle_to_request: mapping is inconsistent")
    normalized = {
        "vocabulary_id": _token(record["vocabulary_id"], field="vocabulary_id"),
        "vocabulary_digest": _sha256(
            record["vocabulary_digest"], field="vocabulary_digest"
        ),
        "bundle_class_count": bundle_count,
        "requested_labels": labels,
        "request_to_bundle_class_index": request_to_bundle,
        "retained_bundle_to_request": retained,
    }
    if bundle is not None and normalized != build_fixed_class_mapping(bundle, labels):
        raise ValueError("class_mapping: immutable bundle vocabulary mismatch")
    return normalized


def _evidence_identity(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="evidence")
    keys = frozenset(
        {
            "activation_record_id",
            "activation_record_digest",
            "report_id",
            "report_digest",
            "trust_domain",
        }
    )
    _keys(record, field="evidence", allowed=keys, required=keys)
    return {
        "activation_record_id": _token(
            record["activation_record_id"], field="activation_record_id"
        ),
        "activation_record_digest": _sha256(
            record["activation_record_digest"], field="activation_record_digest"
        ),
        "report_id": _token(record["report_id"], field="report_id"),
        "report_digest": _sha256(record["report_digest"], field="report_digest"),
        "trust_domain": _enum(
            record["trust_domain"], field="evidence.trust_domain", allowed=TRUST_DOMAINS
        ),
    }


def _reason_codes(value: Any, *, field: str) -> list[str]:
    entries = _list(value, field=field, minimum=0, maximum=MAX_REASONS)
    out = [_enum(item, field=f"{field}[]", allowed=CANDIDATE_REASON_CODES) for item in entries]
    if len(out) != len(set(out)) or out != sorted(out, key=lambda item: item.encode("ascii")):
        raise ValueError(f"{field}: values must be unique and ASCII-byte sorted")
    return out


def _reason_details(value: Any, *, reasons: list[str]) -> list[dict[str, str]]:
    entries = _list(value, field="reason_details", minimum=0, maximum=len(reasons))
    out: list[dict[str, str]] = []
    for index, item in enumerate(entries):
        record = _mapping(item, field=f"reason_details[{index}]")
        keys = frozenset({"reason_code", "detail"})
        _keys(record, field="reason detail", allowed=keys, required=keys)
        reason = _enum(
            record["reason_code"], field="reason_code", allowed=CANDIDATE_REASON_CODES
        )
        detail = _token(record["detail"], field="detail", pattern=_DETAIL_RE)
        out.append({"reason_code": reason, "detail": detail})
    codes = [item["reason_code"] for item in out]
    if len(codes) != len(set(codes)) or any(code not in reasons for code in codes):
        raise ValueError("reason_details: duplicate or unlisted reason")
    if codes != sorted(codes, key=lambda item: item.encode("ascii")):
        raise ValueError("reason_details: must be ASCII-byte sorted")
    return out


def _ranking_trace(value: Any, *, reasons: list[str]) -> list[dict[str, Any]]:
    entries = _list(value, field="ranking_trace", minimum=0, maximum=MAX_TRACE_STEPS)
    out: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        record = _mapping(item, field=f"ranking_trace[{index}]")
        keys = frozenset({"step", "status", "reason_code", "detail"})
        _keys(record, field="ranking trace step", allowed=keys, required=keys)
        status = _enum(
            record["status"],
            field="ranking_trace.status",
            allowed=frozenset({"pass", "failed", "not_applicable"}),
        )
        reason = record["reason_code"]
        detail = record["detail"]
        if status == "failed":
            reason = _enum(reason, field="ranking_trace.reason_code", allowed=CANDIDATE_REASON_CODES)
            if reason not in reasons:
                raise ValueError("ranking_trace: failed reason is not in candidate reasons")
        elif reason is not None:
            raise ValueError("ranking_trace: nonfailed step forbids reason_code")
        if detail is not None:
            detail = _token(detail, field="ranking_trace.detail", pattern=_DETAIL_RE)
        out.append(
            {
                "step": _integer(record["step"], field="ranking_trace.step", minimum=1, maximum=32),
                "status": status,
                "reason_code": reason,
                "detail": detail,
            }
        )
    steps = [item["step"] for item in out]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError("ranking_trace: steps must be unique and ordered")
    return out


def _bundle_identity(value: Any) -> dict[str, Any]:
    record = _mapping(value, field="selected_bundle")
    keys = frozenset(
        {
            "family_id",
            "bundle_id",
            "bundle_version",
            "spec_digest",
            "artifact_set_digest",
            "effective_channel",
        }
    )
    _keys(record, field="selected_bundle", allowed=keys, required=keys)
    return {
        "family_id": _token(record["family_id"], field="family_id", pattern=_COMPONENT_RE),
        "bundle_id": _token(record["bundle_id"], field="bundle_id", pattern=_COMPONENT_RE),
        "bundle_version": _token(
            record["bundle_version"], field="bundle_version", pattern=_VERSION_RE
        ),
        "spec_digest": _sha256(record["spec_digest"], field="spec_digest"),
        "artifact_set_digest": _sha256(
            record["artifact_set_digest"], field="artifact_set_digest"
        ),
        "effective_channel": _enum(
            record["effective_channel"], field="effective_channel", allowed=CHANNELS
        ),
    }


def _candidate_identity(record: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, bytes]:
    return (
        str(record["family_id"]).encode("utf-8"),
        str(record["bundle_id"]).encode("utf-8"),
        str(record["bundle_version"]).encode("utf-8"),
        str(record["spec_digest"]).encode("ascii"),
    )


def _candidate_evaluation(
    value: Any,
    *,
    prompt_mode: str,
    environment_fingerprint: str,
    workload_fingerprint: str,
    protocol_fingerprint: str,
    advertised_gates_digest: str,
    bundle: AlgorithmBundleSpec | None,
) -> dict[str, Any]:
    record = _mapping(value, field="CandidateEvaluation")
    keys = frozenset(
        {
            "family_id",
            "bundle_id",
            "bundle_version",
            "spec_digest",
            "artifact_set_digest",
            "effective_channel",
            "pointed_channels",
            "matching_channels",
            "screening_observation",
            "support_profile_observation",
            "artifact_state_fingerprint",
            "class_mapping",
            "evidence",
            "support_scope",
            "rank_state",
            "rank_position",
            "reason_codes",
            "reason_details",
            "human_summary",
            "ranking_trace",
        }
    )
    _keys(record, field="CandidateEvaluation", allowed=keys, required=keys)
    reasons = _reason_codes(record["reason_codes"], field="reason_codes")
    evidence = None if record["evidence"] is None else _evidence_identity(record["evidence"])
    support_scope = _enum(
        record["support_scope"], field="support_scope", allowed=SUPPORT_SCOPES
    )
    rank_state = _enum(
        record["rank_state"],
        field="rank_state",
        allowed=frozenset({"excluded", "eligible_not_selected", "selected"}),
    )
    rank_position = record["rank_position"]
    if rank_position is not None:
        rank_position = _integer(rank_position, field="rank_position", minimum=1, maximum=128)

    effective_channel = _enum(
        record["effective_channel"], field="effective_channel", allowed=CHANNELS
    )
    pointed = [
        _enum(item, field="pointed_channels[]", allowed=CHANNELS)
        for item in _list(record["pointed_channels"], field="pointed_channels", minimum=0, maximum=3)
    ]
    matching = [
        _enum(item, field="matching_channels[]", allowed=CHANNELS)
        for item in _list(record["matching_channels"], field="matching_channels", minimum=0, maximum=3)
    ]
    canonical_channels = [item for item in ("Candidate", "Experimental", "Stable") if item in pointed]
    canonical_matching = [item for item in ("Candidate", "Experimental", "Stable") if item in matching]
    if pointed != canonical_channels or matching != canonical_matching:
        raise ValueError("candidate channels must be unique and canonically ordered")
    if pointed and effective_channel not in pointed:
        raise ValueError("effective_channel must be one of pointed_channels")
    if any(channel not in pointed for channel in matching):
        raise ValueError("matching_channels must be a subset of pointed_channels")
    if rank_state != "excluded" and effective_channel not in matching:
        raise ValueError("eligible candidate requires a matching effective channel")

    family_id = _token(record["family_id"], field="family_id", pattern=_COMPONENT_RE)
    bundle_id = _token(record["bundle_id"], field="bundle_id", pattern=_COMPONENT_RE)
    bundle_version = _token(
        record["bundle_version"], field="bundle_version", pattern=_VERSION_RE
    )
    spec_digest = _sha256(record["spec_digest"], field="spec_digest")
    artifact_set_digest = _sha256(
        record["artifact_set_digest"], field="artifact_set_digest"
    )
    if bundle is not None:
        bundle_record = bundle.to_dict()
        expected_identity = (
            bundle_record["family_id"],
            bundle_record["bundle_id"],
            bundle_record["bundle_version"],
            bundle.spec_digest,
            bundle.artifact_set_digest,
        )
        if (family_id, bundle_id, bundle_version, spec_digest, artifact_set_digest) != expected_identity:
            raise ValueError("CandidateEvaluation: registry identity mismatch")

    screening = validate_screening_eligibility_observation(
        _mapping(record["screening_observation"], field="screening_observation"),
        bundle=bundle,
    ).to_dict()
    raw_support = record["support_profile_observation"]
    support = None
    if raw_support is not None:
        support_record = _mapping(raw_support, field="support_profile_observation")
        support = validate_support_profile_eligibility_observation(
            support_record,
            expected_family_id=family_id,
            expected_spec_digest=spec_digest,
            expected_channel=effective_channel,
            expected_environment_fingerprint=(
                environment_fingerprint
                if support_record.get("status") == "matching_one"
                else None
            ),
            expected_workload_fingerprint=(
                workload_fingerprint
                if support_record.get("status") == "matching_one"
                else None
            ),
            expected_protocol_fingerprint=(
                protocol_fingerprint
                if support_record.get("status") == "matching_one"
                else None
            ),
            expected_advertised_gates_digest=(
                advertised_gates_digest
                if support_record.get("status") == "matching_one"
                else None
            ),
            evidence_trust_domain=None if evidence is None else evidence["trust_domain"],
            support_scope=support_scope,
        ).to_dict()

    artifact_state = _nullable_sha256(
        record["artifact_state_fingerprint"], field="artifact_state_fingerprint"
    )
    class_mapping = (
        None
        if record["class_mapping"] is None
        else _fixed_class_mapping(record["class_mapping"], bundle=bundle)
    )
    if prompt_mode == "text" and class_mapping is not None:
        raise ValueError("text prompt mode forbids fixed-class mapping")
    if rank_state != "excluded" and prompt_mode == "fixed_classes" and class_mapping is None:
        raise ValueError("eligible fixed-class candidate requires exact class mapping")

    if rank_state == "excluded":
        if not reasons or rank_position is not None:
            raise ValueError("excluded candidate requires reasons and null rank_position")
    else:
        if reasons or record["reason_details"]:
            raise ValueError("eligible candidate cannot contain hard-failure reasons")
        if evidence is None or evidence["trust_domain"] not in {
            "yolozu_managed",
            "site_managed",
        }:
            raise ValueError("eligible candidate requires one trusted evidence identity")
        if artifact_state is None or support_scope == "none":
            raise ValueError("eligible candidate requires artifact state and support scope")
        if support is None:
            raise ValueError("eligible candidate requires one support observation")
        if effective_channel not in {"Experimental", "Stable"}:
            raise ValueError("Candidate channel is not selectable")
        if rank_state == "selected" and rank_position != 1:
            raise ValueError("selected candidate requires rank_position 1")
        if rank_state == "eligible_not_selected" and (
            rank_position is None or rank_position < 2
        ):
            raise ValueError("eligible_not_selected requires rank_position >= 2")
        if evidence["trust_domain"] == "yolozu_managed":
            if support_scope != "public_qualified" or support["status"] != "matching_one":
                raise ValueError("public selection requires one matching managed support profile")
        if evidence["trust_domain"] == "site_managed":
            if support_scope != "site_qualified" or support["status"] not in {
                "matching_one",
                "not_required_site",
            }:
                raise ValueError("site selection requires reviewed site support scope")

    human_summary = _nullable_text(
        record["human_summary"], field="human_summary", maximum_bytes=1024
    )
    if human_summary != _CANDIDATE_SUMMARIES[rank_state]:
        raise ValueError("human_summary: expected code-owned fixed template")
    return {
        "family_id": family_id,
        "bundle_id": bundle_id,
        "bundle_version": bundle_version,
        "spec_digest": spec_digest,
        "artifact_set_digest": artifact_set_digest,
        "effective_channel": effective_channel,
        "pointed_channels": pointed,
        "matching_channels": matching,
        "screening_observation": screening,
        "support_profile_observation": support,
        "artifact_state_fingerprint": artifact_state,
        "class_mapping": class_mapping,
        "evidence": evidence,
        "support_scope": support_scope,
        "rank_state": rank_state,
        "rank_position": rank_position,
        "reason_codes": reasons,
        "reason_details": _reason_details(record["reason_details"], reasons=reasons),
        "human_summary": human_summary,
        "ranking_trace": _ranking_trace(record["ranking_trace"], reasons=reasons),
    }


@dataclass(frozen=True)
class SelectionDecision:
    _record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def decision_digest(self) -> str:
        return str(self._record["decision_digest"])


def validate_selection_decision(
    value: Mapping[str, Any],
    *,
    expected_registry: AlgorithmBundleRegistry | Mapping[str, Any] | None = None,
    as_of: str | datetime | None = None,
) -> SelectionDecision:
    """Validate a complete selected or abstained SelectionDecision v1."""

    record = _mapping(value, field="SelectionDecision")
    keys = frozenset(
        {
            "schema_version",
            "decision_id",
            "status",
            "decided_at",
            "local_job_digest",
            "local_input_digest",
            "artifact_resolver_state_digest",
            "environment_fingerprint",
            "qualification_workload_fingerprint",
            "protocol_fingerprint",
            "advertised_gates_digest",
            "registry_id",
            "registry_digest",
            "registry_trust_domain",
            "lifecycle_projection_digest",
            "lifecycle_trust_domain",
            "ranking_policy",
            "prompt_mode",
            "registry_bundle_count",
            "selected_bundle",
            "selected_evidence",
            "selected_artifact_state_fingerprint",
            "selected_class_mapping",
            "support_scope",
            "reason_codes",
            "human_summary",
            "candidate_evaluations",
            "selection_trace",
            "decision_digest",
        }
    )
    _keys(record, field="SelectionDecision", allowed=keys, required=keys)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("SelectionDecision.schema_version: expected 1")
    status = _enum(
        record["status"], field="status", allowed=frozenset({"selected", "abstained"})
    )
    decided_at, decided_instant = _utc(record["decided_at"], field="decided_at")
    if decided_instant > _as_of(as_of):
        raise ValueError("decided_at: future decision is invalid")
    prompt_mode = _enum(
        record["prompt_mode"],
        field="prompt_mode",
        allowed=frozenset({"fixed_classes", "text"}),
    )
    environment_fingerprint = _sha256(
        record["environment_fingerprint"], field="environment_fingerprint"
    )
    workload_fingerprint = _sha256(
        record["qualification_workload_fingerprint"],
        field="qualification_workload_fingerprint",
    )
    protocol_fingerprint = _sha256(
        record["protocol_fingerprint"], field="protocol_fingerprint"
    )
    advertised_gates_digest = _sha256(
        record["advertised_gates_digest"], field="advertised_gates_digest"
    )

    registry: AlgorithmBundleRegistry | None = None
    if expected_registry is not None:
        registry = (
            expected_registry
            if isinstance(expected_registry, AlgorithmBundleRegistry)
            else validate_algorithm_bundle_registry(expected_registry)
        )
    bundle_by_digest = {} if registry is None else registry.by_spec_digest()
    evaluations_raw = _list(
        record["candidate_evaluations"],
        field="candidate_evaluations",
        minimum=0,
        maximum=MAX_CANDIDATES,
    )
    evaluations: list[dict[str, Any]] = []
    for item in evaluations_raw:
        raw = _mapping(item, field="CandidateEvaluation")
        spec_digest = _sha256(raw.get("spec_digest"), field="spec_digest")
        evaluations.append(
            _candidate_evaluation(
                raw,
                prompt_mode=prompt_mode,
                environment_fingerprint=environment_fingerprint,
                workload_fingerprint=workload_fingerprint,
                protocol_fingerprint=protocol_fingerprint,
                advertised_gates_digest=advertised_gates_digest,
                bundle=bundle_by_digest.get(spec_digest),
            )
        )
    identities = [_candidate_identity(item) for item in evaluations]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("candidate_evaluations: identities must be unique and canonically ordered")
    registry_count = _integer(
        record["registry_bundle_count"],
        field="registry_bundle_count",
        minimum=0,
        maximum=MAX_CANDIDATES,
    )
    if registry_count != len(evaluations):
        raise ValueError("registry_bundle_count: candidate evaluation omission")
    if registry is not None:
        if record["registry_digest"] != registry.registry_digest:
            raise ValueError("registry_digest: expected registry mismatch")
        if registry_count != len(registry.bundles):
            raise ValueError("candidate_evaluations: incomplete expected registry")
        if {item["spec_digest"] for item in evaluations} != set(bundle_by_digest):
            raise ValueError("candidate_evaluations: expected registry identity mismatch")

    eligible = [item for item in evaluations if item["rank_state"] != "excluded"]
    eligible_by_rank = sorted(eligible, key=lambda item: int(item["rank_position"]))
    positions = [item["rank_position"] for item in eligible_by_rank]
    if positions != list(range(1, len(positions) + 1)):
        raise ValueError("rank_position: eligible ranks must be contiguous")
    selected_candidates = [item for item in evaluations if item["rank_state"] == "selected"]

    selected_bundle = (
        None if record["selected_bundle"] is None else _bundle_identity(record["selected_bundle"])
    )
    selected_evidence = (
        None if record["selected_evidence"] is None else _evidence_identity(record["selected_evidence"])
    )
    selected_artifact_state = _nullable_sha256(
        record["selected_artifact_state_fingerprint"],
        field="selected_artifact_state_fingerprint",
    )
    selected_class_mapping = (
        None
        if record["selected_class_mapping"] is None
        else _fixed_class_mapping(
            record["selected_class_mapping"],
            bundle=(
                None
                if selected_bundle is None
                else bundle_by_digest.get(selected_bundle["spec_digest"])
            ),
        )
    )
    support_scope = _enum(
        record["support_scope"], field="support_scope", allowed=SUPPORT_SCOPES
    )
    top_reasons_raw = _list(record["reason_codes"], field="reason_codes", minimum=0, maximum=1)
    if any(item != "no_eligible_candidate" for item in top_reasons_raw):
        raise ValueError("reason_codes: unsupported top-level outcome code")
    if top_reasons_raw != sorted(set(top_reasons_raw), key=lambda item: item.encode("ascii")):
        raise ValueError("reason_codes: duplicate or unordered outcome code")

    if status == "selected":
        if len(selected_candidates) != 1 or top_reasons_raw:
            raise ValueError("selected decision requires exactly one selected candidate")
        selected = selected_candidates[0]
        expected_bundle = {
            key: selected[key]
            for key in (
                "family_id",
                "bundle_id",
                "bundle_version",
                "spec_digest",
                "artifact_set_digest",
                "effective_channel",
            )
        }
        if selected_bundle != expected_bundle:
            raise ValueError("selected_bundle: does not match selected candidate")
        if selected_evidence != selected["evidence"]:
            raise ValueError("selected_evidence: does not match selected candidate")
        if selected_artifact_state != selected["artifact_state_fingerprint"]:
            raise ValueError("selected artifact state: candidate mismatch")
        if selected_class_mapping != selected["class_mapping"]:
            raise ValueError("selected_class_mapping: candidate mismatch")
        if support_scope != selected["support_scope"]:
            raise ValueError("support_scope: selected candidate mismatch")
        if record["registry_trust_domain"] != "yolozu_managed" or record[
            "lifecycle_trust_domain"
        ] != "yolozu_managed":
            raise ValueError("selected decision requires managed registry and lifecycle trust")
    else:
        if eligible or selected_candidates:
            raise ValueError("abstained decision cannot hide an eligible candidate")
        if any(
            item is not None
            for item in (
                selected_bundle,
                selected_evidence,
                selected_artifact_state,
                selected_class_mapping,
            )
        ):
            raise ValueError("abstained decision requires null selected identities")
        if support_scope != "none" or top_reasons_raw != ["no_eligible_candidate"]:
            raise ValueError("abstained decision requires none scope and no-eligible reason")

    trace_raw = _list(
        record["selection_trace"],
        field="selection_trace",
        minimum=0,
        maximum=MAX_CANDIDATES,
    )
    trace: list[dict[str, Any]] = []
    for index, item in enumerate(trace_raw):
        entry = _mapping(item, field=f"selection_trace[{index}]")
        trace_keys = frozenset({"rank_position", "spec_digest"})
        _keys(entry, field="selection trace entry", allowed=trace_keys, required=trace_keys)
        trace.append(
            {
                "rank_position": _integer(
                    entry["rank_position"], field="selection_trace.rank_position", minimum=1, maximum=128
                ),
                "spec_digest": _sha256(entry["spec_digest"], field="selection_trace.spec_digest"),
            }
        )
    expected_trace = [
        {"rank_position": item["rank_position"], "spec_digest": item["spec_digest"]}
        for item in eligible_by_rank
    ]
    if trace != expected_trace:
        raise ValueError("selection_trace: must contain the complete deterministic rank order")

    normalized = {
        "schema_version": 1,
        "decision_id": _token(
            record["decision_id"], field="decision_id", pattern=_COMPONENT_RE
        ),
        "status": status,
        "decided_at": decided_at,
        "local_job_digest": _sha256(record["local_job_digest"], field="local_job_digest"),
        "local_input_digest": _sha256(
            record["local_input_digest"], field="local_input_digest"
        ),
        "artifact_resolver_state_digest": _sha256(
            record["artifact_resolver_state_digest"],
            field="artifact_resolver_state_digest",
        ),
        "environment_fingerprint": environment_fingerprint,
        "qualification_workload_fingerprint": workload_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "advertised_gates_digest": advertised_gates_digest,
        "registry_id": _token(record["registry_id"], field="registry_id"),
        "registry_digest": _sha256(record["registry_digest"], field="registry_digest"),
        "registry_trust_domain": _enum(
            record["registry_trust_domain"],
            field="registry_trust_domain",
            allowed=TRUST_DOMAINS,
        ),
        "lifecycle_projection_digest": _sha256(
            record["lifecycle_projection_digest"],
            field="lifecycle_projection_digest",
        ),
        "lifecycle_trust_domain": _enum(
            record["lifecycle_trust_domain"],
            field="lifecycle_trust_domain",
            allowed=TRUST_DOMAINS,
        ),
        "ranking_policy": _enum(
            record["ranking_policy"], field="ranking_policy", allowed=RANKING_POLICIES
        ),
        "prompt_mode": prompt_mode,
        "registry_bundle_count": registry_count,
        "selected_bundle": selected_bundle,
        "selected_evidence": selected_evidence,
        "selected_artifact_state_fingerprint": selected_artifact_state,
        "selected_class_mapping": selected_class_mapping,
        "support_scope": support_scope,
        "reason_codes": top_reasons_raw,
        "human_summary": _nullable_text(
            record["human_summary"], field="human_summary", maximum_bytes=1024
        ),
        "candidate_evaluations": evaluations,
        "selection_trace": trace,
        "decision_digest": _sha256(record["decision_digest"], field="decision_digest"),
    }
    if normalized["human_summary"] != _DECISION_SUMMARIES[status]:
        raise ValueError("human_summary: expected code-owned fixed template")
    if len(canonical_json_v1(normalized)) > MAX_SELECTION_DECISION_BYTES:
        raise ValueError("registry/evidence_limit_exceeded")
    expected_digest = canonical_sha256_v1(normalized, own_digest_field="decision_digest")
    if normalized["decision_digest"] != expected_digest:
        raise ValueError("decision_digest: digest mismatch")
    return SelectionDecision(normalized)
