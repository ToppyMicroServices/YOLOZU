"""Fail-closed candidate-screening records and projection provider.

The canonical stream is reviewed metadata only. Loading and projecting it never
fetches, imports, downloads, registers, or executes candidate code or weights.
Trust comes from the loader-selected path, never from a JSON claim.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .bundles import AlgorithmBundleSpec, validate_algorithm_bundle_spec
from .canonical import canonical_json_v1, canonical_sha256_v1
from .control_records import MAX_CONTROL_RECORD_BYTES, load_bounded_jsonl_bytes
from .safe_https import HttpsLocation, SafeHttpsError
from .selection import (
    ScreeningEligibilityObservation,
    validate_screening_eligibility_observation,
)

__all__ = [
    "CandidateScreeningProjection",
    "CandidateScreeningRecord",
    "MAX_SCREENING_RECORDS",
    "MAX_SCREENING_STREAM_BYTES",
    "build_screening_eligibility_observation",
    "compute_candidate_screening_stream_key",
    "load_candidate_screening_jsonl_bytes",
    "project_candidate_screening_records",
    "validate_candidate_screening_record",
]


MAX_SCREENING_RECORDS = 8192
MAX_SCREENING_STREAM_BYTES = 64 * 1024 * 1024
ZERO_DIGEST = "0" * 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_ROLE_RE = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
_SPDX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)

_RESULTS = frozenset({"pass", "unknown", "fail"})
_OVERALL_STATUSES = frozenset({"pass", "hold", "reject"})
_TRUST_DOMAINS = frozenset({"yolozu_managed", "operator_asserted"})
_TASKS = frozenset({"object_detection", "instance_segmentation"})
_PROMPT_MODES = frozenset({"fixed_classes", "text"})

_CHECK_ORDER = (
    "source_provenance",
    "source_integrity",
    "code_license",
    "weight_license",
    "dataset_evaluation_license",
    "weight_source_integrity",
    "local_availability",
    "task_prompt_output_fit",
    "predictions_interface_mapping",
    "runtime_provider",
    "compute_memory",
    "maintenance",
    "security_supply_chain",
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


def _enum(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field}: unsupported value")
    return value


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum}")
    return value


def _nullable_integer(
    value: Any, *, field: str, minimum: int, maximum: int
) -> int | None:
    if value is None:
        return None
    return _integer(value, field=field, minimum=minimum, maximum=maximum)


def _token(value: Any, *, field: str, pattern: re.Pattern[str] = _ID_RE) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field}: invalid identifier")
    return value


def _nullable_token(
    value: Any, *, field: str, pattern: re.Pattern[str] = _ID_RE
) -> str | None:
    if value is None:
        return None
    return _token(value, field=field, pattern=pattern)


def _sha256(value: Any, *, field: str, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")
    if not allow_zero and value == ZERO_DIGEST:
        raise ValueError(f"{field}: zero digest is reserved")
    return value


def _nullable_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


def _safe_text(value: Any, *, field: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected non-empty text")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{field}: exceeds {maximum_bytes} UTF-8 bytes")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{field}: control characters are invalid")
    return value


def _utc(value: Any, *, field: str) -> str:
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
    return value


def _string_list(
    value: Any,
    *,
    field: str,
    allowed: frozenset[str] | None = None,
    minimum: int = 0,
    maximum: int = 8,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{field}: expected {minimum}..{maximum} items")
    output = [
        _enum(item, field=f"{field}[]", allowed=allowed)
        if allowed is not None
        else _token(item, field=f"{field}[]")
        for item in value
    ]
    if len(output) != len(set(output)):
        raise ValueError(f"{field}: duplicate value")
    return output


def _review_reference(value: Any, *, field: str) -> dict[str, str]:
    record = _mapping(value, field=field)
    kind = record.get("kind")
    if kind == "public_repository_id":
        keys = frozenset({"kind", "value"})
        _keys(record, field=field, allowed=keys, required=keys)
        return {
            "kind": kind,
            "value": _token(record["value"], field=f"{field}.value"),
        }
    if kind == "site_local_status":
        keys = frozenset({"kind", "status"})
        _keys(record, field=field, allowed=keys, required=keys)
        return {
            "kind": kind,
            "status": _enum(
                record["status"],
                field=f"{field}.status",
                allowed=frozenset({"present", "not_available"}),
            ),
        }
    raise ValueError(f"{field}.kind: unsupported value")


def _reference_is_available(value: Mapping[str, str]) -> bool:
    return value["kind"] == "public_repository_id" or value.get("status") == "present"


def _base_assessment(value: Any, *, field: str) -> tuple[dict[str, Any], str]:
    record = _mapping(value, field=field)
    if "result" not in record or "reference" not in record:
        raise ValueError(f"{field}: result and reference are required")
    result = _enum(record["result"], field=f"{field}.result", allowed=_RESULTS)
    reference = _review_reference(record["reference"], field=f"{field}.reference")
    if result != "unknown" and not _reference_is_available(reference):
        raise ValueError(f"{field}: pass/fail requires an available reference")
    return {"result": result, "reference": reference}, result


def _source_provenance(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.source_provenance"
    record = _mapping(value, field=field)
    keys = frozenset({"result", "revision_kind", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    revision_kind = _enum(
        record["revision_kind"],
        field=f"{field}.revision_kind",
        allowed=frozenset({"immutable", "mutable", "unknown"}),
    )
    expected = {"immutable": "pass", "mutable": "fail", "unknown": "unknown"}
    if result != expected[revision_kind]:
        raise ValueError(f"{field}: result does not match revision_kind")
    output["revision_kind"] = revision_kind
    return output


def _integrity_assessment(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset({"result", "procedure_id", "expected_sha256", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    procedure = _nullable_token(record["procedure_id"], field=f"{field}.procedure_id")
    digest = _nullable_sha256(
        record["expected_sha256"], field=f"{field}.expected_sha256"
    )
    if result == "pass" and (procedure is None or digest is None):
        raise ValueError(f"{field}: pass requires procedure and digest")
    if result == "unknown" and (procedure is not None or digest is not None):
        raise ValueError(f"{field}: unknown forbids asserted procedure or digest")
    output.update({"procedure_id": procedure, "expected_sha256": digest})
    return output


def _license_assessment(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset({"result", "spdx_id", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    spdx_id = _nullable_token(record["spdx_id"], field=f"{field}.spdx_id", pattern=_SPDX_RE)
    if result == "pass" and spdx_id is None:
        raise ValueError(f"{field}: pass requires an SPDX identifier")
    if result == "unknown" and spdx_id is not None:
        raise ValueError(f"{field}: unknown forbids an asserted SPDX identifier")
    output["spdx_id"] = spdx_id
    return output


def _weight_source_integrity(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.weight_source_integrity"
    record = _mapping(value, field=field)
    keys = frozenset(
        {"result", "source_kind", "procedure_id", "expected_sha256", "reference"}
    )
    _keys(record, field=field, allowed=keys, required=keys)
    output = _integrity_assessment(
        {
            key: record[key]
            for key in ("result", "procedure_id", "expected_sha256", "reference")
        },
        field=field,
    )
    source_kind = _enum(
        record["source_kind"],
        field=f"{field}.source_kind",
        allowed=frozenset(
            {"official_release", "official_repository", "unverified", "unknown"}
        ),
    )
    expected = {
        "official_release": "pass",
        "official_repository": "pass",
        "unverified": "fail",
        "unknown": "unknown",
    }
    if output["result"] != expected[source_kind]:
        raise ValueError(f"{field}: result does not match source_kind")
    output["source_kind"] = source_kind
    return output


def _local_availability(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.local_availability"
    record = _mapping(value, field=field)
    keys = frozenset({"result", "mode", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    mode = _enum(
        record["mode"],
        field=f"{field}.mode",
        allowed=frozenset(
            {"local_artifacts", "downloadable_artifacts", "hosted_only", "unknown"}
        ),
    )
    expected = {
        "local_artifacts": "pass",
        "downloadable_artifacts": "pass",
        "hosted_only": "fail",
        "unknown": "unknown",
    }
    if result != expected[mode]:
        raise ValueError(f"{field}: result does not match mode")
    output["mode"] = mode
    return output


def _simple_assessment(value: Any, *, field: str) -> dict[str, Any]:
    record = _mapping(value, field=field)
    keys = frozenset({"result", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    return _base_assessment(record, field=field)[0]


def _interface_mapping(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.predictions_interface_mapping"
    record = _mapping(value, field=field)
    keys = frozenset(
        {"result", "interface_contract_id", "mapping_revision", "reference"}
    )
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    interface_id = _nullable_token(
        record["interface_contract_id"], field=f"{field}.interface_contract_id"
    )
    mapping_revision = _nullable_token(
        record["mapping_revision"], field=f"{field}.mapping_revision"
    )
    if result == "pass" and (
        interface_id != "predictions-v2" or mapping_revision is None
    ):
        raise ValueError(f"{field}: pass requires the predictions-v2 mapping")
    if result == "unknown" and (
        interface_id is not None or mapping_revision is not None
    ):
        raise ValueError(f"{field}: unknown forbids an asserted mapping")
    output.update(
        {
            "interface_contract_id": interface_id,
            "mapping_revision": mapping_revision,
        }
    )
    return output


def _runtime_provider(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.runtime_provider"
    record = _mapping(value, field=field)
    keys = frozenset(
        {"result", "runtime_ids", "provider_ids", "requires_new_surface", "reference"}
    )
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    runtimes = _string_list(record["runtime_ids"], field=f"{field}.runtime_ids")
    providers = _string_list(record["provider_ids"], field=f"{field}.provider_ids")
    requires_new = record["requires_new_surface"]
    if type(requires_new) is not bool:
        raise ValueError(f"{field}.requires_new_surface: expected boolean")
    if requires_new and result != "unknown":
        raise ValueError(f"{field}: a new runtime surface must remain unknown/hold")
    if result == "pass" and (not runtimes or not providers):
        raise ValueError(f"{field}: pass requires runtime and provider identities")
    output.update(
        {
            "runtime_ids": runtimes,
            "provider_ids": providers,
            "requires_new_surface": requires_new,
        }
    )
    return output


def _compute_memory(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.compute_memory"
    record = _mapping(value, field=field)
    keys = frozenset(
        {
            "result",
            "estimated_compute_operations",
            "estimated_peak_memory_bytes",
            "reference",
        }
    )
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    compute = _nullable_integer(
        record["estimated_compute_operations"],
        field=f"{field}.estimated_compute_operations",
        minimum=1,
        maximum=2**63 - 1,
    )
    memory = _nullable_integer(
        record["estimated_peak_memory_bytes"],
        field=f"{field}.estimated_peak_memory_bytes",
        minimum=1,
        maximum=2**63 - 1,
    )
    if result == "pass" and (compute is None or memory is None):
        raise ValueError(f"{field}: pass requires compute and memory estimates")
    if result == "unknown" and (compute is not None or memory is not None):
        raise ValueError(f"{field}: unknown forbids asserted estimates")
    output.update(
        {
            "estimated_compute_operations": compute,
            "estimated_peak_memory_bytes": memory,
        }
    )
    return output


def _maintenance(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.maintenance"
    record = _mapping(value, field=field)
    keys = frozenset({"result", "status", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    status = _enum(
        record["status"],
        field=f"{field}.status",
        allowed=frozenset({"maintained", "stale", "unknown"}),
    )
    expected = {"maintained": "pass", "stale": "fail", "unknown": "unknown"}
    if result != expected[status]:
        raise ValueError(f"{field}: result does not match status")
    output["status"] = status
    return output


def _security_supply_chain(value: Any) -> dict[str, Any]:
    field = "mechanical_checks.security_supply_chain"
    record = _mapping(value, field=field)
    keys = frozenset({"result", "status", "reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    output, result = _base_assessment(record, field=field)
    status = _enum(
        record["status"],
        field=f"{field}.status",
        allowed=frozenset({"no_known_concern", "concern_present", "unknown"}),
    )
    expected = {
        "no_known_concern": "pass",
        "concern_present": "fail",
        "unknown": "unknown",
    }
    if result != expected[status]:
        raise ValueError(f"{field}: result does not match status")
    output["status"] = status
    return output


def _mechanical_checks(value: Any) -> dict[str, dict[str, Any]]:
    field = "mechanical_checks"
    record = _mapping(value, field=field)
    keys = frozenset(_CHECK_ORDER)
    _keys(record, field=field, allowed=keys, required=keys)
    return {
        "source_provenance": _source_provenance(record["source_provenance"]),
        "source_integrity": _integrity_assessment(
            record["source_integrity"], field="mechanical_checks.source_integrity"
        ),
        "code_license": _license_assessment(
            record["code_license"], field="mechanical_checks.code_license"
        ),
        "weight_license": _license_assessment(
            record["weight_license"], field="mechanical_checks.weight_license"
        ),
        "dataset_evaluation_license": _license_assessment(
            record["dataset_evaluation_license"],
            field="mechanical_checks.dataset_evaluation_license",
        ),
        "weight_source_integrity": _weight_source_integrity(
            record["weight_source_integrity"]
        ),
        "local_availability": _local_availability(record["local_availability"]),
        "task_prompt_output_fit": _simple_assessment(
            record["task_prompt_output_fit"],
            field="mechanical_checks.task_prompt_output_fit",
        ),
        "predictions_interface_mapping": _interface_mapping(
            record["predictions_interface_mapping"]
        ),
        "runtime_provider": _runtime_provider(record["runtime_provider"]),
        "compute_memory": _compute_memory(record["compute_memory"]),
        "maintenance": _maintenance(record["maintenance"]),
        "security_supply_chain": _security_supply_chain(
            record["security_supply_chain"]
        ),
    }


def _requested_capability(value: Any) -> dict[str, Any]:
    field = "candidate.requested_capability"
    record = _mapping(value, field=field)
    keys = frozenset(
        {"task", "prompt_modes", "output_interface_contract_id"}
    )
    _keys(record, field=field, allowed=keys, required=keys)
    return {
        "task": _enum(record["task"], field=f"{field}.task", allowed=_TASKS),
        "prompt_modes": _string_list(
            record["prompt_modes"],
            field=f"{field}.prompt_modes",
            allowed=_PROMPT_MODES,
            minimum=1,
            maximum=2,
        ),
        "output_interface_contract_id": _enum(
            record["output_interface_contract_id"],
            field=f"{field}.output_interface_contract_id",
            allowed=frozenset({"predictions-v2"}),
        ),
    }


def _candidate(value: Any) -> dict[str, Any]:
    field = "candidate"
    record = _mapping(value, field=field)
    keys = frozenset(
        {"candidate_id", "source", "immutable_revision", "requested_capability"}
    )
    _keys(record, field=field, allowed=keys, required=keys)
    try:
        source = HttpsLocation.from_mapping(
            _mapping(record["source"], field="candidate.source")
        ).to_mapping()
    except SafeHttpsError as exc:
        raise ValueError("candidate.source: invalid canonical HTTPS location") from exc
    return {
        "candidate_id": _token(record["candidate_id"], field="candidate.candidate_id"),
        "source": source,
        "immutable_revision": _safe_text(
            record["immutable_revision"],
            field="candidate.immutable_revision",
            maximum_bytes=256,
        ),
        "requested_capability": _requested_capability(
            record["requested_capability"]
        ),
    }


def compute_candidate_screening_stream_key(
    candidate: Mapping[str, Any],
) -> str:
    """Digest the exact source, immutable revision, and requested capability."""

    normalized = _candidate(candidate)
    return canonical_sha256_v1(
        {
            "source": normalized["source"],
            "immutable_revision": normalized["immutable_revision"],
            "requested_capability": normalized["requested_capability"],
        }
    )


def _human_review(value: Any) -> dict[str, Any]:
    field = "human_review"
    record = _mapping(value, field=field)
    keys = frozenset({"status", "reviewer_role_id", "review_reference"})
    _keys(record, field=field, allowed=keys, required=keys)
    return {
        "status": _enum(
            record["status"],
            field=f"{field}.status",
            allowed=frozenset({"reviewed", "unreviewed"}),
        ),
        "reviewer_role_id": _token(
            record["reviewer_role_id"],
            field=f"{field}.reviewer_role_id",
            pattern=_ROLE_RE,
        ),
        "review_reference": _review_reference(
            record["review_reference"], field=f"{field}.review_reference"
        ),
    }


def _derived_outcome(
    checks: Mapping[str, Mapping[str, Any]],
    human_review: Mapping[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    has_fail = False
    has_unknown = False
    for name in _CHECK_ORDER:
        result = str(checks[name]["result"])
        if result == "fail":
            has_fail = True
            reasons.append(f"{name}_failed")
        elif result == "unknown":
            has_unknown = True
            reason = (
                "runtime_provider_new_surface_hold"
                if name == "runtime_provider"
                and checks[name].get("requires_new_surface") is True
                else f"{name}_unknown"
            )
            reasons.append(reason)
    if human_review["status"] != "reviewed":
        has_unknown = True
        reasons.append("human_review_unreviewed")
    status = "reject" if has_fail else ("hold" if has_unknown else "pass")
    return status, reasons


@dataclass(frozen=True)
class CandidateScreeningRecord:
    _record: dict[str, Any]
    source_trust_domain: str

    def to_dict(self) -> dict[str, Any]:
        return _copy(self._record)

    @property
    def record_digest(self) -> str:
        return str(self._record["record_digest"])

    @property
    def stream_key(self) -> str:
        return str(self._record["stream_key"])

    @property
    def overall_status(self) -> str:
        return str(self._record["overall_status"])


def validate_candidate_screening_record(
    value: Mapping[str, Any],
    *,
    source_trust_domain: str = "operator_asserted",
) -> CandidateScreeningRecord:
    """Validate one immutable CandidateScreeningRecord interface contract."""

    trust = _enum(
        source_trust_domain,
        field="source_trust_domain",
        allowed=_TRUST_DOMAINS,
    )
    record = _mapping(value, field="CandidateScreeningRecord")
    keys = frozenset(
        {
            "schema_version",
            "stream_key",
            "sequence",
            "previous_record_digest",
            "record_id",
            "record_digest",
            "candidate",
            "mechanical_checks",
            "human_review",
            "overall_status",
            "reason_codes",
            "issuer_claim",
            "reviewed_at",
            "supersedes_record_id",
            "supersedes_record_digest",
        }
    )
    _keys(record, field="CandidateScreeningRecord", allowed=keys, required=keys)
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise ValueError("CandidateScreeningRecord.schema_version: expected 1")
    candidate = _candidate(record["candidate"])
    stream_key = _sha256(record["stream_key"], field="stream_key")
    expected_stream_key = compute_candidate_screening_stream_key(candidate)
    if stream_key != expected_stream_key:
        raise ValueError("stream_key: candidate decision key mismatch")
    sequence = _integer(
        record["sequence"], field="sequence", minimum=1, maximum=MAX_SCREENING_RECORDS
    )
    previous = _sha256(
        record["previous_record_digest"],
        field="previous_record_digest",
        allow_zero=True,
    )
    supersedes_id = _nullable_token(
        record["supersedes_record_id"], field="supersedes_record_id"
    )
    supersedes_digest = _nullable_sha256(
        record["supersedes_record_digest"], field="supersedes_record_digest"
    )
    if sequence == 1:
        if previous != ZERO_DIGEST or supersedes_id is not None or supersedes_digest is not None:
            raise ValueError("sequence 1 requires zero predecessor and no supersedes target")
    elif (
        previous == ZERO_DIGEST
        or supersedes_id is None
        or supersedes_digest is None
        or supersedes_digest != previous
    ):
        raise ValueError("superseding record requires the exact predecessor identity")

    checks = _mechanical_checks(record["mechanical_checks"])
    human_review = _human_review(record["human_review"])
    status, reasons = _derived_outcome(checks, human_review)
    supplied_reasons = _string_list(
        record["reason_codes"],
        field="reason_codes",
        minimum=0,
        maximum=len(_CHECK_ORDER) + 1,
    )
    if supplied_reasons != reasons:
        raise ValueError("reason_codes: must equal the deterministic failed/unknown checks")
    if _enum(
        record["overall_status"],
        field="overall_status",
        allowed=_OVERALL_STATUSES,
    ) != status:
        raise ValueError("overall_status: does not match screening facts")

    issuer = _enum(
        record["issuer_claim"],
        field="issuer_claim",
        allowed=frozenset(
            {"repository_source", "site_source", "operator_source", "unknown"}
        ),
    )
    if trust == "yolozu_managed" and (
        issuer != "repository_source"
        or human_review["review_reference"]["kind"] != "public_repository_id"
    ):
        raise ValueError("managed screening requires repository review provenance")
    normalized = {
        "schema_version": 1,
        "stream_key": stream_key,
        "sequence": sequence,
        "previous_record_digest": previous,
        "record_id": _token(record["record_id"], field="record_id"),
        "record_digest": _sha256(record["record_digest"], field="record_digest"),
        "candidate": candidate,
        "mechanical_checks": checks,
        "human_review": human_review,
        "overall_status": status,
        "reason_codes": reasons,
        "issuer_claim": issuer,
        "reviewed_at": _utc(record["reviewed_at"], field="reviewed_at"),
        "supersedes_record_id": supersedes_id,
        "supersedes_record_digest": supersedes_digest,
    }
    expected_digest = canonical_sha256_v1(
        normalized, own_digest_field="record_digest"
    )
    if normalized["record_digest"] != expected_digest:
        raise ValueError("record_digest: digest mismatch")
    if len(canonical_json_v1(normalized)) > MAX_CONTROL_RECORD_BYTES:
        raise ValueError("CandidateScreeningRecord exceeds 4 MiB")
    return CandidateScreeningRecord(normalized, trust)


@dataclass(frozen=True)
class CandidateScreeningProjection:
    current_by_stream: dict[str, CandidateScreeningRecord]
    record_by_digest: dict[str, CandidateScreeningRecord]
    source_trust_domain: str
    projection_digest: str


def project_candidate_screening_records(
    records: Sequence[Mapping[str, Any] | CandidateScreeningRecord],
    *,
    source_trust_domain: str = "operator_asserted",
) -> CandidateScreeningProjection:
    """Project independent per-candidate chains by sequence, never timestamps."""

    trust = _enum(
        source_trust_domain,
        field="source_trust_domain",
        allowed=_TRUST_DOMAINS,
    )
    if len(records) > MAX_SCREENING_RECORDS:
        raise ValueError("control_stream_limit_exceeded")
    current: dict[str, CandidateScreeningRecord] = {}
    by_digest: dict[str, CandidateScreeningRecord] = {}
    record_ids: set[str] = set()
    for item in records:
        validated = (
            item
            if isinstance(item, CandidateScreeningRecord)
            else validate_candidate_screening_record(
                item, source_trust_domain=trust
            )
        )
        if validated.source_trust_domain != trust:
            raise ValueError("screening stream mixes source trust domains")
        record = validated.to_dict()
        if record["record_id"] in record_ids:
            raise ValueError("screening stream has duplicate record_id")
        if record["record_digest"] in by_digest:
            raise ValueError("screening stream has duplicate record_digest")
        predecessor = current.get(record["stream_key"])
        if predecessor is None:
            if record["sequence"] != 1:
                raise ValueError("screening stream has a per-key sequence gap")
        else:
            prior = predecessor.to_dict()
            if record["sequence"] != prior["sequence"] + 1:
                raise ValueError("screening stream has a per-key sequence gap or fork")
            if (
                record["previous_record_digest"] != predecessor.record_digest
                or record["supersedes_record_id"] != prior["record_id"]
                or record["supersedes_record_digest"] != predecessor.record_digest
            ):
                raise ValueError("screening stream has a dangling or wrong predecessor")
        current[record["stream_key"]] = validated
        by_digest[record["record_digest"]] = validated
        record_ids.add(record["record_id"])
    heads = [
        {
            "stream_key": stream_key,
            "record_id": current[stream_key].to_dict()["record_id"],
            "record_digest": current[stream_key].record_digest,
        }
        for stream_key in sorted(current, key=lambda item: item.encode("ascii"))
    ]
    return CandidateScreeningProjection(
        current_by_stream=current,
        record_by_digest=by_digest,
        source_trust_domain=trust,
        projection_digest=canonical_sha256_v1(heads),
    )


def _validate_screening_stream_envelope(data: bytes) -> None:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > MAX_SCREENING_STREAM_BYTES:
        raise ValueError("control_stream_limit_exceeded")
    if not data:
        return
    if not data.endswith(b"\n"):
        raise ValueError("screening stream has a partial suffix")
    if data.count(b"\n") > MAX_SCREENING_RECORDS:
        raise ValueError("control_stream_limit_exceeded")


def load_candidate_screening_jsonl_bytes(
    data: bytes,
    *,
    source_trust_domain: str = "operator_asserted",
) -> CandidateScreeningProjection:
    """Load a complete bounded JSONL stream through the shared token reader."""

    _validate_screening_stream_envelope(data)
    try:
        records = load_bounded_jsonl_bytes(
            data,
            label="candidate screening",
            max_records=MAX_SCREENING_RECORDS,
        )
    except ValueError as exc:
        raise ValueError("candidate screening stream is invalid") from exc
    return project_candidate_screening_records(
        records, source_trust_domain=source_trust_domain
    )


def _screening_value(
    *,
    provider_id: str,
    provenance_class: str,
    stream_key: str | None,
    source_revision: str | None,
    status: str,
    current_record: CandidateScreeningRecord | None,
    projection_head_digest: str | None,
    trust_domain: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "provider_id": provider_id,
        "provider_version": "1",
        "provenance_class": provenance_class,
        "screening_stream_key": stream_key,
        "source_revision": source_revision,
        "status": status,
        "current_record_id": None,
        "current_record_digest": None,
        "projection_head_digest": None,
        "trust_domain": trust_domain,
        "observation_digest": ZERO_DIGEST,
    }
    if current_record is not None:
        value["current_record_id"] = current_record.to_dict()["record_id"]
        value["current_record_digest"] = current_record.record_digest
        value["projection_head_digest"] = projection_head_digest
    value["observation_digest"] = canonical_sha256_v1(
        value, own_digest_field="observation_digest"
    )
    return value


def build_screening_eligibility_observation(
    bundle: AlgorithmBundleSpec | Mapping[str, Any],
    projection: CandidateScreeningProjection | None,
) -> ScreeningEligibilityObservation:
    """Project one bundle binding into the selector's file-free observation."""

    checked = (
        bundle if isinstance(bundle, AlgorithmBundleSpec) else validate_algorithm_bundle_spec(bundle)
    )
    bundle_record = checked.to_dict()
    if bundle_record["provenance_class"] == "existing_code_owned":
        return validate_screening_eligibility_observation(
            _screening_value(
                provider_id="no_screening_required",
                provenance_class="existing_code_owned",
                stream_key=None,
                source_revision=None,
                status="not_applicable",
                current_record=None,
                projection_head_digest=None,
                trust_domain="unknown",
            ),
            bundle=checked,
        )

    binding = bundle_record["screening_binding"]
    base = {
        "provider_id": "candidate-screening-projection",
        "provenance_class": "screened_candidate",
        "stream_key": binding["stream_key"],
        "source_revision": binding["source_revision"],
    }
    if projection is None:
        value = _screening_value(
            **base,
            status="absent",
            current_record=None,
            projection_head_digest=None,
            trust_domain="unknown",
        )
        return validate_screening_eligibility_observation(value, bundle=checked)
    if not isinstance(projection, CandidateScreeningProjection):
        raise TypeError("projection must be a validated CandidateScreeningProjection")
    current = projection.current_by_stream.get(binding["stream_key"])
    if projection.source_trust_domain != "yolozu_managed":
        status = "untrusted"
        current_for_observation = None
        trust = projection.source_trust_domain
    elif current is None:
        status = "absent"
        current_for_observation = None
        trust = "unknown"
    else:
        record = current.to_dict()
        if record["candidate"]["immutable_revision"] != binding["source_revision"]:
            status = "revision_mismatch"
            current_for_observation = None
            trust = "yolozu_managed"
        elif record["overall_status"] == "pass" and (
            record["record_id"] != binding["pass_record_id"]
            or record["record_digest"] != binding["pass_record_digest"]
        ):
            status = "conflict"
            current_for_observation = None
            trust = "yolozu_managed"
        else:
            status = {
                "pass": "current_pass",
                "hold": "current_hold",
                "reject": "current_reject",
            }[record["overall_status"]]
            current_for_observation = current
            trust = "yolozu_managed"
    value = _screening_value(
        **base,
        status=status,
        current_record=current_for_observation,
        projection_head_digest=(
            None if current_for_observation is None else current_for_observation.record_digest
        ),
        trust_domain=trust,
    )
    return validate_screening_eligibility_observation(
        value,
        bundle=checked,
        source_trust_domain=trust,
    )
