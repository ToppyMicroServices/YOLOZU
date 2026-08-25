"""Reviewed dormant support-profile sets and their sole eligibility provider."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .bundle_registry import LoadedAlgorithmBundleRegistry
from .bundles import (
    ZERO_DIGEST,
    AlgorithmBundleSpec,
    SupportProfileProjection,
    SupportProfileSpec,
    project_support_profiles,
    validate_support_profile_record,
    validate_support_profile_spec,
)
from .canonical import canonical_json_v1, canonical_sha256_v1
from .contracts import EnvironmentProfile, ImageJobSpec, QualificationWorkloadProfile
from .control_records import load_bounded_json_bytes, load_bounded_jsonl_bytes
from .control_stream import (
    atomic_replace_control_stream,
    read_control_stream_bytes,
    resolve_confined_regular_file,
    resolve_workspace_root,
)
from .qualification import QUALIFICATION_PROTOCOL_FINGERPRINT
from .selection import (
    SupportProfileEligibilityObservation,
    validate_support_profile_eligibility_observation,
)
from .selector import compute_advertised_gates_digest

__all__ = [
    "MAX_SUPPORT_PROFILE_STREAM_BYTES",
    "SupportProfileReviewOutcome",
    "build_support_profile_eligibility_observation",
    "load_support_profile_jsonl_bytes",
    "review_image_pipeline_support_profiles",
]


MAX_SUPPORT_PROFILE_STREAM_BYTES = 64 * 1024 * 1024
MAX_SUPPORT_PROFILE_RECORDS = 128
CANONICAL_SUPPORT_PROFILE_STREAM = Path(
    "yolozu/data/adaptive_routing/support_profiles.jsonl"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REVIEW_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_REVIEWER_ROLES = frozenset({"repo_maintainer", "release_reviewer"})
_PUBLIC_CHANNELS = frozenset({"Experimental", "Stable"})

FaultHook = Callable[[str], None]


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Compatibility seam over the shared bounded control-stream reader."""

    return read_control_stream_bytes(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )


@dataclass(frozen=True)
class _Gate:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class SupportProfileReviewOutcome:
    """Bounded dry-run/apply result for one dormant complete set review."""

    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"]
    approved: bool
    family_id: str | None
    channel: str | None
    observed_head_digest: str
    observed_current_profile_set_record_digest: str | None
    observed_current_profile_set_digest: str | None
    proposed_profile_set_digest: str | None
    gates: tuple[_Gate, ...]
    planned_records: tuple[dict[str, Any], ...]
    applied_record_digests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "support_profile_review_outcome",
            "status": self.status,
            "approved": self.approved,
            "family_id": self.family_id,
            "channel": self.channel,
            "observed_head_digest": self.observed_head_digest,
            "observed_current_profile_set_record_digest": (
                self.observed_current_profile_set_record_digest
            ),
            "observed_current_profile_set_digest": (
                self.observed_current_profile_set_digest
            ),
            "proposed_profile_set_digest": self.proposed_profile_set_digest,
            "gates": [item.to_dict() for item in self.gates],
            "planned_records": [dict(item) for item in self.planned_records],
            "applied_record_digests": list(self.applied_record_digests),
            "support_state_changed": False,
            "advertised_support_changed": False,
            "scope": "reviewed_dormant_target_only",
        }


def _gate(gates: list[_Gate], code: str, detail: str) -> None:
    if not any(item.code == code for item in gates):
        gates.append(_Gate(code, detail))


def load_support_profile_jsonl_bytes(
    data: bytes,
    *,
    source_trust_domain: str,
) -> SupportProfileProjection:
    """Load one bounded stream and derive trust only from the caller's path boundary."""

    if len(data) > MAX_SUPPORT_PROFILE_STREAM_BYTES:
        raise ValueError("support-profile stream exceeds 64 MiB")
    records = load_bounded_jsonl_bytes(
        data,
        label="support-profile stream",
        max_records=MAX_SUPPORT_PROFILE_RECORDS,
    )
    return project_support_profiles(
        records,
        source_trust_domain=source_trust_domain,
    )


def _proposal(
    raw: bytes,
    *,
    expected_family_id: str | None,
    expected_channel: str | None,
) -> tuple[str, str, tuple[SupportProfileSpec, ...]]:
    payload = load_bounded_json_bytes(raw, label="support-profile set proposal")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "family_id",
        "channel",
        "complete_profile_ids",
        "profiles",
    }:
        raise ValueError("proposal fields do not match v1")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("proposal schema_version must be 1")
    family_id = payload["family_id"]
    channel = payload["channel"]
    if not isinstance(family_id, str) or _COMPONENT_RE.fullmatch(family_id) is None:
        raise ValueError("proposal family_id is invalid")
    if channel not in _PUBLIC_CHANNELS:
        raise ValueError("proposal channel must be Experimental or Stable")
    if expected_family_id is not None and family_id != expected_family_id:
        raise ValueError("proposal family_id does not match the requested family")
    if expected_channel is not None and channel != expected_channel:
        raise ValueError("proposal channel does not match the requested channel")
    raw_ids = payload["complete_profile_ids"]
    raw_profiles = payload["profiles"]
    if not isinstance(raw_ids, list) or not isinstance(raw_profiles, list):
        raise ValueError("proposal profile IDs and profiles must be arrays")
    if not 1 <= len(raw_ids) <= 32 or len(raw_profiles) != len(raw_ids):
        raise ValueError("proposal requires one complete ordered set of 1..32 profiles")
    if any(not isinstance(item, str) or _COMPONENT_RE.fullmatch(item) is None for item in raw_ids):
        raise ValueError("proposal contains an invalid complete profile ID")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("proposal complete profile IDs contain a duplicate")
    profiles = tuple(validate_support_profile_spec(item) for item in raw_profiles)
    observed_ids = [item.to_dict()["profile_id"] for item in profiles]
    if observed_ids != raw_ids:
        raise ValueError("proposal profiles must exactly cover complete_profile_ids in order")
    if raw != canonical_json_v1(dict(payload)) + b"\n":
        raise ValueError("proposal must use exact canonical_json_v1 bytes plus LF")
    return family_id, channel, profiles


def _record(
    *,
    sequence: int,
    previous: str,
    kind: str,
    reviewer_role_id: str,
    public_review_id: str,
    reason: str,
    occurred_at: str,
    profile: Mapping[str, Any] | None = None,
    family_id: str | None = None,
    channel: str | None = None,
    references: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    identity = (
        str(profile["profile_digest"])[:16]
        if profile is not None
        else canonical_sha256_v1(list(references))[:16]
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "stream_id": "support-profiles-v1",
        "sequence": sequence,
        "previous_record_digest": previous,
        "record_id": f"support-{sequence}-{identity}",
        "kind": kind,
        "reviewer_role_id": reviewer_role_id,
        "review_reference": {
            "kind": "public_repository_id",
            "value": public_review_id,
        },
        "issuer_claim": "repository_source",
        "reason": reason,
        "occurred_at": occurred_at,
        "record_digest": ZERO_DIGEST,
    }
    if profile is not None:
        value["profile"] = dict(profile)
    else:
        refs = [dict(item) for item in references]
        value.update(
            {
                "family_id": family_id,
                "channel": channel,
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            }
        )
    value["record_digest"] = canonical_sha256_v1(
        value,
        own_digest_field="record_digest",
    )
    return value


def _utc_now(value: str | datetime | None) -> str:
    if value is None:
        current = datetime.now(timezone.utc).replace(microsecond=0)
    elif isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware UTC")
        current = value.astimezone(timezone.utc).replace(microsecond=0)
    elif isinstance(value, str):
        try:
            current = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError("occurred_at must use exact RFC3339 UTC seconds") from exc
        if current.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("occurred_at is non-canonical")
    else:
        raise ValueError("occurred_at must use exact RFC3339 UTC seconds")
    return current.strftime("%Y-%m-%dT%H:%M:%SZ")


def _outcome(
    *,
    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"],
    approved: bool,
    family_id: str | None,
    channel: str | None,
    projection: SupportProfileProjection,
    current: Mapping[str, Any] | None,
    proposed_set_digest: str | None,
    gates: list[_Gate],
    planned: list[dict[str, Any]],
    applied: Sequence[str] = (),
) -> SupportProfileReviewOutcome:
    return SupportProfileReviewOutcome(
        status=status,
        approved=approved,
        family_id=family_id,
        channel=channel,
        observed_head_digest=projection.head_digest,
        observed_current_profile_set_record_digest=(
            None if current is None else str(current["record_digest"])
        ),
        observed_current_profile_set_digest=(
            None if current is None else str(current["profile_set_digest"])
        ),
        proposed_profile_set_digest=proposed_set_digest,
        gates=tuple(gates),
        planned_records=tuple(planned),
        applied_record_digests=tuple(applied),
    )


def review_image_pipeline_support_profiles(
    *,
    proposal_path: str | Path | None,
    family_id: str | None,
    channel: str | None,
    workspace_root: str | Path,
    expected_head_digest: str | None,
    expected_current_profile_set_record_digest: str | None,
    expected_current_profile_set_digest: str | None,
    expect_no_current_profile_set: bool = False,
    reviewer_role_id: str | None,
    public_review_id: str | None,
    reason: str | None,
    approve: bool = False,
    occurred_at: str | datetime | None = None,
    fault_hook: FaultHook | None = None,
) -> SupportProfileReviewOutcome:
    """Dry-run or append one complete reviewed dormant support-profile set."""

    gates: list[_Gate] = []
    planned: list[dict[str, Any]] = []
    projection = project_support_profiles((), source_trust_domain="yolozu_managed")
    raw_stream = b""
    stream: Path | None = None
    proposal_profiles: tuple[SupportProfileSpec, ...] = ()
    proposed_set_digest: str | None = None
    current: Mapping[str, Any] | None = None

    try:
        workspace = resolve_workspace_root(workspace_root)
    except (OSError, ValueError) as exc:
        _gate(gates, "workspace_invalid", str(exc))
        workspace = Path(os.path.abspath(Path(workspace_root)))

    canonical = workspace / CANONICAL_SUPPORT_PROFILE_STREAM
    try:
        stream = resolve_confined_regular_file(
            canonical,
            workspace=workspace,
            label="canonical support-profile stream",
        )
        if stream != canonical.resolve(strict=True):
            raise ValueError("support-profile stream is not the canonical SSOT")
        raw_stream = _read_regular(
            stream,
            maximum_bytes=MAX_SUPPORT_PROFILE_STREAM_BYTES,
            label="support-profile stream",
        )
        projection = load_support_profile_jsonl_bytes(
            raw_stream,
            source_trust_domain="yolozu_managed",
        )
    except (OSError, ValueError) as exc:
        _gate(gates, "support_profile_stream_invalid", str(exc))

    if family_id is None or _COMPONENT_RE.fullmatch(family_id) is None:
        _gate(gates, "family_id_invalid", "an exact bounded family_id is required")
    if channel not in _PUBLIC_CHANNELS:
        _gate(gates, "channel_invalid", "channel must be Experimental or Stable")

    if proposal_path is None:
        _gate(gates, "proposal_missing", "a workspace-confined canonical proposal is required")
    else:
        try:
            proposal_file = resolve_confined_regular_file(
                proposal_path,
                workspace=workspace,
                label="support-profile proposal",
            )
            proposal_raw = _read_regular(
                proposal_file,
                maximum_bytes=4 * 1024 * 1024,
                label="support-profile proposal",
            )
            proposal_family, proposal_channel, proposal_profiles = _proposal(
                proposal_raw,
                expected_family_id=family_id,
                expected_channel=channel,
            )
            family_id = proposal_family
            channel = proposal_channel
        except (OSError, TypeError, ValueError) as exc:
            _gate(gates, "proposal_invalid", str(exc))

    if expected_head_digest is None or _SHA256_RE.fullmatch(expected_head_digest) is None:
        _gate(gates, "expected_head_invalid", "an exact current global head digest is required")
    elif expected_head_digest != projection.head_digest:
        _gate(gates, "stale_head", "expected global head does not match the observed head")

    if family_id is not None and channel is not None:
        current = projection.assignments.get((family_id, channel))
    if expect_no_current_profile_set:
        if (
            expected_current_profile_set_record_digest is not None
            or expected_current_profile_set_digest is not None
        ):
            _gate(gates, "current_expectation_conflict", "initial none forbids current set digests")
        if current is not None:
            _gate(gates, "stale_current_set", "a current dormant set already exists")
    else:
        if (
            expected_current_profile_set_record_digest is None
            or _SHA256_RE.fullmatch(expected_current_profile_set_record_digest) is None
            or expected_current_profile_set_digest is None
            or _SHA256_RE.fullmatch(expected_current_profile_set_digest) is None
        ):
            _gate(
                gates,
                "current_expectation_missing",
                "replacement requires exact current set-record and set digests",
            )
        elif current is None or (
            current["record_digest"] != expected_current_profile_set_record_digest
            or current["profile_set_digest"] != expected_current_profile_set_digest
        ):
            _gate(gates, "stale_current_set", "expected current dormant set does not match")

    if reviewer_role_id not in _REVIEWER_ROLES:
        _gate(gates, "reviewer_role_invalid", "a non-personal repository review role is required")
    if not isinstance(public_review_id, str) or _REVIEW_RE.fullmatch(public_review_id) is None:
        _gate(gates, "public_review_invalid", "a bounded public repository review ID is required")
    if (
        not isinstance(reason, str)
        or not reason
        or len(reason.encode("utf-8")) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in reason)
    ):
        _gate(gates, "reason_invalid", "a bounded review reason is required")
    try:
        occurred = _utc_now(occurred_at)
    except ValueError as exc:
        _gate(gates, "occurred_at_invalid", str(exc))
        occurred = datetime.now(timezone.utc).replace(microsecond=0).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    if proposal_profiles:
        references = [
            {
                "profile_id": item.to_dict()["profile_id"],
                "profile_digest": item.profile_digest,
            }
            for item in proposal_profiles
        ]
        proposed_set_digest = canonical_sha256_v1(references)
        if current is not None and current["profile_set_digest"] == proposed_set_digest:
            _gate(gates, "profile_set_unchanged", "the proposed complete set is already current")
        sequence = len(projection.record_by_digest) + 1
        previous = projection.head_digest
        for profile in proposal_profiles:
            payload = profile.to_dict()
            existing = projection.definitions.get(payload["profile_id"])
            if existing is not None:
                if existing.to_dict() != payload:
                    _gate(
                        gates,
                        "profile_id_reused",
                        "an immutable profile ID cannot be reused with changed bytes",
                    )
                continue
            item = _record(
                sequence=sequence,
                previous=previous,
                kind="profile_definition",
                reviewer_role_id=str(reviewer_role_id),
                public_review_id=str(public_review_id),
                reason=str(reason),
                occurred_at=occurred,
                profile=payload,
            )
            planned.append(item)
            sequence += 1
            previous = item["record_digest"]
        assignment = _record(
            sequence=sequence,
            previous=previous,
            kind="profile_set_assignment",
            reviewer_role_id=str(reviewer_role_id),
            public_review_id=str(public_review_id),
            reason=str(reason),
            occurred_at=occurred,
            family_id=family_id,
            channel=channel,
            references=references,
        )
        planned.append(assignment)
        if len(projection.record_by_digest) + len(planned) > MAX_SUPPORT_PROFILE_RECORDS:
            _gate(gates, "record_limit_exceeded", "planned records exceed the global 128-record cap")
        if not gates:
            try:
                for item in planned:
                    validate_support_profile_record(
                        item,
                        source_trust_domain="yolozu_managed",
                    )
                projected = project_support_profiles(
                    [
                        *(record.to_dict() for record in projection.record_by_digest.values()),
                        *planned,
                    ],
                    source_trust_domain="yolozu_managed",
                )
                expected = projected.assignments[(str(family_id), str(channel))]
                if (
                    expected["profiles"] != references
                    or expected["profile_set_digest"] != proposed_set_digest
                    or expected["record_digest"] != assignment["record_digest"]
                ):
                    raise ValueError("planned complete set projection mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                planned.clear()
                _gate(gates, "planned_review_invalid", str(exc))

    if not approve:
        return _outcome(
            status="dry_run_ready" if not gates else "dry_run_blocked",
            approved=False,
            family_id=family_id,
            channel=channel,
            projection=projection,
            current=current,
            proposed_set_digest=proposed_set_digest,
            gates=gates,
            planned=planned,
        )
    if gates or stream is None or not planned:
        return _outcome(
            status="apply_failed",
            approved=True,
            family_id=family_id,
            channel=channel,
            projection=projection,
            current=current,
            proposed_set_digest=proposed_set_digest,
            gates=gates,
            planned=planned,
        )

    replacement = raw_stream + b"".join(canonical_json_v1(item) + b"\n" for item in planned)
    try:
        latest = _read_regular(
            stream,
            maximum_bytes=MAX_SUPPORT_PROFILE_STREAM_BYTES,
            label="support-profile stream",
        )
        latest_projection = load_support_profile_jsonl_bytes(
            latest,
            source_trust_domain="yolozu_managed",
        )
        if latest != raw_stream or latest_projection.head_digest != projection.head_digest:
            raise ValueError("support-profile stream changed before mutation")
        atomic_replace_control_stream(
            path=stream,
            observed_bytes=raw_stream,
            replacement_bytes=replacement,
            maximum_bytes=MAX_SUPPORT_PROFILE_STREAM_BYTES,
            label="support-profile stream",
            fault_hook=fault_hook,
        )
        readback = _read_regular(
            stream,
            maximum_bytes=MAX_SUPPORT_PROFILE_STREAM_BYTES,
            label="support-profile stream readback",
        )
        readback_projection = load_support_profile_jsonl_bytes(
            readback,
            source_trust_domain="yolozu_managed",
        )
        observed = readback_projection.assignments[(str(family_id), str(channel))]
        if (
            not readback.startswith(raw_stream)
            or readback_projection.head_digest != planned[-1]["record_digest"]
            or observed["record_digest"] != planned[-1]["record_digest"]
            or observed["profiles"] != planned[-1]["profiles"]
            or observed["profile_set_digest"] != proposed_set_digest
        ):
            raise ValueError("support-profile stream readback mismatch")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _gate(gates, "atomic_write_failed", str(exc))
        return _outcome(
            status="apply_failed",
            approved=True,
            family_id=family_id,
            channel=channel,
            projection=projection,
            current=current,
            proposed_set_digest=proposed_set_digest,
            gates=gates,
            planned=planned,
        )
    return _outcome(
        status="applied",
        approved=True,
        family_id=family_id,
        channel=channel,
        projection=readback_projection,
        current=readback_projection.assignments[(str(family_id), str(channel))],
        proposed_set_digest=proposed_set_digest,
        gates=[],
        planned=planned,
        applied=[item["record_digest"] for item in planned],
    )


def _snapshot_status(
    *,
    pointer: Mapping[str, Any],
    profiles: SupportProfileProjection,
) -> tuple[str, tuple[SupportProfileSpec, ...]]:
    required = (
        "support_profile_index_head",
        "profile_set_record_id",
        "profile_set_record_digest",
        "profile_set_digest",
        "profiles",
    )
    if any(pointer.get(field) is None for field in required):
        return "absent", ()
    head = str(pointer["support_profile_index_head"])
    set_record_digest = str(pointer["profile_set_record_digest"])
    set_record = profiles.record_by_digest.get(set_record_digest)
    head_record = profiles.record_by_digest.get(head)
    if set_record is None or head_record is None:
        return "absent", ()
    set_payload = set_record.to_dict()
    if (
        set_payload.get("kind") != "profile_set_assignment"
        or set_payload.get("record_id") != pointer["profile_set_record_id"]
        or set_payload.get("record_digest") != set_record_digest
        or set_payload.get("profiles") != pointer["profiles"]
        or set_payload.get("profile_set_digest") != pointer["profile_set_digest"]
        or head_record.to_dict()["sequence"] < set_payload["sequence"]
    ):
        return "conflict", ()
    current = head
    prefix: list[Any] = []
    while current != ZERO_DIGEST:
        record = profiles.record_by_digest.get(current)
        if record is None:
            return "absent", ()
        prefix.append(record)
        current = str(record.to_dict()["previous_record_digest"])
    if set_record not in prefix:
        return "conflict", ()
    if any(record.source_trust_domain != "yolozu_managed" for record in prefix):
        return "untrusted", ()
    resolved: list[SupportProfileSpec] = []
    for reference in pointer["profiles"]:
        profile = profiles.definitions.get(reference["profile_id"])
        if profile is None:
            return "absent", ()
        if profile.profile_digest != reference["profile_digest"]:
            return "conflict", ()
        resolved.append(profile)
    return "valid", tuple(resolved)


def build_support_profile_eligibility_observation(
    *,
    registry: LoadedAlgorithmBundleRegistry,
    profiles: SupportProfileProjection,
    bundle: AlgorithmBundleSpec,
    channel: str,
    job: ImageJobSpec,
    environment: EnvironmentProfile,
    workload: QualificationWorkloadProfile,
    evidence_trust_domain: str,
    support_scope: str,
) -> SupportProfileEligibilityObservation | None:
    """Build the only loader-derived support observation used for routing."""

    bundle_record = bundle.to_dict()
    pointer = registry.lifecycle.channel_pointers.get(
        (bundle_record["family_id"], channel)
    )
    if pointer is None or pointer["bundle_spec_digest"] != bundle.spec_digest:
        return None
    snapshot_status, resolved = _snapshot_status(pointer=pointer, profiles=profiles)
    status = snapshot_status
    matched: dict[str, Any] | None = None
    advertised_digest = compute_advertised_gates_digest(job)
    if status == "valid":
        if (
            registry.registry_trust_domain != "yolozu_managed"
            or registry.lifecycle_trust_domain != "yolozu_managed"
        ):
            status = "untrusted"
        elif evidence_trust_domain == "site_managed" and support_scope == "site_qualified":
            status = "not_required_site"
        else:
            matching = []
            for profile in resolved:
                value = profile.to_dict()
                if (
                    value["task"] == job.to_dict()["task"]
                    and value["environment_fingerprint"]
                    == environment.environment_fingerprint
                    and value["qualification_workload_fingerprint"]
                    == workload.workload_fingerprint
                    and value["protocol_fingerprint"]
                    == QUALIFICATION_PROTOCOL_FINGERPRINT
                    and canonical_sha256_v1(value["advertised_constraints"])
                    == advertised_digest
                ):
                    matching.append(value)
            if len(matching) == 1:
                status = "matching_one"
                matched = matching[0]
            elif len(matching) > 1:
                status = "conflict"
            else:
                status = "no_match"
    trust = "yolozu_managed" if status == "matching_one" else (
        "operator_asserted" if status == "untrusted" else "unknown"
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "provider_id": "support_profile_projection",
        "provider_version": "1",
        "family_id": bundle_record["family_id"],
        "bundle_spec_digest": bundle.spec_digest,
        "channel": channel,
        "lifecycle_assignment_id": (
            "assignment-" + pointer["lifecycle_event_digest"][:16]
        ),
        "lifecycle_assignment_digest": pointer["lifecycle_event_digest"],
        "support_profile_index_head_digest": pointer["support_profile_index_head"],
        "profile_set_record_id": pointer["profile_set_record_id"],
        "profile_set_record_digest": pointer["profile_set_record_digest"],
        "profile_set_digest": pointer["profile_set_digest"],
        "status": status,
        "profile_id": None if matched is None else matched["profile_id"],
        "profile_digest": None if matched is None else matched["profile_digest"],
        "environment_fingerprint": (
            None if matched is None else environment.environment_fingerprint
        ),
        "qualification_workload_fingerprint": (
            None if matched is None else workload.workload_fingerprint
        ),
        "protocol_fingerprint": (
            None if matched is None else QUALIFICATION_PROTOCOL_FINGERPRINT
        ),
        "advertised_gates_digest": None if matched is None else advertised_digest,
        "trust_domain": trust,
        "observation_digest": ZERO_DIGEST,
    }
    value["observation_digest"] = canonical_sha256_v1(
        value,
        own_digest_field="observation_digest",
    )
    return validate_support_profile_eligibility_observation(
        value,
        source_trust_domain=trust,
        evidence_trust_domain=evidence_trust_domain,
        support_scope=support_scope,
    )
