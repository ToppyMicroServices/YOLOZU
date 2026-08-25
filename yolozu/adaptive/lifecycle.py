"""Reviewed versioned lifecycle maintenance and explicit channel rollback."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .bundles import (
    EMPTY_PROFILE_SET_DIGEST,
    ZERO_DIGEST,
    AlgorithmBundleRegistry,
    AlgorithmBundleSpec,
    BundleLifecycleProjection,
    SupportProfileProjection,
    project_bundle_lifecycle,
    validate_algorithm_bundle_registry,
    validate_bundle_lifecycle_record,
    validate_support_profile_snapshot,
)
from .canonical import canonical_json_v1, canonical_sha256_v1
from .control_records import (
    MAX_CONTROL_RECORD_BYTES,
    MAX_CONTROL_STREAM_BYTES,
    load_bounded_json_bytes,
    load_bounded_jsonl_bytes,
)
from .control_stream import (
    atomic_replace_control_stream,
    read_control_stream_bytes,
    resolve_confined_regular_file,
    resolve_workspace_root,
)
from .evidence import (
    EvidenceActivationProjection,
    EvidenceActivationRecord,
    QualificationReport,
    compute_evidence_selection_key,
    load_evidence_activation_jsonl_bytes,
    project_evidence_activations,
    validate_qualification_report,
)
from .screening import (
    build_screening_eligibility_observation,
    load_candidate_screening_jsonl_bytes,
)
from .support_profiles import (
    MAX_SUPPORT_PROFILE_STREAM_BYTES,
    load_support_profile_jsonl_bytes,
)

__all__ = [
    "MAX_LIFECYCLE_STREAM_BYTES",
    "LifecycleUpdateOutcome",
    "update_image_pipeline_lifecycle",
]


MAX_LIFECYCLE_STREAM_BYTES = 64 * 1024 * 1024
MAX_LIFECYCLE_RECORDS = 4096
CANONICAL_ADAPTIVE_ROOT = Path("yolozu/data/adaptive_routing")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PUBLIC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_ACTOR_ROLES = frozenset({"repo_maintainer", "release_reviewer"})
_PUBLIC_CHANNELS = frozenset({"Experimental", "Stable"})
_OPERATIONS = frozenset(
    {"disable", "enable", "revoke", "review-license", "rollback-channel"}
)
_LICENSE_STATES = frozenset({"approved", "unknown", "blocked"})

FaultHook = Callable[[str], None]


@dataclass(frozen=True)
class _Gate:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class LifecycleUpdateOutcome:
    """Bounded dry-run/apply result for one exact lifecycle operation."""

    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"]
    operation: str
    approved: bool
    family_id: str | None
    channel: str | None
    affected_bundle_spec_digest: str | None
    target_bundle_spec_digest: str | None
    observed_lifecycle_head_digest: str
    observed_bundle_state_event_digest: str | None
    observed_channel_pointer_digest: str | None
    gates: tuple[_Gate, ...]
    planned_records: tuple[dict[str, Any], ...]
    applied_record_digests: tuple[str, ...]
    lifecycle_changed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "lifecycle_update_outcome",
            "status": self.status,
            "operation": self.operation,
            "approved": self.approved,
            "family_id": self.family_id,
            "channel": self.channel,
            "affected_bundle_spec_digest": self.affected_bundle_spec_digest,
            "target_bundle_spec_digest": self.target_bundle_spec_digest,
            "observed_lifecycle_head_digest": self.observed_lifecycle_head_digest,
            "observed_bundle_state_event_digest": (
                self.observed_bundle_state_event_digest
            ),
            "observed_channel_pointer_digest": self.observed_channel_pointer_digest,
            "gates": [gate.to_dict() for gate in self.gates],
            "planned_records": [dict(record) for record in self.planned_records],
            "applied_record_digests": list(self.applied_record_digests),
            "lifecycle_changed": self.lifecycle_changed,
            "bundle_specs_changed": False,
            "support_profiles_changed": False,
            "evidence_changed": False,
            "artifacts_changed": False,
        }


@dataclass(frozen=True)
class _LifecycleState:
    workspace: Path
    registry: AlgorithmBundleRegistry
    bundles: dict[str, AlgorithmBundleSpec]
    support_profiles: SupportProfileProjection
    lifecycle_records: tuple[dict[str, Any], ...]
    lifecycle: BundleLifecycleProjection
    lifecycle_path: Path
    lifecycle_bytes: bytes


def _gate(gates: list[_Gate], code: str, detail: str) -> None:
    if not any(gate.code == code for gate in gates):
        encoded = detail.encode("utf-8")
        if len(encoded) > 512:
            detail = encoded[:509].decode("utf-8", errors="ignore") + "..."
        gates.append(_Gate(code, detail))


def _canonical_file(workspace: Path, basename: str, *, label: str) -> Path:
    path = resolve_confined_regular_file(
        CANONICAL_ADAPTIVE_ROOT / basename,
        workspace=workspace,
        label=label,
    )
    expected = (workspace / CANONICAL_ADAPTIVE_ROOT / basename).resolve(strict=True)
    if path != expected:
        raise ValueError(f"{label} is not the canonical SSOT")
    return path


def _read_canonical(
    workspace: Path,
    basename: str,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[Path, bytes]:
    path = _canonical_file(workspace, basename, label=label)
    return (
        path,
        read_control_stream_bytes(
            path,
            maximum_bytes=maximum_bytes,
            label=label,
        ),
    )


def _load_state(workspace: Path) -> _LifecycleState:
    workspace = resolve_workspace_root(workspace)
    _, registry_bytes = _read_canonical(
        workspace,
        "bundle_specs.json",
        maximum_bytes=MAX_CONTROL_RECORD_BYTES,
        label="bundle registry",
    )
    lifecycle_path, lifecycle_bytes = _read_canonical(
        workspace,
        "bundle_lifecycle.jsonl",
        maximum_bytes=MAX_LIFECYCLE_STREAM_BYTES,
        label="bundle lifecycle",
    )
    _, support_bytes = _read_canonical(
        workspace,
        "support_profiles.jsonl",
        maximum_bytes=MAX_SUPPORT_PROFILE_STREAM_BYTES,
        label="support-profile stream",
    )
    registry_payload = load_bounded_json_bytes(
        registry_bytes,
        label="algorithm bundle registry",
    )
    if not isinstance(registry_payload, Mapping):
        raise ValueError("algorithm bundle registry must be an object")
    registry = validate_algorithm_bundle_registry(registry_payload)
    support_profiles = load_support_profile_jsonl_bytes(
        support_bytes,
        source_trust_domain="yolozu_managed",
    )
    lifecycle_payload = load_bounded_jsonl_bytes(
        lifecycle_bytes,
        label="bundle lifecycle",
        max_records=MAX_LIFECYCLE_RECORDS,
    )
    lifecycle_records = tuple(dict(record) for record in lifecycle_payload)
    lifecycle = project_bundle_lifecycle(
        registry,
        lifecycle_records,
        source_trust_domain="yolozu_managed",
        support_profiles=support_profiles,
    )
    return _LifecycleState(
        workspace=workspace,
        registry=registry,
        bundles=registry.by_spec_digest(),
        support_profiles=support_profiles,
        lifecycle_records=lifecycle_records,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
        lifecycle_bytes=lifecycle_bytes,
    )


def _utc(value: str | datetime | None) -> tuple[str, datetime]:
    if value is None:
        parsed = datetime.now(timezone.utc).replace(microsecond=0)
    elif isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware UTC")
        parsed = value.astimezone(timezone.utc).replace(microsecond=0)
    elif isinstance(value, str):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError("occurred_at must use exact RFC3339 UTC seconds") from exc
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("occurred_at is non-canonical")
    else:
        raise ValueError("occurred_at must use exact RFC3339 UTC seconds")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ"), parsed


def _artifact_members(bundle: AlgorithmBundleSpec) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": artifact["artifact_id"],
            "expected_size_bytes": artifact["expected_size_bytes"],
            "sha256": artifact["sha256"],
        }
        for artifact in bundle.to_dict()["artifacts"]
    ]


def _common_event(
    *,
    state: _LifecycleState,
    operation: str,
    actor_role_id: str,
    public_review_id: str,
    review_status: str,
    reason: str,
    occurred_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stream_id": "bundle-lifecycle-v1",
        "sequence": len(state.lifecycle_records) + 1,
        "previous_event_digest": state.lifecycle.head_digest,
        "maintenance_operation": operation.replace("-", "_"),
        "reviewer_role_id": actor_role_id,
        "actor_role_id": actor_role_id,
        "review_reference": {
            "kind": "public_repository_id",
            "value": public_review_id,
        },
        "review_status": review_status,
        "issuer_claim": "repository_source",
        "reason": reason,
        "occurred_at": occurred_at,
        "event_digest": ZERO_DIGEST,
    }


def _finish_event(value: dict[str, Any]) -> dict[str, Any]:
    value["event_digest"] = canonical_sha256_v1(
        value,
        own_digest_field="event_digest",
    )
    return value


def _global_event(
    *,
    state: _LifecycleState,
    operation: str,
    bundle: AlgorithmBundleSpec,
    bundle_state: Mapping[str, Any],
    license_reviews: Sequence[Mapping[str, Any]],
    actor_role_id: str,
    public_review_id: str,
    review_status: str,
    reason: str,
    occurred_at: str,
) -> dict[str, Any]:
    event_type = "license_review" if operation == "review-license" else operation
    target_state = {
        "disable": "disabled",
        "enable": "enabled",
        "revoke": "revoked",
        "review-license": bundle_state["bundle_state"],
    }[operation]
    value = _common_event(
        state=state,
        operation=operation,
        actor_role_id=actor_role_id,
        public_review_id=public_review_id,
        review_status=review_status,
        reason=reason,
        occurred_at=occurred_at,
    )
    value.update(
        {
            "event_scope": "bundle_global",
            "event_type": event_type,
            "family_id": bundle.to_dict()["family_id"],
            "bundle_spec_digest": bundle.spec_digest,
            "artifact_set_digest": bundle.artifact_set_digest,
            "bundle_state": target_state,
            "artifact_license_reviews": [dict(item) for item in license_reviews],
            "artifact_members": _artifact_members(bundle),
            "existing_runs_reproducible": True,
        }
    )
    return _finish_event(value)


def _event_by_digest(
    lifecycle: BundleLifecycleProjection,
) -> dict[str, dict[str, Any]]:
    return {event.event_digest: event.to_dict() for event in lifecycle.events}


def _load_bindings_proposal(
    path: str | Path,
    *,
    workspace: Path,
) -> list[dict[str, Any]]:
    proposal_path = resolve_confined_regular_file(
        path,
        workspace=workspace,
        label="rollback evidence bindings",
    )
    raw = read_control_stream_bytes(
        proposal_path,
        maximum_bytes=4 * 1024 * 1024,
        label="rollback evidence bindings",
    )
    payload = load_bounded_json_bytes(raw, label="rollback evidence bindings")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "bindings",
    }:
        raise ValueError("rollback evidence bindings fields do not match v1")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("rollback evidence bindings schema_version must be 1")
    bindings = payload["bindings"]
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 32:
        raise ValueError("rollback evidence bindings require 1..32 items")
    normalized: list[dict[str, Any]] = []
    fields = {
        "profile_id",
        "profile_digest",
        "activation_id",
        "activation_digest",
        "trust_domain_claim",
    }
    for index, item in enumerate(bindings):
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError(f"rollback evidence binding {index + 1} fields are invalid")
        value = dict(item)
        if (
            not isinstance(value["profile_id"], str)
            or _COMPONENT_RE.fullmatch(value["profile_id"]) is None
            or not isinstance(value["activation_id"], str)
            or _PUBLIC_ID_RE.fullmatch(value["activation_id"]) is None
            or any(
                not isinstance(value[field], str)
                or _SHA256_RE.fullmatch(value[field]) is None
                for field in ("profile_digest", "activation_digest")
            )
            or value["trust_domain_claim"] != "yolozu_managed"
        ):
            raise ValueError(f"rollback evidence binding {index + 1} is invalid")
        normalized.append(value)
    if raw != canonical_json_v1(dict(payload)) + b"\n":
        raise ValueError("rollback evidence bindings must use canonical_json_v1 plus LF")
    return normalized


def _load_screening(state: _LifecycleState) -> Any:
    _, raw = _read_canonical(
        state.workspace,
        "candidate_screening.jsonl",
        maximum_bytes=64 * 1024 * 1024,
        label="candidate-screening stream",
    )
    return load_candidate_screening_jsonl_bytes(
        raw,
        source_trust_domain="yolozu_managed",
    )


def _load_evidence(
    state: _LifecycleState,
    *,
    as_of: datetime,
) -> tuple[
    EvidenceActivationProjection,
    dict[str, EvidenceActivationRecord],
]:
    _, raw = _read_canonical(
        state.workspace,
        "evidence_activation.jsonl",
        maximum_bytes=MAX_CONTROL_STREAM_BYTES,
        label="evidence activation stream",
    )
    records = load_evidence_activation_jsonl_bytes(raw)
    reports: list[QualificationReport] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        identity = (str(record["report_id"]), str(record["report_digest"]))
        if identity in seen:
            continue
        seen.add(identity)
        report_path = resolve_confined_regular_file(
            CANONICAL_ADAPTIVE_ROOT
            / "qualification_reports"
            / identity[0]
            / "qualification_report.json",
            workspace=state.workspace,
            label="qualification report",
        )
        report_raw = read_control_stream_bytes(
            report_path,
            maximum_bytes=MAX_CONTROL_RECORD_BYTES,
            label="qualification report",
        )
        payload = load_bounded_json_bytes(report_raw, label="QualificationReport")
        if not isinstance(payload, Mapping):
            raise ValueError("qualification report must be an object")
        report = validate_qualification_report(payload, as_of=as_of)
        if report.report_id != identity[0] or report.report_digest != identity[1]:
            raise ValueError("qualification report identity does not match activation")
        reports.append(report)
    projection = project_evidence_activations(
        records,
        reports,
        source_trust_domain="yolozu_managed",
        as_of=as_of,
    )
    validated_events = {
        event.event_digest: event for event in projection.events
    }
    return projection, validated_events


def _active_event_by_key(
    projection: EvidenceActivationProjection,
) -> dict[str, EvidenceActivationRecord]:
    output: dict[str, EvidenceActivationRecord] = {}
    for event in projection.events:
        value = event.to_dict()
        if value["state"] == "active":
            output[value["selection_key"]] = event
        else:
            output.pop(value["selection_key"], None)
    return output


def _expected_rollback_bindings(
    *,
    target: AlgorithmBundleSpec,
    profiles: Sequence[Mapping[str, Any]],
    support_profiles: SupportProfileProjection,
    evidence: EvidenceActivationProjection,
) -> list[dict[str, Any]]:
    active_events = _active_event_by_key(evidence)
    output: list[dict[str, Any]] = []
    for reference in profiles:
        profile = support_profiles.definitions.get(str(reference["profile_id"]))
        if profile is None or profile.profile_digest != reference["profile_digest"]:
            raise ValueError("prior advertised profile is absent or conflicts")
        value = profile.to_dict()
        key = compute_evidence_selection_key(
            bundle_spec_digest=target.spec_digest,
            artifact_set_digest=target.artifact_set_digest,
            environment_fingerprint=value["environment_fingerprint"],
            qualification_workload_fingerprint=value[
                "qualification_workload_fingerprint"
            ],
            protocol_fingerprint=value["protocol_fingerprint"],
        )
        report = evidence.active_by_selection_key.get(key)
        activation = active_events.get(key)
        if report is None or activation is None:
            raise ValueError("target lacks one current activation for every prior profile")
        report_value = report.to_dict()
        activation_value = activation.to_dict()
        if (
            report_value["bundle_spec_digest"] != target.spec_digest
            or report_value["artifact_set_digest"] != target.artifact_set_digest
            or report_value["environment_fingerprint"]
            != value["environment_fingerprint"]
            or report_value["qualification_workload_fingerprint"]
            != value["qualification_workload_fingerprint"]
            or report_value["protocol_fingerprint"] != value["protocol_fingerprint"]
            or activation_value["trust_domain"] != "yolozu_managed"
        ):
            raise ValueError("target activation selection key is mismatched or untrusted")
        output.append(
            {
                "profile_id": reference["profile_id"],
                "profile_digest": reference["profile_digest"],
                "activation_id": activation_value["event_id"],
                "activation_digest": activation_value["event_digest"],
                "trust_domain_claim": "yolozu_managed",
            }
        )
    if len({item["activation_digest"] for item in output}) != len(output):
        raise ValueError("one activation cannot cover multiple advertised profiles")
    return output


def _rollback_event(
    *,
    state: _LifecycleState,
    family_id: str,
    channel: str,
    prior: Mapping[str, Any],
    target: AlgorithmBundleSpec | None,
    target_state: Mapping[str, Any] | None,
    target_prior_assignment_digest: str | None,
    bindings: Sequence[Mapping[str, Any]],
    actor_role_id: str,
    public_review_id: str,
    review_status: str,
    reason: str,
    occurred_at: str,
) -> dict[str, Any]:
    value = _common_event(
        state=state,
        operation="rollback-channel",
        actor_role_id=actor_role_id,
        public_review_id=public_review_id,
        review_status=review_status,
        reason=reason,
        occurred_at=occurred_at,
    )
    if target is None:
        value.update(
            {
                "event_scope": "channel_none",
                "event_type": "channel_none",
                "family_id": family_id,
                "channel": channel,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
                "prior_bundle_spec_digest": prior["bundle_spec_digest"],
                "prior_artifact_set_digest": prior["artifact_set_digest"],
                "prior_support_profile_index_head": prior[
                    "support_profile_index_head"
                ],
                "prior_profile_set_record_digest": prior[
                    "profile_set_record_digest"
                ],
                "prior_profile_set_digest": prior["profile_set_digest"],
            }
        )
    else:
        assert target_state is not None
        assert target_prior_assignment_digest is not None
        value.update(
            {
                "event_scope": "channel_assignment",
                "event_type": "public_assignment",
                "family_id": family_id,
                "channel": channel,
                "target_bundle_spec_digest": target.spec_digest,
                "target_artifact_set_digest": target.artifact_set_digest,
                "target_artifact_license_reviews": [
                    dict(item)
                    for item in target_state["artifact_license_reviews"]
                ],
                "support_profile_index_head": state.support_profiles.head_digest,
                "profile_set_record_id": prior["profile_set_record_id"],
                "profile_set_record_digest": prior[
                    "profile_set_record_digest"
                ],
                "profile_set_digest": prior["profile_set_digest"],
                "profiles": [dict(item) for item in prior["profiles"]],
                "evidence_bindings": [dict(item) for item in bindings],
                "rollback_target_prior_assignment_digest": (
                    target_prior_assignment_digest
                ),
            }
        )
    return _finish_event(value)


def _outcome(
    *,
    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"],
    operation: str,
    approved: bool,
    family_id: str | None,
    channel: str | None,
    affected_spec: str | None,
    target_spec: str | None,
    lifecycle: BundleLifecycleProjection,
    state_event: str | None,
    pointer_event: str | None,
    gates: list[_Gate],
    planned: Sequence[Mapping[str, Any]],
    applied: Sequence[str] = (),
    lifecycle_changed: bool | None,
) -> LifecycleUpdateOutcome:
    return LifecycleUpdateOutcome(
        status=status,
        operation=operation,
        approved=approved,
        family_id=family_id,
        channel=channel,
        affected_bundle_spec_digest=affected_spec,
        target_bundle_spec_digest=target_spec,
        observed_lifecycle_head_digest=lifecycle.head_digest,
        observed_bundle_state_event_digest=state_event,
        observed_channel_pointer_digest=pointer_event,
        gates=tuple(gates),
        planned_records=tuple(dict(record) for record in planned),
        applied_record_digests=tuple(applied),
        lifecycle_changed=lifecycle_changed,
    )


def update_image_pipeline_lifecycle(
    *,
    operation: str,
    workspace_root: str | Path,
    family_id: str | None,
    bundle_spec_digest: str | None,
    artifact_set_digest: str | None,
    expected_lifecycle_head_digest: str | None,
    expected_bundle_state_event_digest: str | None = None,
    channel: str | None = None,
    expected_current_pointer_digest: str | None = None,
    expected_prior_assignment_digest: str | None = None,
    expected_support_profile_index_head: str | None = None,
    expected_prior_support_profile_index_head: str | None = None,
    expected_prior_profile_set_record_digest: str | None = None,
    expected_prior_profile_set_digest: str | None = None,
    target_bundle_spec_digest: str | None = None,
    target_artifact_set_digest: str | None = None,
    evidence_bindings_path: str | Path | None = None,
    license_reviews: Sequence[Mapping[str, Any]] | None = None,
    actor_role_id: str | None = None,
    public_review_id: str | None = None,
    review_status: str | None = None,
    reason: str | None = None,
    approve: bool = False,
    occurred_at: str | datetime | None = None,
    fault_hook: FaultHook | None = None,
) -> LifecycleUpdateOutcome:
    """Dry-run or append one exact reviewed lifecycle maintenance event."""

    gates: list[_Gate] = []
    planned: list[dict[str, Any]] = []
    fallback_lifecycle = BundleLifecycleProjection(ZERO_DIGEST, {}, {}, ())
    loaded: _LifecycleState | None = None
    affected: AlgorithmBundleSpec | None = None
    target: AlgorithmBundleSpec | None = None
    current_state: Mapping[str, Any] | None = None
    current_pointer: Mapping[str, Any] | None = None
    state_event_digest: str | None = None
    pointer_event_digest: str | None = None
    target_prior_assignment_digest: str | None = None

    if operation not in _OPERATIONS:
        _gate(gates, "operation_invalid", "operation must be disable, enable, revoke, review-license, or rollback-channel")
    try:
        occurred_text, occurred = _utc(occurred_at)
    except ValueError as exc:
        _gate(gates, "occurred_at_invalid", str(exc))
        occurred_text, occurred = _utc(None)
    try:
        workspace = resolve_workspace_root(workspace_root)
        loaded = _load_state(workspace)
    except (OSError, TypeError, ValueError) as exc:
        _gate(gates, "canonical_state_invalid", str(exc))
        workspace = Path(os.path.abspath(Path(workspace_root)))

    if family_id is None or _COMPONENT_RE.fullmatch(family_id) is None:
        _gate(gates, "family_id_invalid", "an exact bounded family_id is required")
    if bundle_spec_digest is None or _SHA256_RE.fullmatch(bundle_spec_digest) is None:
        _gate(gates, "bundle_spec_digest_invalid", "an exact affected bundle spec digest is required")
    if artifact_set_digest is None or _SHA256_RE.fullmatch(artifact_set_digest) is None:
        _gate(gates, "artifact_set_digest_invalid", "an exact affected artifact-set digest is required")
    if (
        expected_lifecycle_head_digest is None
        or _SHA256_RE.fullmatch(expected_lifecycle_head_digest) is None
    ):
        _gate(gates, "expected_lifecycle_head_invalid", "an exact lifecycle head digest is required")
    elif loaded is not None and expected_lifecycle_head_digest != loaded.lifecycle.head_digest:
        _gate(gates, "stale_lifecycle_head", "expected lifecycle head does not match current head")
    if actor_role_id not in _ACTOR_ROLES:
        _gate(gates, "actor_role_invalid", "a repository maintainer or release reviewer role is required")
    if not isinstance(public_review_id, str) or _PUBLIC_ID_RE.fullmatch(public_review_id) is None:
        _gate(gates, "public_review_invalid", "a bounded public repository review ID is required")
    if review_status != "approved":
        _gate(gates, "review_not_approved", "an approved public repository review status is required")
    if (
        not isinstance(reason, str)
        or not reason
        or len(reason.encode("utf-8")) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in reason)
    ):
        _gate(gates, "reason_invalid", "a bounded review reason is required")

    if loaded is not None and bundle_spec_digest is not None:
        affected = loaded.bundles.get(bundle_spec_digest)
        if affected is None:
            _gate(gates, "affected_bundle_unknown", "affected bundle spec is not registered")
        else:
            affected_value = affected.to_dict()
            if affected_value["family_id"] != family_id:
                _gate(gates, "affected_family_mismatch", "affected bundle belongs to a different family")
            if affected.artifact_set_digest != artifact_set_digest:
                _gate(gates, "affected_artifact_mismatch", "affected artifact-set digest does not match the immutable spec")
            current_state = loaded.lifecycle.bundle_states.get(bundle_spec_digest)
            if current_state is None:
                _gate(gates, "bundle_state_absent", "affected bundle has no lifecycle state")
            else:
                state_event_digest = str(current_state["event_digest"])

    is_rollback = operation == "rollback-channel"
    if not is_rollback:
        if expected_bundle_state_event_digest is None or _SHA256_RE.fullmatch(expected_bundle_state_event_digest) is None:
            _gate(gates, "expected_bundle_state_invalid", "an exact current bundle-state event digest is required")
        elif state_event_digest is not None and expected_bundle_state_event_digest != state_event_digest:
            _gate(gates, "stale_bundle_state", "expected bundle-state event does not match current state")
        if any(
            value is not None
            for value in (
                channel,
                expected_current_pointer_digest,
                expected_prior_assignment_digest,
                expected_support_profile_index_head,
                expected_prior_support_profile_index_head,
                expected_prior_profile_set_record_digest,
                expected_prior_profile_set_digest,
                target_bundle_spec_digest,
                target_artifact_set_digest,
                evidence_bindings_path,
            )
        ):
            _gate(gates, "operation_argument_conflict", "bundle-global operations forbid channel rollback arguments")
        if operation == "review-license":
            if license_reviews is None:
                _gate(gates, "license_reviews_missing", "review-license requires the complete ordered artifact review set")
        elif license_reviews is not None:
            _gate(gates, "license_reviews_forbidden", "only review-license accepts artifact review input")
        if current_state is not None:
            current_status = current_state["bundle_state"]
            expected_prior = {"disable": "enabled", "enable": "disabled"}.get(operation)
            if current_status == "revoked":
                _gate(gates, "revoked_terminal", "a revoked bundle can never transition again")
            elif expected_prior is not None and current_status != expected_prior:
                _gate(gates, "invalid_state_transition", f"{operation} requires bundle state {expected_prior}")
            if operation == "review-license" and license_reviews is not None:
                normalized_reviews = [dict(item) for item in license_reviews]
                if normalized_reviews == current_state["artifact_license_reviews"]:
                    _gate(gates, "license_review_unchanged", "review-license must change at least one artifact review")
            else:
                normalized_reviews = [
                    dict(item) for item in current_state["artifact_license_reviews"]
                ]
        else:
            normalized_reviews = []
        if loaded is not None and affected is not None and current_state is not None and not gates:
            event = _global_event(
                state=loaded,
                operation=operation,
                bundle=affected,
                bundle_state=current_state,
                license_reviews=normalized_reviews,
                actor_role_id=str(actor_role_id),
                public_review_id=str(public_review_id),
                review_status=str(review_status),
                reason=str(reason),
                occurred_at=occurred_text,
            )
            planned.append(event)
    else:
        if expected_bundle_state_event_digest is not None or license_reviews is not None:
            _gate(gates, "operation_argument_conflict", "rollback-channel forbids bundle-state and license-review arguments")
        if channel not in _PUBLIC_CHANNELS:
            _gate(gates, "channel_invalid", "rollback channel must be Experimental or Stable")
        if loaded is not None and family_id is not None and channel is not None:
            current_pointer = loaded.lifecycle.channel_pointers.get((family_id, channel))
        if current_pointer is None:
            _gate(gates, "current_pointer_absent", "rollback requires one current channel assignment")
        else:
            pointer_event_digest = str(current_pointer["lifecycle_event_digest"])
            if current_pointer["bundle_spec_digest"] != bundle_spec_digest:
                _gate(gates, "affected_pointer_mismatch", "affected bundle is not the current channel pointer")
            if current_pointer["artifact_set_digest"] != artifact_set_digest:
                _gate(gates, "affected_pointer_artifact_mismatch", "affected artifact set is not the current channel pointer")
        for supplied, code, detail in (
            (expected_current_pointer_digest, "expected_pointer_invalid", "an exact current pointer digest is required"),
            (expected_prior_assignment_digest, "expected_prior_assignment_invalid", "the exact prior active assignment digest is required"),
            (expected_support_profile_index_head, "expected_support_head_invalid", "the exact current support-profile head is required"),
            (expected_prior_support_profile_index_head, "expected_prior_support_head_invalid", "the exact prior assignment support head is required"),
            (expected_prior_profile_set_record_digest, "expected_prior_set_record_invalid", "the exact historical profile-set record digest is required"),
            (expected_prior_profile_set_digest, "expected_prior_set_invalid", "the exact prior advertised-set digest is required"),
        ):
            if supplied is None or _SHA256_RE.fullmatch(supplied) is None:
                _gate(gates, code, detail)
        if pointer_event_digest is not None:
            if expected_current_pointer_digest != pointer_event_digest:
                _gate(gates, "stale_channel_pointer", "expected current pointer does not match")
            if expected_prior_assignment_digest != pointer_event_digest:
                _gate(gates, "stale_prior_assignment", "expected prior assignment does not match")
        if loaded is not None:
            if expected_support_profile_index_head != loaded.support_profiles.head_digest:
                _gate(gates, "stale_support_head", "expected current support-profile head does not match")
        if current_pointer is not None:
            if expected_prior_support_profile_index_head != current_pointer["support_profile_index_head"]:
                _gate(gates, "stale_prior_support_head", "expected prior support head does not match the active snapshot")
            if expected_prior_profile_set_record_digest != current_pointer["profile_set_record_digest"]:
                _gate(gates, "stale_prior_set_record", "expected historical profile-set record does not match")
            if expected_prior_profile_set_digest != current_pointer["profile_set_digest"]:
                _gate(gates, "stale_prior_set", "expected prior advertised set does not match")
            if loaded is not None and pointer_event_digest is not None:
                prior_event = _event_by_digest(loaded.lifecycle).get(pointer_event_digest)
                if prior_event is None:
                    _gate(gates, "prior_assignment_missing", "current pointer event is absent from lifecycle history")
                else:
                    try:
                        validate_support_profile_snapshot(
                            prior_event,
                            loaded.support_profiles,
                        )
                    except ValueError as exc:
                        _gate(gates, "prior_snapshot_invalid", str(exc))

        target_state: Mapping[str, Any] | None = None
        expected_bindings: list[dict[str, Any]] = []
        target_none = target_bundle_spec_digest == "none"
        if target_bundle_spec_digest is None:
            _gate(gates, "rollback_target_missing", "rollback requires an exact target bundle digest or none")
        elif not target_none and _SHA256_RE.fullmatch(target_bundle_spec_digest) is None:
            _gate(gates, "rollback_target_invalid", "rollback target must be a lowercase SHA-256 or none")
        if target_none:
            if target_artifact_set_digest is not None or evidence_bindings_path is not None:
                _gate(gates, "none_target_arguments", "rollback target none forbids target artifact and evidence bindings")
        elif loaded is not None and target_bundle_spec_digest is not None:
            target = loaded.bundles.get(target_bundle_spec_digest)
            if target is None:
                _gate(gates, "rollback_target_unknown", "rollback target bundle is not registered")
            else:
                if target.to_dict()["family_id"] != family_id:
                    _gate(gates, "cross_family_target", "rollback target belongs to a different family")
                if target.spec_digest == bundle_spec_digest:
                    _gate(gates, "rollback_target_current", "rollback target must differ from the current pointer")
                if target_artifact_set_digest is None or _SHA256_RE.fullmatch(target_artifact_set_digest) is None:
                    _gate(gates, "target_artifact_invalid", "non-none rollback requires the exact target artifact-set digest")
                elif target.artifact_set_digest != target_artifact_set_digest:
                    _gate(gates, "target_artifact_mismatch", "rollback target artifact set does not match its immutable spec")
                target_state = loaded.lifecycle.bundle_states.get(target.spec_digest)
                if target_state is None:
                    _gate(gates, "target_state_absent", "rollback target has no lifecycle state")
                elif target_state["bundle_state"] != "enabled" or any(
                    review["review_state"] != "approved"
                    for review in target_state["artifact_license_reviews"]
                ):
                    _gate(gates, "target_not_eligible", "rollback target is disabled, revoked, or license-blocked")
                if not any(
                    event.to_dict()["event_type"] == "candidate_registration"
                    and event.to_dict().get("target_bundle_spec_digest") == target.spec_digest
                    for event in loaded.lifecycle.events
                ):
                    _gate(gates, "target_not_candidate_registered", "rollback target lacks immutable Candidate registration")
                prior_target_assignments = [
                    event.to_dict()
                    for event in loaded.lifecycle.events
                    if event.to_dict()["event_type"] == "public_assignment"
                    and event.to_dict().get("family_id") == family_id
                    and event.to_dict().get("channel") == channel
                    and event.to_dict().get("target_bundle_spec_digest")
                    == target.spec_digest
                ]
                if not prior_target_assignments:
                    _gate(
                        gates,
                        "target_not_prior_assignment",
                        "rollback target was never assigned to this exact channel",
                    )
                else:
                    target_prior_assignment_digest = str(
                        prior_target_assignments[-1]["event_digest"]
                    )
                try:
                    screening = build_screening_eligibility_observation(
                        target,
                        _load_screening(loaded),
                    ).to_dict()
                    if screening["status"] not in {"current_pass", "not_applicable"}:
                        _gate(gates, "target_screening_ineligible", "rollback target lacks a current trusted screening pass")
                    if screening["status"] == "current_pass" and screening["trust_domain"] != "yolozu_managed":
                        _gate(gates, "target_screening_untrusted", "rollback target screening is not repository-managed")
                except (OSError, TypeError, ValueError) as exc:
                    _gate(gates, "screening_invalid", str(exc))
                if evidence_bindings_path is None:
                    _gate(gates, "evidence_bindings_missing", "non-none rollback requires exact ordered activation bindings")
                    supplied_bindings: list[dict[str, Any]] = []
                else:
                    try:
                        supplied_bindings = _load_bindings_proposal(
                            evidence_bindings_path,
                            workspace=loaded.workspace,
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        _gate(gates, "evidence_bindings_invalid", str(exc))
                        supplied_bindings = []
                if current_pointer is not None and not any(
                    gate.code in {"canonical_state_invalid", "prior_snapshot_invalid"}
                    for gate in gates
                ):
                    try:
                        evidence, _events = _load_evidence(loaded, as_of=occurred)
                        expected_bindings = _expected_rollback_bindings(
                            target=target,
                            profiles=current_pointer["profiles"],
                            support_profiles=loaded.support_profiles,
                            evidence=evidence,
                        )
                        if supplied_bindings != expected_bindings:
                            _gate(gates, "evidence_bindings_mismatch", "caller bindings must exactly equal the complete derived active set")
                    except (OSError, TypeError, ValueError) as exc:
                        _gate(gates, "target_evidence_invalid", str(exc))
                        expected_bindings = []
                else:
                    expected_bindings = []
            if target is None:
                target_state = None
                expected_bindings = []
        else:
            target_state = None
            expected_bindings = []

        if (
            loaded is not None
            and current_pointer is not None
            and not gates
        ):
            event = _rollback_event(
                state=loaded,
                family_id=str(family_id),
                channel=str(channel),
                prior=current_pointer,
                target=None if target_none else target,
                target_state=target_state,
                target_prior_assignment_digest=(
                    None if target_none else target_prior_assignment_digest
                ),
                bindings=[] if target_none else expected_bindings,
                actor_role_id=str(actor_role_id),
                public_review_id=str(public_review_id),
                review_status=str(review_status),
                reason=str(reason),
                occurred_at=occurred_text,
            )
            planned.append(event)

    lifecycle = fallback_lifecycle if loaded is None else loaded.lifecycle
    if loaded is not None and planned and not gates:
        try:
            validate_bundle_lifecycle_record(
                planned[0],
                registry=loaded.registry,
                source_trust_domain="yolozu_managed",
            )
            projected = project_bundle_lifecycle(
                loaded.registry,
                [*loaded.lifecycle_records, *planned],
                source_trust_domain="yolozu_managed",
                support_profiles=loaded.support_profiles,
            )
            if projected.head_digest != planned[-1]["event_digest"]:
                raise ValueError("planned lifecycle head mismatch")
            if is_rollback:
                observed = projected.channel_pointers.get((str(family_id), str(channel)))
                if target_bundle_spec_digest == "none":
                    if observed is not None:
                        raise ValueError("planned rollback did not produce an unassigned channel")
                elif observed is None or observed["bundle_spec_digest"] != target_bundle_spec_digest:
                    raise ValueError("planned rollback did not produce the exact target pointer")
        except (KeyError, TypeError, ValueError) as exc:
            planned.clear()
            _gate(gates, "planned_transition_invalid", str(exc))

    if not approve:
        return _outcome(
            status="dry_run_ready" if not gates else "dry_run_blocked",
            operation=operation,
            approved=False,
            family_id=family_id,
            channel=channel,
            affected_spec=bundle_spec_digest,
            target_spec=target_bundle_spec_digest,
            lifecycle=lifecycle,
            state_event=state_event_digest,
            pointer_event=pointer_event_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=False,
        )
    if gates or loaded is None or not planned:
        return _outcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            family_id=family_id,
            channel=channel,
            affected_spec=bundle_spec_digest,
            target_spec=target_bundle_spec_digest,
            lifecycle=lifecycle,
            state_event=state_event_digest,
            pointer_event=pointer_event_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=False,
        )

    replacement = loaded.lifecycle_bytes + canonical_json_v1(planned[0]) + b"\n"
    try:
        latest = _load_state(loaded.workspace)
        if (
            latest.lifecycle_bytes != loaded.lifecycle_bytes
            or latest.lifecycle.head_digest != loaded.lifecycle.head_digest
            or latest.registry.to_dict() != loaded.registry.to_dict()
            or latest.support_profiles.head_digest
            != loaded.support_profiles.head_digest
        ):
            raise ValueError(
                "registry, lifecycle, or support-profile state changed before mutation"
            )
        if is_rollback and target_bundle_spec_digest != "none":
            latest_target = latest.bundles.get(str(target_bundle_spec_digest))
            latest_target_state = latest.lifecycle.bundle_states.get(
                str(target_bundle_spec_digest)
            )
            if latest_target is None or latest_target_state is None:
                raise ValueError("rollback target disappeared before mutation")
            if latest_target_state["bundle_state"] != "enabled" or any(
                review["review_state"] != "approved"
                for review in latest_target_state["artifact_license_reviews"]
            ):
                raise ValueError("rollback target became ineligible before mutation")
            screening = build_screening_eligibility_observation(
                latest_target,
                _load_screening(latest),
            ).to_dict()
            if screening["status"] not in {"current_pass", "not_applicable"}:
                raise ValueError("rollback target screening changed before mutation")
            if (
                screening["status"] == "current_pass"
                and screening["trust_domain"] != "yolozu_managed"
            ):
                raise ValueError("rollback target screening trust changed before mutation")
            latest_pointer = latest.lifecycle.channel_pointers.get(
                (str(family_id), str(channel))
            )
            if latest_pointer is None:
                raise ValueError("rollback pointer disappeared before mutation")
            latest_evidence, _events = _load_evidence(latest, as_of=occurred)
            latest_bindings = _expected_rollback_bindings(
                target=latest_target,
                profiles=latest_pointer["profiles"],
                support_profiles=latest.support_profiles,
                evidence=latest_evidence,
            )
            if (
                latest_bindings != expected_bindings
                or latest_bindings != planned[0]["evidence_bindings"]
            ):
                raise ValueError("rollback evidence changed before mutation")
        atomic_replace_control_stream(
            path=loaded.lifecycle_path,
            observed_bytes=loaded.lifecycle_bytes,
            replacement_bytes=replacement,
            maximum_bytes=MAX_LIFECYCLE_STREAM_BYTES,
            label="bundle lifecycle",
            fault_hook=fault_hook,
        )
        readback = _load_state(loaded.workspace)
        if readback.lifecycle.head_digest != planned[0]["event_digest"]:
            raise ValueError("lifecycle readback head mismatch")
        if not readback.lifecycle_bytes.startswith(loaded.lifecycle_bytes):
            raise ValueError("lifecycle readback changed immutable history")
        if is_rollback:
            observed = readback.lifecycle.channel_pointers.get(
                (str(family_id), str(channel))
            )
            if target_bundle_spec_digest == "none":
                if observed is not None:
                    raise ValueError("rollback readback did not preserve none target")
            elif observed is None or observed["bundle_spec_digest"] != target_bundle_spec_digest:
                raise ValueError("rollback readback target mismatch")
        else:
            observed_state = readback.lifecycle.bundle_states.get(str(bundle_spec_digest))
            if observed_state is None or observed_state["event_digest"] != planned[0]["event_digest"]:
                raise ValueError("bundle-state readback mismatch")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _gate(gates, "atomic_write_failed", str(exc))
        return _outcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            family_id=family_id,
            channel=channel,
            affected_spec=bundle_spec_digest,
            target_spec=target_bundle_spec_digest,
            lifecycle=loaded.lifecycle,
            state_event=state_event_digest,
            pointer_event=pointer_event_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=None,
        )
    return _outcome(
        status="applied",
        operation=operation,
        approved=True,
        family_id=family_id,
        channel=channel,
        affected_spec=bundle_spec_digest,
        target_spec=target_bundle_spec_digest,
        lifecycle=readback.lifecycle,
        state_event=(
            planned[0]["event_digest"] if not is_rollback else state_event_digest
        ),
        pointer_event=(
            planned[0]["event_digest"] if is_rollback else pointer_event_digest
        ),
        gates=[],
        planned=planned,
        applied=[planned[0]["event_digest"]],
        lifecycle_changed=True,
    )
