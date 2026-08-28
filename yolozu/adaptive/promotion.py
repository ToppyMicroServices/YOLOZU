"""Reviewed, evidence-bound image-pipeline promotion governance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .bundles import ZERO_DIGEST, AlgorithmBundleSpec, BundleLifecycleProjection
from .canonical import canonical_json_v1, canonical_sha256_v1
from .control_records import MAX_CONTROL_RECORD_BYTES, load_bounded_json_bytes
from .control_stream import (
    atomic_replace_control_stream,
    read_control_stream_bytes,
    resolve_confined_regular_file,
    resolve_workspace_root,
)
from .evidence import (
    EvidenceActivationProjection,
    compute_evidence_selection_key,
)
from .isolation import _code_owned_isolated_services
from .lifecycle import (
    CANONICAL_ADAPTIVE_ROOT,
    MAX_LIFECYCLE_STREAM_BYTES,
    _active_event_by_key,
    _common_event,
    _finish_event,
    _load_bindings_proposal,
    _load_evidence,
    _load_screening,
    _load_state,
    _utc,
)
from .screening import build_screening_eligibility_observation
from .selector import _pipeline_matches_report

__all__ = ["PromotionOutcome", "promote_image_pipeline"]


_SHA256 = frozenset("0123456789abcdef")
_REPOSITORY_ROLES = frozenset({"repo_maintainer", "release_reviewer"})
_OPERATIONS = {
    ("Candidate", "Experimental"): "promote_candidate_to_experimental",
    ("Experimental", "Stable"): "promote_experimental_to_stable",
}
_DRILL_FAILURE_CODES = (
    "artifact_hash_mismatch",
    "runtime_unavailable",
    "out_of_memory_or_timeout",
    "metric_regression",
    "license_failure",
    "interface_contract_failure",
)
FaultHook = Callable[[str], None]


@dataclass(frozen=True)
class _Gate:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class PromotionOutcome:
    """Bounded result for one dry-run or approved promotion attempt."""

    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"]
    operation: str | None
    approved: bool
    family_id: str | None
    source_channel: str | None
    target_channel: str | None
    target_bundle_spec_digest: str | None
    observed_lifecycle_head_digest: str
    observed_source_pointer_digest: str | None
    observed_target_pointer_digest: str | None
    observed_support_profile_index_head: str
    gates: tuple[_Gate, ...]
    planned_record: dict[str, Any] | None
    applied_record_digest: str | None
    lifecycle_changed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "image_pipeline_promotion_outcome",
            "status": self.status,
            "operation": self.operation,
            "approved": self.approved,
            "family_id": self.family_id,
            "source_channel": self.source_channel,
            "target_channel": self.target_channel,
            "target_bundle_spec_digest": self.target_bundle_spec_digest,
            "observed_lifecycle_head_digest": self.observed_lifecycle_head_digest,
            "observed_source_pointer_digest": self.observed_source_pointer_digest,
            "observed_target_pointer_digest": self.observed_target_pointer_digest,
            "observed_support_profile_index_head": (
                self.observed_support_profile_index_head
            ),
            "gates": [gate.to_dict() for gate in self.gates],
            "planned_record": (
                None if self.planned_record is None else dict(self.planned_record)
            ),
            "applied_record_digest": self.applied_record_digest,
            "lifecycle_changed": self.lifecycle_changed,
            "bundle_specs_changed": False,
            "support_profiles_changed": False,
            "evidence_changed": False,
            "screening_changed": False,
            "artifacts_changed": False,
            "derived_projection_written": False,
        }


def _gate(gates: list[_Gate], code: str, detail: str) -> None:
    if any(item.code == code for item in gates):
        return
    encoded = detail.encode("utf-8")
    if len(encoded) > 512:
        detail = encoded[:509].decode("utf-8", errors="ignore") + "..."
    gates.append(_Gate(code, detail))


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _bounded(value: Any, *, maximum: int = 512) -> bool:
    return isinstance(value, str) and 1 <= len(value.encode("utf-8")) <= maximum


def _profile_echo(
    values: Sequence[Mapping[str, Any]] | None,
    gates: list[_Gate],
) -> list[dict[str, str]]:
    if values is None or not 1 <= len(values) <= 32:
        _gate(
            gates,
            "profiles_invalid",
            "the complete ordered 1..32 profile echo is required",
        )
        return []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, Mapping) or set(item) != {
            "profile_id",
            "profile_digest",
        }:
            _gate(
                gates,
                "profiles_invalid",
                f"profile echo {index + 1} fields are invalid",
            )
            continue
        profile_id = item.get("profile_id")
        profile_digest = item.get("profile_digest")
        if not _bounded(profile_id, maximum=128) or not _digest(profile_digest):
            _gate(gates, "profiles_invalid", f"profile echo {index + 1} is invalid")
            continue
        if profile_id in seen:
            _gate(
                gates,
                "profiles_duplicate",
                "profile echo contains a duplicate profile_id",
            )
        seen.add(str(profile_id))
        output.append(
            {"profile_id": str(profile_id), "profile_digest": str(profile_digest)}
        )
    return output


def _reports_for_profiles(
    *,
    bundle: AlgorithmBundleSpec,
    profiles: Sequence[Mapping[str, Any]],
    support_profiles: Any,
    evidence: EvidenceActivationProjection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_events = _active_event_by_key(evidence)
    bindings: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for reference in profiles:
        profile = support_profiles.definitions.get(str(reference["profile_id"]))
        if profile is None or profile.profile_digest != reference["profile_digest"]:
            raise ValueError(
                "advertised profile is absent or conflicts with its digest"
            )
        profile_value = profile.to_dict()
        key = compute_evidence_selection_key(
            bundle_spec_digest=bundle.spec_digest,
            artifact_set_digest=bundle.artifact_set_digest,
            environment_fingerprint=profile_value["environment_fingerprint"],
            qualification_workload_fingerprint=profile_value[
                "qualification_workload_fingerprint"
            ],
            protocol_fingerprint=profile_value["protocol_fingerprint"],
        )
        report = evidence.active_by_selection_key.get(key)
        event = active_events.get(key)
        if report is None or event is None:
            raise ValueError(
                "one current activation is required for every exact profile"
            )
        report_value = report.to_dict()
        event_value = event.to_dict()
        if (
            report_value["status"] != "qualified"
            or report_value["bundle_spec_digest"] != bundle.spec_digest
            or report_value["artifact_set_digest"] != bundle.artifact_set_digest
            or report_value["environment_fingerprint"]
            != profile_value["environment_fingerprint"]
            or report_value["qualification_workload_fingerprint"]
            != profile_value["qualification_workload_fingerprint"]
            or report_value["protocol_fingerprint"]
            != profile_value["protocol_fingerprint"]
            or event_value["trust_domain"] != "yolozu_managed"
            or not _pipeline_matches_report(bundle.to_dict(), report_value)
        ):
            raise ValueError("activation/report identity is mismatched or untrusted")
        bindings.append(
            {
                "profile_id": reference["profile_id"],
                "profile_digest": reference["profile_digest"],
                "activation_id": event_value["event_id"],
                "activation_digest": event_value["event_digest"],
                "trust_domain_claim": "yolozu_managed",
            }
        )
        reports.append(report_value)
    if len({item["activation_digest"] for item in bindings}) != len(bindings):
        raise ValueError("one activation cannot cover multiple advertised profiles")
    return bindings, reports


def _metric_source(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report["execution_mode"] == "batch":
        source = report["conservative_aggregates"]
    else:
        source = report["sustained_section"]
        if source["status"] != "completed":
            raise ValueError("soft-realtime report lacks a complete sustained section")
    if not isinstance(source, Mapping):
        raise ValueError("qualification metrics are unavailable")
    return source


def _ratio(source: Mapping[str, Any], *, mode: str) -> Fraction:
    prefix = "repeat_" if mode == "batch" else ""
    return Fraction(
        int(source[f"{prefix}throughput_processed_count"]) * 1_000_000_000,
        int(source[f"{prefix}throughput_duration_ns"]),
    )


def _memory_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "collector_id",
            "collector_version",
            "collector_source_digest",
            "scope",
            "covered_processes",
            "covered_devices",
        )
    }


def _absolute_profile_gates(
    profile: Mapping[str, Any], report: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    constraints = profile["advertised_constraints"]
    mode = constraints["execution_mode"]
    if report["execution_mode"] != mode:
        return ["execution_mode_metric_mismatch"]
    source = _metric_source(report)
    cold = report["cold_start"]
    if "max_cold_start_ms" in constraints and (
        cold["status"] != "known"
        or Decimal(cold["cold_start_ms"]) > Decimal(constraints["max_cold_start_ms"])
    ):
        reasons.append("cold_start_gate_failed")
    for name in ("p95_latency_ms", "p99_latency_ms"):
        maximum = constraints.get(f"max_{name}")
        if maximum is not None and Decimal(source[name]) > Decimal(maximum):
            reasons.append(f"{name}_gate_failed")
    rss = source["runner_tree_peak_rss"]
    if "max_runner_tree_peak_rss_bytes" in constraints and (
        rss["status"] != "known"
        or rss["value_bytes"] > constraints["max_runner_tree_peak_rss_bytes"]
    ):
        reasons.append("runner_tree_peak_rss_gate_failed")
    accelerator = source["accelerator_process_tree_peak"]
    if "max_accelerator_process_tree_peak_bytes" in constraints and (
        accelerator["status"] != "known"
        or accelerator["value_bytes"]
        > constraints["max_accelerator_process_tree_peak_bytes"]
    ):
        reasons.append("accelerator_memory_gate_failed")
    throughput_key = (
        "min_repeat_throughput_fps" if mode == "batch" else "min_sustained_fps"
    )
    if throughput_key in constraints and _ratio(source, mode=mode) < Fraction(
        Decimal(constraints[throughput_key])
    ):
        reasons.append("throughput_gate_failed")
    quality_requirement = constraints.get("quality_requirement")
    quality = report["quality"]
    if quality_requirement is not None:
        comparable = all(
            quality.get(report_key) == quality_requirement[requirement_key]
            for report_key, requirement_key in (
                ("metric_id", "metric_id"),
                ("direction", "direction"),
                ("threshold_context", "threshold"),
                ("evaluation_dataset_id", "evaluation_dataset_id"),
                ("evaluation_dataset_sha256", "evaluation_dataset_sha256"),
                ("evaluation_protocol_sha256", "evaluation_protocol_sha256"),
                ("evaluation_vocabulary_id", "evaluation_vocabulary_id"),
            )
        )
        if quality.get("status") != "known" or not comparable:
            reasons.append("quality_metric_incomparable")
        else:
            measured = Decimal(quality["measured_value"])
            threshold = Decimal(quality_requirement["threshold"])
            if (
                quality["direction"] == "higher_is_better" and measured < threshold
            ) or (quality["direction"] == "lower_is_better" and measured > threshold):
                reasons.append("quality_gate_failed")
    return reasons


def _stable_profile_contract(profile: Mapping[str, Any]) -> list[str]:
    constraints = profile["advertised_constraints"]
    required = {
        "max_p95_latency_ms",
        "max_p99_latency_ms",
        "max_runner_tree_peak_rss_bytes",
        "quality_requirement",
    }
    required.add(
        "min_repeat_throughput_fps"
        if constraints["execution_mode"] == "batch"
        else "min_sustained_fps"
    )
    missing = sorted(required - set(constraints), key=lambda item: item.encode("ascii"))
    return [f"stable profile has no preregistered {name}" for name in missing]


def _compare_stable_reports(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if candidate["execution_mode"] != baseline["execution_mode"]:
        return ["stable execution mode is incomparable"]
    for field in (
        "environment_fingerprint",
        "qualification_workload_fingerprint",
        "protocol_fingerprint",
        "latency_interval",
        "collector",
    ):
        if candidate[field] != baseline[field]:
            reasons.append(f"stable {field} is incomparable")
    candidate_source = _metric_source(candidate)
    baseline_source = _metric_source(baseline)
    if _ratio(candidate_source, mode=candidate["execution_mode"]) < _ratio(
        baseline_source, mode=baseline["execution_mode"]
    ):
        reasons.append("stable throughput regressed")
    for name in ("p95_latency_ms", "p99_latency_ms"):
        if Decimal(candidate_source[name]) > Decimal(baseline_source[name]):
            reasons.append(f"stable {name} regressed")
    for name in ("runner_tree_peak_rss", "accelerator_process_tree_peak"):
        observed = candidate_source[name]
        prior = baseline_source[name]
        if observed["status"] == "unknown" or prior["status"] == "unknown":
            reasons.append(f"stable {name} is unknown")
            continue
        if observed["status"] != prior["status"]:
            reasons.append(f"stable {name} is incomparable")
            continue
        if _memory_identity(observed) != _memory_identity(prior):
            reasons.append(f"stable {name} collector is incomparable")
        if (
            observed["status"] == "known"
            and observed["value_bytes"] > prior["value_bytes"]
        ):
            reasons.append(f"stable {name} regressed")
    quality = candidate["quality"]
    baseline_quality = baseline["quality"]
    identity_fields = (
        "metric_id",
        "direction",
        "threshold_context",
        "evaluation_dataset_id",
        "evaluation_dataset_sha256",
        "evaluation_protocol_sha256",
        "evaluation_vocabulary_id",
    )
    if (
        quality.get("status") != "known"
        or baseline_quality.get("status") != "known"
        or any(
            quality.get(field) != baseline_quality.get(field)
            for field in identity_fields
        )
    ):
        reasons.append("stable quality metric is unknown or incomparable")
    else:
        observed = Decimal(quality["measured_value"])
        prior = Decimal(baseline_quality["measured_value"])
        if (quality["direction"] == "higher_is_better" and observed < prior) or (
            quality["direction"] == "lower_is_better" and observed > prior
        ):
            reasons.append("stable quality regressed")
    return reasons


def _load_drill_report(
    path: str | Path,
    *,
    workspace: Path,
    family_id: str,
    bundle_spec_digest: str,
    lifecycle_head_digest: str,
    profile_set_digest: str,
) -> dict[str, Any]:
    report_path = resolve_confined_regular_file(
        path, workspace=workspace, label="promotion failure-drill report"
    )
    raw = read_control_stream_bytes(
        report_path,
        maximum_bytes=MAX_CONTROL_RECORD_BYTES,
        label="promotion failure-drill report",
    )
    payload = load_bounded_json_bytes(raw, label="promotion failure-drill report")
    required = {
        "schema_version",
        "family_id",
        "bundle_spec_digest",
        "lifecycle_head_digest",
        "profile_set_digest",
        "status",
        "failure_codes",
        "public_repository_reference",
        "automated_pass_reference",
        "report_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("promotion failure-drill report fields do not match v1")
    value = dict(payload)
    if (
        value["schema_version"] != 1
        or value["family_id"] != family_id
        or value["bundle_spec_digest"] != bundle_spec_digest
        or value["lifecycle_head_digest"] != lifecycle_head_digest
        or value["profile_set_digest"] != profile_set_digest
        or value["status"] != "passed"
        or value["failure_codes"] != list(_DRILL_FAILURE_CODES)
        or not _bounded(value["public_repository_reference"], maximum=128)
        or not _bounded(value["automated_pass_reference"], maximum=128)
        or not _digest(value["report_digest"])
    ):
        raise ValueError(
            "promotion failure-drill report does not match the exact review"
        )
    if value["report_digest"] != canonical_sha256_v1(
        value, own_digest_field="report_digest"
    ):
        raise ValueError("promotion failure-drill report digest mismatch")
    if raw != canonical_json_v1(value) + b"\n":
        raise ValueError(
            "promotion failure-drill report must use canonical_json_v1 plus LF"
        )
    return value


def _outcome(
    *,
    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"],
    operation: str | None,
    approved: bool,
    family_id: str | None,
    source_channel: str | None,
    target_channel: str | None,
    target_spec: str | None,
    lifecycle: BundleLifecycleProjection,
    source_pointer: Mapping[str, Any] | None,
    target_pointer: Mapping[str, Any] | None,
    support_head: str,
    gates: list[_Gate],
    planned: Mapping[str, Any] | None,
    applied: str | None = None,
    lifecycle_changed: bool | None,
) -> PromotionOutcome:
    return PromotionOutcome(
        status=status,
        operation=operation,
        approved=approved,
        family_id=family_id,
        source_channel=source_channel,
        target_channel=target_channel,
        target_bundle_spec_digest=target_spec,
        observed_lifecycle_head_digest=lifecycle.head_digest,
        observed_source_pointer_digest=(
            None
            if source_pointer is None
            else str(source_pointer["lifecycle_event_digest"])
        ),
        observed_target_pointer_digest=(
            None
            if target_pointer is None
            else str(target_pointer["lifecycle_event_digest"])
        ),
        observed_support_profile_index_head=support_head,
        gates=tuple(gates),
        planned_record=None if planned is None else dict(planned),
        applied_record_digest=applied,
        lifecycle_changed=lifecycle_changed,
    )


def promote_image_pipeline(
    *,
    workspace_root: str | Path,
    family_id: str | None,
    source_channel: str | None,
    target_channel: str | None,
    bundle_spec_digest: str | None,
    expected_source_pointer_digest: str | None,
    expected_target_pointer_digest: str | None,
    expected_lifecycle_head_digest: str | None,
    expected_support_profile_index_head: str | None,
    expected_profile_set_record_digest: str | None,
    expected_profile_set_digest: str | None,
    profiles: Sequence[Mapping[str, Any]] | None,
    evidence_bindings_path: str | Path | None,
    rollback_target: str | None,
    approver_role_id: str | None,
    public_review_id: str | None,
    reason: str | None,
    failure_drill_report_path: str | Path | None = None,
    approve: bool = False,
    occurred_at: str | datetime | None = None,
    fault_hook: FaultHook | None = None,
) -> PromotionOutcome:
    """Dry-run or append one exact reviewed public channel assignment."""

    gates: list[_Gate] = []
    planned: dict[str, Any] | None = None
    operation = _OPERATIONS.get((str(source_channel), str(target_channel)))
    if operation is None:
        _gate(
            gates,
            "promotion_pair_invalid",
            "promotion must be Candidate to Experimental or Experimental to Stable",
        )
    try:
        occurred_text, occurred = _utc(occurred_at)
    except ValueError as exc:
        _gate(gates, "occurred_at_invalid", str(exc))
        occurred_text, occurred = _utc(None)
    try:
        state = _load_state(resolve_workspace_root(workspace_root))
    except (OSError, TypeError, ValueError) as exc:
        _gate(gates, "canonical_state_invalid", str(exc))
        fallback = BundleLifecycleProjection(ZERO_DIGEST, {}, {}, ())
        return _outcome(
            status="apply_failed" if approve else "dry_run_blocked",
            operation=operation,
            approved=approve,
            family_id=family_id,
            source_channel=source_channel,
            target_channel=target_channel,
            target_spec=bundle_spec_digest,
            lifecycle=fallback,
            source_pointer=None,
            target_pointer=None,
            support_head=ZERO_DIGEST,
            gates=gates,
            planned=None,
            lifecycle_changed=False,
        )

    source_pointer = state.lifecycle.channel_pointers.get(
        (str(family_id), str(source_channel))
    )
    target_pointer = state.lifecycle.channel_pointers.get(
        (str(family_id), str(target_channel))
    )
    bundle = state.bundles.get(str(bundle_spec_digest))
    profile_echo = _profile_echo(profiles, gates)
    expected_target = (
        None
        if target_pointer is None
        else str(target_pointer["lifecycle_event_digest"])
    )

    if not _bounded(family_id, maximum=128):
        _gate(gates, "family_id_invalid", "an exact bounded family_id is required")
    if not _digest(bundle_spec_digest):
        _gate(
            gates,
            "bundle_spec_digest_invalid",
            "an exact bundle-spec digest is required",
        )
    if bundle is None:
        _gate(
            gates,
            "bundle_unregistered",
            "the immutable target bundle is not registered",
        )
    elif bundle.to_dict()["family_id"] != family_id:
        _gate(
            gates, "bundle_family_mismatch", "target bundle belongs to another family"
        )
    if source_pointer is None:
        _gate(gates, "source_pointer_missing", "the exact source channel is unassigned")
    elif (
        source_pointer["bundle_spec_digest"] != bundle_spec_digest
        or source_pointer["lifecycle_event_digest"] != expected_source_pointer_digest
    ):
        _gate(
            gates,
            "source_pointer_stale",
            "target bundle is not the exact current source pointer",
        )
    if expected_target_pointer_digest not in {"none", expected_target}:
        _gate(
            gates,
            "target_pointer_stale",
            "expected target pointer must be exact or literal none",
        )
    if (expected_target_pointer_digest == "none") != (target_pointer is None):
        _gate(
            gates,
            "target_pointer_stale",
            "expected target pointer does not match current state",
        )
    if expected_lifecycle_head_digest != state.lifecycle.head_digest:
        _gate(gates, "lifecycle_head_stale", "expected lifecycle head does not match")
    if expected_support_profile_index_head != state.support_profiles.head_digest:
        _gate(
            gates,
            "support_profile_head_stale",
            "expected support-profile head does not match",
        )

    assignment = state.support_profiles.assignments.get(
        (str(family_id), str(target_channel))
    )
    if assignment is None:
        _gate(
            gates,
            "profile_set_missing",
            "the canonical target-channel profile set is absent",
        )
        expected_profiles: list[dict[str, Any]] = []
    else:
        expected_profiles = [dict(item) for item in assignment["profiles"]]
        if assignment["record_digest"] != expected_profile_set_record_digest:
            _gate(
                gates,
                "profile_set_record_stale",
                "expected profile-set record digest does not match",
            )
        if assignment["profile_set_digest"] != expected_profile_set_digest:
            _gate(
                gates,
                "profile_set_digest_stale",
                "expected profile-set digest does not match",
            )
        if profile_echo != expected_profiles:
            _gate(
                gates,
                "profile_set_echo_mismatch",
                "profile echo must exactly equal the canonical ordered set",
            )

    bundle_state = (
        None
        if bundle is None
        else state.lifecycle.bundle_states.get(bundle.spec_digest)
    )
    if bundle_state is None:
        _gate(
            gates, "bundle_state_missing", "target bundle lacks global lifecycle state"
        )
    elif bundle_state["bundle_state"] != "enabled" or any(
        review["review_state"] != "approved"
        for review in bundle_state["artifact_license_reviews"]
    ):
        _gate(
            gates,
            "bundle_ineligible",
            "target bundle is disabled, revoked, or license-blocked",
        )
    if approver_role_id not in _REPOSITORY_ROLES:
        _gate(
            gates,
            "approver_role_invalid",
            "a non-personal repository approver role is required",
        )
    if not _bounded(public_review_id, maximum=128):
        _gate(
            gates,
            "public_review_invalid",
            "a bounded public repository review ID is required",
        )
    if not _bounded(reason):
        _gate(gates, "reason_invalid", "a bounded non-empty reason is required")
    if rollback_target not in {"none", "prior"}:
        _gate(
            gates,
            "rollback_target_invalid",
            "rollback target must be explicit none or prior",
        )
    elif rollback_target == "prior" and target_pointer is None:
        _gate(
            gates,
            "rollback_target_missing",
            "prior rollback target requires a current target assignment",
        )

    expected_bindings: list[dict[str, Any]] = []
    candidate_reports: list[dict[str, Any]] = []
    evidence: EvidenceActivationProjection | None = None
    if bundle is not None and expected_profiles:
        try:
            evidence, _ = _load_evidence(state, as_of=occurred)
            expected_bindings, candidate_reports = _reports_for_profiles(
                bundle=bundle,
                profiles=expected_profiles,
                support_profiles=state.support_profiles,
                evidence=evidence,
            )
        except (OSError, TypeError, ValueError) as exc:
            _gate(gates, "promotion_evidence_invalid", str(exc))
    if evidence_bindings_path is None:
        _gate(
            gates,
            "evidence_bindings_missing",
            "complete exact evidence bindings are required",
        )
        supplied_bindings: list[dict[str, Any]] = []
    else:
        try:
            supplied_bindings = _load_bindings_proposal(
                evidence_bindings_path, workspace=state.workspace
            )
        except (OSError, TypeError, ValueError) as exc:
            _gate(gates, "evidence_bindings_invalid", str(exc))
            supplied_bindings = []
    if supplied_bindings != expected_bindings:
        _gate(
            gates,
            "evidence_bindings_mismatch",
            "bindings must exactly equal the complete current managed set",
        )

    if bundle is not None and operation == "promote_candidate_to_experimental":
        try:
            screening = build_screening_eligibility_observation(
                bundle, _load_screening(state)
            ).to_dict()
            if screening["status"] != "current_pass":
                _gate(
                    gates,
                    "screening_pass_missing",
                    "one current screening pass is required",
                )
            elif screening["trust_domain"] != "yolozu_managed":
                _gate(
                    gates,
                    "screening_untrusted",
                    "screening pass is not repository-managed",
                )
        except (OSError, TypeError, ValueError) as exc:
            _gate(gates, "screening_invalid", str(exc))

    bundle_record = None if bundle is None else bundle.to_dict()
    if (
        bundle_record is not None
        and bundle_record["execution_binding"]["status"] != "bound"
    ):
        _gate(
            gates,
            "interface_contract_unbound",
            "promotion requires one immutable bound and validated execution interface contract",
        )
    if (
        bundle_record is not None
        and bundle_record["execution_binding"]["status"] == "bound"
        and bundle_record["execution_trust_class"] == "third_party_isolated"
    ):
        _gate(
            gates,
            "candidate_build_record_missing",
            "third-party promotion requires a current CandidateBuildRecord",
        )
        service = _code_owned_isolated_services().get(bundle_record["runner_id"])
        try:
            capability = None if service is None else service.capability
            if (
                capability is None
                or capability.status != "available"
                or capability.image_present is not True
                or capability.policy_digest
                != bundle_record["execution_isolation_policy_digest"]
            ):
                _gate(
                    gates,
                    "isolation_capability_missing",
                    "third-party promotion requires matching live isolation",
                )
        except (AttributeError, TypeError, ValueError):
            _gate(
                gates,
                "isolation_capability_invalid",
                "live isolation capability is invalid",
            )

    drill: dict[str, Any] | None = None
    comparator_status = "not_applicable_candidate_to_experimental"
    if operation == "promote_experimental_to_stable":
        if failure_drill_report_path is None:
            _gate(
                gates,
                "failure_drill_missing",
                "Stable promotion requires one exact passed failure-drill report",
            )
        elif bundle_spec_digest is not None and expected_profile_set_digest is not None:
            try:
                drill = _load_drill_report(
                    failure_drill_report_path,
                    workspace=state.workspace,
                    family_id=str(family_id),
                    bundle_spec_digest=bundle_spec_digest,
                    lifecycle_head_digest=state.lifecycle.head_digest,
                    profile_set_digest=expected_profile_set_digest,
                )
                if drill["automated_pass_reference"] == public_review_id:
                    _gate(
                        gates,
                        "human_approval_not_distinct",
                        "human review must differ from automated pass",
                    )
            except (OSError, TypeError, ValueError) as exc:
                _gate(gates, "failure_drill_invalid", str(exc))
        if target_pointer is None:
            comparator_status = "comparator_not_applicable_first_assignment"
        else:
            comparator_status = "exact_current_stable"
            if target_pointer["profiles"] != expected_profiles:
                _gate(
                    gates,
                    "stable_profile_set_changed",
                    "Stable replacement forbids profile expansion, narrowing, or reordering",
                )
        for profile_ref, report in zip(
            expected_profiles, candidate_reports, strict=False
        ):
            profile = state.support_profiles.definitions[
                profile_ref["profile_id"]
            ].to_dict()
            for detail in _stable_profile_contract(profile):
                _gate(gates, "stable_profile_contract_incomplete", detail)
            try:
                for detail in _absolute_profile_gates(profile, report):
                    _gate(gates, "stable_absolute_gate_failed", detail)
            except (KeyError, TypeError, ValueError) as exc:
                _gate(gates, "stable_absolute_gate_invalid", str(exc))
        if target_pointer is not None and evidence is not None:
            prior_bundle = state.bundles.get(str(target_pointer["bundle_spec_digest"]))
            if prior_bundle is None:
                _gate(
                    gates,
                    "stable_comparator_missing",
                    "current Stable bundle is unregistered",
                )
            else:
                try:
                    _, baseline_reports = _reports_for_profiles(
                        bundle=prior_bundle,
                        profiles=expected_profiles,
                        support_profiles=state.support_profiles,
                        evidence=evidence,
                    )
                    for candidate, baseline in zip(
                        candidate_reports, baseline_reports, strict=True
                    ):
                        for detail in _compare_stable_reports(candidate, baseline):
                            _gate(gates, "stable_regression", detail)
                except (OSError, TypeError, ValueError) as exc:
                    _gate(gates, "stable_comparator_invalid", str(exc))
    elif failure_drill_report_path is not None:
        _gate(
            gates,
            "failure_drill_forbidden",
            "Candidate to Experimental forbids Stable drill input",
        )

    if (
        rollback_target == "prior"
        and target_pointer is not None
        and evidence is not None
    ):
        prior_bundle = state.bundles.get(str(target_pointer["bundle_spec_digest"]))
        if prior_bundle is None:
            _gate(gates, "rollback_target_ineligible", "prior target is not registered")
        else:
            prior_state = state.lifecycle.bundle_states.get(prior_bundle.spec_digest)
            if (
                prior_state is None
                or prior_state["bundle_state"] != "enabled"
                or any(
                    review["review_state"] != "approved"
                    for review in prior_state["artifact_license_reviews"]
                )
            ):
                _gate(
                    gates,
                    "rollback_target_ineligible",
                    "prior target is not currently eligible",
                )
            try:
                _reports_for_profiles(
                    bundle=prior_bundle,
                    profiles=target_pointer["profiles"],
                    support_profiles=state.support_profiles,
                    evidence=evidence,
                )
            except (OSError, TypeError, ValueError) as exc:
                _gate(gates, "rollback_target_evidence_invalid", str(exc))

    if (
        not gates
        and bundle is not None
        and bundle_state is not None
        and assignment is not None
    ):
        planned = _common_event(
            state=state,
            operation=operation.replace("-", "_"),
            actor_role_id=str(approver_role_id),
            public_review_id=str(public_review_id),
            review_status="approved",
            reason=str(reason),
            occurred_at=occurred_text,
        )
        planned.update(
            {
                "event_scope": "channel_assignment",
                "event_type": "public_assignment",
                "family_id": str(family_id),
                "channel": str(target_channel),
                "target_bundle_spec_digest": bundle.spec_digest,
                "target_artifact_set_digest": bundle.artifact_set_digest,
                "target_artifact_license_reviews": [
                    dict(item) for item in bundle_state["artifact_license_reviews"]
                ],
                "support_profile_index_head": state.support_profiles.head_digest,
                "profile_set_record_id": assignment["record_id"],
                "profile_set_record_digest": assignment["record_digest"],
                "profile_set_digest": assignment["profile_set_digest"],
                "profiles": expected_profiles,
                "evidence_bindings": expected_bindings,
                "promotion_source_channel": str(source_channel),
                "promotion_source_pointer_digest": expected_source_pointer_digest,
                "promotion_target_pointer_digest": expected_target,
                "rollback_target_status": (
                    "prior_assignment"
                    if rollback_target == "prior"
                    else "none_abstention"
                ),
                "stable_comparator_status": comparator_status,
                "failure_drill_report_digest": (
                    None if drill is None else drill["report_digest"]
                ),
                "failure_drill_reference": (
                    None if drill is None else drill["public_repository_reference"]
                ),
                "automated_pass_reference": (
                    None if drill is None else drill["automated_pass_reference"]
                ),
            }
        )
        if rollback_target == "prior" and target_pointer is not None:
            planned["rollback_target_prior_assignment_digest"] = target_pointer[
                "lifecycle_event_digest"
            ]
        planned = _finish_event(planned)
        try:
            from .bundles import project_bundle_lifecycle

            projected = project_bundle_lifecycle(
                state.registry,
                [*state.lifecycle_records, planned],
                source_trust_domain="yolozu_managed",
                support_profiles=state.support_profiles,
            )
            observed = projected.channel_pointers.get(
                (str(family_id), str(target_channel))
            )
            if (
                observed is None
                or observed["lifecycle_event_digest"] != planned["event_digest"]
            ):
                raise ValueError(
                    "planned promotion did not produce the exact target pointer"
                )
        except (KeyError, TypeError, ValueError) as exc:
            planned = None
            _gate(gates, "planned_transition_invalid", str(exc))

    if not approve:
        return _outcome(
            status="dry_run_ready" if not gates else "dry_run_blocked",
            operation=operation,
            approved=False,
            family_id=family_id,
            source_channel=source_channel,
            target_channel=target_channel,
            target_spec=bundle_spec_digest,
            lifecycle=state.lifecycle,
            source_pointer=source_pointer,
            target_pointer=target_pointer,
            support_head=state.support_profiles.head_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=False,
        )
    if gates or planned is None:
        return _outcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            family_id=family_id,
            source_channel=source_channel,
            target_channel=target_channel,
            target_spec=bundle_spec_digest,
            lifecycle=state.lifecycle,
            source_pointer=source_pointer,
            target_pointer=target_pointer,
            support_head=state.support_profiles.head_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=False,
        )

    immutable_paths = (
        CANONICAL_ADAPTIVE_ROOT / "bundle_specs.json",
        CANONICAL_ADAPTIVE_ROOT / "support_profiles.jsonl",
        CANONICAL_ADAPTIVE_ROOT / "evidence_activation.jsonl",
        CANONICAL_ADAPTIVE_ROOT / "candidate_screening.jsonl",
    )
    immutable_before = {
        path: read_control_stream_bytes(
            resolve_confined_regular_file(
                path, workspace=state.workspace, label=str(path)
            ),
            maximum_bytes=MAX_LIFECYCLE_STREAM_BYTES,
            label=str(path),
        )
        for path in immutable_paths
    }
    replacement = state.lifecycle_bytes + canonical_json_v1(planned) + b"\n"
    try:
        latest = _load_state(state.workspace)
        if (
            latest.lifecycle_bytes != state.lifecycle_bytes
            or latest.support_profiles.head_digest != state.support_profiles.head_digest
            or latest.registry.to_dict() != state.registry.to_dict()
        ):
            raise ValueError(
                "canonical registry, lifecycle, or support state changed before mutation"
            )
        latest_evidence, _ = _load_evidence(latest, as_of=occurred)
        latest_bundle = latest.bundles[bundle.spec_digest]
        latest_bindings, _ = _reports_for_profiles(
            bundle=latest_bundle,
            profiles=expected_profiles,
            support_profiles=latest.support_profiles,
            evidence=latest_evidence,
        )
        if latest_bindings != expected_bindings:
            raise ValueError("promotion evidence changed before mutation")
        atomic_replace_control_stream(
            path=state.lifecycle_path,
            observed_bytes=state.lifecycle_bytes,
            replacement_bytes=replacement,
            maximum_bytes=MAX_LIFECYCLE_STREAM_BYTES,
            label="bundle lifecycle",
            fault_hook=fault_hook,
        )
        readback = _load_state(state.workspace)
        pointer = readback.lifecycle.channel_pointers.get(
            (str(family_id), str(target_channel))
        )
        if (
            readback.lifecycle.head_digest != planned["event_digest"]
            or not readback.lifecycle_bytes.startswith(state.lifecycle_bytes)
            or pointer is None
            or pointer["lifecycle_event_digest"] != planned["event_digest"]
            or pointer["profiles"] != expected_profiles
            or pointer["evidence_bindings"] != expected_bindings
        ):
            raise ValueError("promotion lifecycle readback mismatch")
        for path, before in immutable_before.items():
            after = read_control_stream_bytes(
                resolve_confined_regular_file(
                    path, workspace=state.workspace, label=str(path)
                ),
                maximum_bytes=MAX_LIFECYCLE_STREAM_BYTES,
                label=str(path),
            )
            if after != before:
                raise ValueError(f"immutable promotion input changed: {path}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _gate(gates, "atomic_write_failed", str(exc))
        return _outcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            family_id=family_id,
            source_channel=source_channel,
            target_channel=target_channel,
            target_spec=bundle_spec_digest,
            lifecycle=state.lifecycle,
            source_pointer=source_pointer,
            target_pointer=target_pointer,
            support_head=state.support_profiles.head_digest,
            gates=gates,
            planned=planned,
            lifecycle_changed=None,
        )
    return _outcome(
        status="applied",
        operation=operation,
        approved=True,
        family_id=family_id,
        source_channel=source_channel,
        target_channel=target_channel,
        target_spec=bundle_spec_digest,
        lifecycle=readback.lifecycle,
        source_pointer=source_pointer,
        target_pointer=pointer,
        support_head=readback.support_profiles.head_digest,
        gates=[],
        planned=planned,
        applied=planned["event_digest"],
        lifecycle_changed=True,
    )
