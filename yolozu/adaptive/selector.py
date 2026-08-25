"""Pure deterministic qualified-pipeline selection.

The selector consumes only validated, in-memory observations.  It performs no
filesystem, registry, support-stream, model, runner, or network I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from typing import Any, Mapping

from .bundle_registry import LoadedAlgorithmBundleRegistry
from .bundles import AlgorithmBundleSpec, build_fixed_class_mapping
from .canonical import canonical_json_v1, canonical_sha256_v1
from .contracts import (
    ACCELERATOR_PROVIDER_IDS,
    CPU_PROVIDER_IDS,
    EnvironmentProfile,
    ImageJobSpec,
    QualificationWorkloadProfile,
    validate_environment_profile,
    validate_image_job_spec,
    validate_qualification_workload_profile,
)
from .evidence import (
    EvidenceActivationProjection,
    EvidenceActivationRecord,
    LocalArtifactInventory,
    QualificationReport,
    compute_evidence_selection_key,
    validate_local_artifact_inventory,
)
from .selection import (
    ScreeningEligibilityObservation,
    SelectionDecision,
    SupportProfileEligibilityObservation,
    validate_screening_eligibility_observation,
    validate_selection_decision,
    validate_support_profile_eligibility_observation,
)

__all__ = [
    "EvidenceEligibilityObservation",
    "IsolationCapabilityObservation",
    "compute_advertised_gates_digest",
    "evidence_eligibility_from_projection",
    "select_qualified_pipeline",
]


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_EVIDENCE_STATUSES = frozenset(
    {
        "active",
        "absent",
        "inactive",
        "revoked",
        "superseded",
        "expired",
        "future_dated",
        "conflict",
        "untrusted",
        "not_qualified",
    }
)
_EVIDENCE_REASON = {
    "absent": "evidence_inactive",
    "inactive": "evidence_inactive",
    "revoked": "evidence_revoked",
    "superseded": "evidence_superseded",
    "expired": "evidence_expired",
    "future_dated": "evidence_future_dated",
    "conflict": "evidence_conflict",
    "untrusted": "evidence_untrusted",
    "not_qualified": "evidence_not_qualified",
}
_CHANNEL_ORDER = ("Candidate", "Experimental", "Stable")
_MAX_SELECTION_INPUT_BYTES = 128 * 1024 * 1024


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")
    return value


def _utc(value: str, *, field: str) -> datetime:
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
    return parsed


def _as_job(value: ImageJobSpec | Mapping[str, Any]) -> ImageJobSpec:
    return value if isinstance(value, ImageJobSpec) else validate_image_job_spec(value)


def _as_environment(
    value: EnvironmentProfile | Mapping[str, Any],
) -> EnvironmentProfile:
    return (
        value
        if isinstance(value, EnvironmentProfile)
        else validate_environment_profile(value)
    )


def _as_workload(
    value: QualificationWorkloadProfile | Mapping[str, Any],
) -> QualificationWorkloadProfile:
    return (
        value
        if isinstance(value, QualificationWorkloadProfile)
        else validate_qualification_workload_profile(value)
    )


def _canonical_channels(channels: set[str]) -> list[str]:
    return [channel for channel in _CHANNEL_ORDER if channel in channels]


def _identity(bundle: AlgorithmBundleSpec) -> tuple[bytes, bytes, bytes, bytes]:
    record = bundle.to_dict()
    return (
        record["family_id"].encode("utf-8"),
        record["bundle_id"].encode("utf-8"),
        record["bundle_version"].encode("utf-8"),
        bundle.spec_digest.encode("ascii"),
    )


def compute_advertised_gates_digest(
    job: ImageJobSpec | Mapping[str, Any],
) -> str:
    """Hash exactly the request gates advertised by a support profile."""

    record = _as_job(job).to_dict()
    gates: dict[str, Any] = {"execution_mode": record["execution_mode"]}
    for field in (
        "max_cold_start_ms",
        "max_p95_latency_ms",
        "max_runner_tree_peak_rss_bytes",
        "max_accelerator_process_tree_peak_bytes",
        "min_repeat_throughput_fps",
        "min_sustained_fps",
        "quality_requirement",
    ):
        if field in record:
            gates[field] = record[field]
    return canonical_sha256_v1(gates)


@dataclass(frozen=True)
class EvidenceEligibilityObservation:
    """Already validated evidence state for one immutable selection key."""

    selection_key: str
    status: str
    activation_record: EvidenceActivationRecord | None = None
    qualification_report: QualificationReport | None = None

    def __post_init__(self) -> None:
        _sha256(self.selection_key, field="selection_key")
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError("evidence status: unsupported value")
        if self.status != "active":
            if (
                self.activation_record is not None
                or self.qualification_report is not None
            ):
                raise ValueError(
                    "inactive evidence observation forbids record identities"
                )
            return
        if not isinstance(
            self.activation_record, EvidenceActivationRecord
        ) or not isinstance(self.qualification_report, QualificationReport):
            raise ValueError("active evidence requires one activation and report")
        activation = self.activation_record.to_dict()
        report = self.qualification_report.to_dict()
        if activation["state"] != "active":
            raise ValueError("active evidence observation requires an active event")
        if activation["selection_key"] != self.selection_key:
            raise ValueError("active evidence selection key mismatch")
        if (
            activation["report_id"] != report["report_id"]
            or activation["report_digest"] != report["report_digest"]
        ):
            raise ValueError("active evidence report identity mismatch")
        expected_key = compute_evidence_selection_key(
            bundle_spec_digest=report["bundle_spec_digest"],
            artifact_set_digest=report["artifact_set_digest"],
            environment_fingerprint=report["environment_fingerprint"],
            qualification_workload_fingerprint=report[
                "qualification_workload_fingerprint"
            ],
            protocol_fingerprint=report["protocol_fingerprint"],
        )
        if expected_key != self.selection_key:
            raise ValueError("active evidence report selection key mismatch")


def evidence_eligibility_from_projection(
    projection: EvidenceActivationProjection,
) -> dict[str, EvidenceEligibilityObservation]:
    """Derive current evidence observations without choosing among events."""

    if not isinstance(projection, EvidenceActivationProjection):
        raise TypeError("projection must be EvidenceActivationProjection")
    result: dict[str, EvidenceEligibilityObservation] = {}
    events_by_key: dict[str, list[EvidenceActivationRecord]] = {}
    for event in projection.events:
        key = event.to_dict()["selection_key"]
        events_by_key.setdefault(key, []).append(event)
    for key, report in projection.active_by_selection_key.items():
        identity = (report.report_id, report.report_digest)
        active = [
            event
            for event in events_by_key.get(key, [])
            if event.to_dict()["state"] == "active"
            and (
                event.to_dict()["report_id"],
                event.to_dict()["report_digest"],
            )
            == identity
        ]
        if len(active) != 1:
            raise ValueError("active evidence projection lacks one exact active event")
        result[key] = EvidenceEligibilityObservation(key, "active", active[0], report)
    for key, reason in projection.terminal_reason_by_selection_key.items():
        status = "revoked" if reason == "evidence_revoked" else "inactive"
        result[key] = EvidenceEligibilityObservation(key, status)
    return result


@dataclass(frozen=True)
class IsolationCapabilityObservation:
    """Bounded live isolation capability supplied by the calling service."""

    status: str
    backend_id: str | None = None
    backend_version: str | None = None
    isolation_policy_digest: str | None = None
    image_present: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "unsupported"}:
            raise ValueError("isolation capability status: unsupported value")
        if self.status == "supported":
            if not self.backend_id or not self.backend_version:
                raise ValueError("supported isolation requires backend identity")
            if self.isolation_policy_digest is None:
                raise ValueError("supported isolation requires policy digest")
            _sha256(
                self.isolation_policy_digest,
                field="isolation_policy_digest",
            )
            if type(self.image_present) is not bool:
                raise ValueError("supported isolation requires image presence status")
        elif any(
            item is not None
            for item in (
                self.backend_id,
                self.backend_version,
                self.isolation_policy_digest,
                self.image_present,
            )
        ):
            raise ValueError("unsupported isolation forbids capability claims")


def _workload_matches_job(job: Mapping[str, Any], workload: Mapping[str, Any]) -> bool:
    for field in (
        "task",
        "input_mode",
        "execution_mode",
        "compute_policy",
        "provider_allowlist",
        "precision_allowlist",
        "batch_size",
        "concurrency",
        "max_results_per_image",
    ):
        if workload[field] != job[field]:
            return False
    quality = job.get("quality_requirement")
    workload_quality = workload.get("quality_identity")
    if quality is None:
        return workload_quality is None
    return workload_quality == {
        key: quality[key]
        for key in (
            "metric_id",
            "direction",
            "evaluation_dataset_id",
            "evaluation_dataset_sha256",
            "evaluation_protocol_sha256",
            "evaluation_vocabulary_id",
        )
    }


def _support_reason(status: str) -> str:
    return {
        "no_match": "support_profile_mismatch",
        "absent": "support_profile_mismatch",
        "untrusted": "support_profile_untrusted",
        "conflict": "support_profile_conflict",
    }.get(status, "support_profile_mismatch")


def _support_matches_pointer(
    observation: Mapping[str, Any], pointer: Mapping[str, Any]
) -> bool:
    return all(
        observation[field] == pointer[pointer_field]
        for field, pointer_field in (
            ("lifecycle_assignment_digest", "lifecycle_event_digest"),
            ("support_profile_index_head_digest", "support_profile_index_head"),
            ("profile_set_record_id", "profile_set_record_id"),
            ("profile_set_record_digest", "profile_set_record_digest"),
            ("profile_set_digest", "profile_set_digest"),
        )
    )


def _provider_probe_ids(provider: str) -> set[str]:
    observed = {provider}
    if provider in CPU_PROVIDER_IDS:
        observed.add("cpu")
    if "cuda" in provider:
        observed.add("cuda")
    if "tensorrt" in provider:
        observed.add("tensorrt")
    if "mps" in provider:
        observed.add("mps")
    if "coreml" in provider:
        observed.add("coreml")
    return observed


def _runtime_and_hardware_reasons(
    *,
    bundle: Mapping[str, Any],
    job: Mapping[str, Any],
    environment: Mapping[str, Any],
    isolation: IsolationCapabilityObservation | None,
) -> list[str]:
    reasons: set[str] = set()
    runtime = bundle["runtime"]
    provider = runtime["provider_id"]

    if provider not in CPU_PROVIDER_IDS | ACCELERATOR_PROVIDER_IDS:
        reasons.add("provider_not_allowed")
    if job["compute_policy"] == "cpu_only" and provider not in CPU_PROVIDER_IDS:
        reasons.add("compute_policy_mismatch")
    if (
        job["compute_policy"] == "accelerator_required"
        and provider not in ACCELERATOR_PROVIDER_IDS
    ):
        reasons.add("compute_policy_mismatch")
    if job["provider_allowlist"] and provider not in job["provider_allowlist"]:
        reasons.add("provider_not_allowed")
    if (
        job["precision_allowlist"]
        and runtime["precision"] not in job["precision_allowlist"]
    ):
        reasons.add("precision_not_allowed")

    os_probe = environment["os"]
    if os_probe["probe_status"] == "present":
        architecture = runtime["architecture"]
        if architecture != "any" and os_probe["architecture"] != architecture:
            reasons.add("hardware_unavailable")
    elif os_probe["probe_status"] == "absent":
        reasons.add("hardware_unavailable")
    else:
        reasons.add("hardware_probe_unknown")

    runtime_matches = [
        item
        for item in environment["runtimes"]
        if item["runtime_id"] == runtime["runtime_id"]
    ]
    if len(runtime_matches) != 1:
        reasons.add("runtime_probe_unknown")
    else:
        observed_runtime = runtime_matches[0]
        if observed_runtime["probe_status"] == "absent":
            reasons.add("runtime_unavailable")
        elif observed_runtime["probe_status"] != "present":
            reasons.add("runtime_probe_unknown")
        elif observed_runtime["version"] != runtime[
            "runtime_version"
        ] or not _provider_probe_ids(provider).intersection(
            observed_runtime["provider_ids"]
        ):
            reasons.add("runtime_unavailable")

    if provider in CPU_PROVIDER_IDS:
        cpu_status = environment["cpu"]["probe_status"]
        if cpu_status == "absent":
            reasons.add("hardware_unavailable")
        elif cpu_status != "present":
            reasons.add("hardware_probe_unknown")
    elif provider in ACCELERATOR_PROVIDER_IDS:
        aliases = {provider}
        if "cuda" in provider or "tensorrt" in provider:
            aliases.update({"cuda", "tensorrt"})
        if "mps" in provider or "coreml" in provider:
            aliases.update({"mps", "coreml"})
        accelerators = [
            item
            for item in environment["accelerators"]
            if item["accelerator_id"] in aliases
        ]
        present = [item for item in accelerators if item["probe_status"] == "present"]
        if not present:
            if accelerators and all(
                item["probe_status"] == "absent" for item in accelerators
            ):
                reasons.add("hardware_unavailable")
            else:
                reasons.add("hardware_probe_unknown")
        elif runtime["accelerator_requirement"] in {"optional", "required"}:
            required = runtime["minimum_accelerator_memory_bytes"]
            known = [
                item["memory"]["value_bytes"]
                for item in present
                if item["memory"]["probe_status"] == "present"
            ]
            if known and max(known) < required:
                reasons.add("hardware_unavailable")
            elif not known:
                reasons.add("hardware_probe_unknown")

    if bundle["execution_trust_class"] == "third_party_isolated":
        if isolation is None:
            reasons.add("isolation_required")
        elif isolation.status != "supported":
            reasons.add("isolation_unsupported")
        elif isolation.image_present is not True:
            reasons.add("isolation_image_missing")
        elif (
            isolation.isolation_policy_digest
            != bundle["execution_isolation_policy_digest"]
        ):
            reasons.add("isolation_policy_mismatch")
    return sorted(reasons, key=lambda item: item.encode("ascii"))


def _pipeline_matches_report(
    bundle: Mapping[str, Any], report: Mapping[str, Any]
) -> bool:
    expected_pipeline = {
        name: {
            "id": bundle[name]["id"],
            "version": bundle[name]["version"],
            "source_digest": bundle[name]["digest"],
        }
        for name in ("decoder", "preprocess", "postprocess")
    }
    expected_pipeline["model_input"] = {
        "id": "bundle_model_input_shapes",
        "version": "1",
        "source_digest": canonical_sha256_v1(bundle["model_input_shapes"]),
    }
    if report["resolved_pipeline"] != expected_pipeline:
        return False
    runtime = bundle["runtime"]
    return report["source_runtime_provenance"] == {
        "model_source_id": bundle["model_source_id"],
        "model_revision": bundle["model_revision"],
        "runtime_id": runtime["runtime_id"],
        "runtime_version": runtime["runtime_version"],
        "provider_id": runtime["provider_id"],
        "provider_version": runtime["provider_version"],
    }


def _ratio(count: int, duration_ns: int) -> Fraction:
    return Fraction(count * 1_000_000_000, duration_ns)


def _decimal_fraction(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _quality_rank(value: str, direction: str) -> Decimal:
    measured = Decimal(value)
    return -measured if direction == "higher_is_better" else measured


def _performance_rank_and_reasons(
    *,
    report: Mapping[str, Any],
    job: Mapping[str, Any],
    identity: tuple[bytes, bytes, bytes, bytes],
) -> tuple[list[str], tuple[Any, ...] | None]:
    reasons: set[str] = set()
    if report["execution_mode"] != job["execution_mode"]:
        return ["execution_mode_metric_mismatch"], None

    if job["execution_mode"] == "batch":
        source = report["conservative_aggregates"]
        throughput = _ratio(
            source["repeat_throughput_processed_count"],
            source["repeat_throughput_duration_ns"],
        )
    else:
        source = report["sustained_section"]
        if source["status"] != "completed":
            return ["execution_mode_metric_mismatch"], None
        throughput = _ratio(
            source["throughput_processed_count"],
            source["throughput_duration_ns"],
        )
    p95 = Decimal(source["p95_latency_ms"])
    rss_record = source["runner_tree_peak_rss"]
    accelerator_record = source["accelerator_process_tree_peak"]
    rss = rss_record["value_bytes"] if rss_record["status"] == "known" else None
    accelerator = (
        accelerator_record["value_bytes"]
        if accelerator_record["status"] == "known"
        else None
    )

    quality_requirement = job.get("quality_requirement")
    quality = report["quality"]
    quality_key: Decimal | None = None
    if quality_requirement is not None:
        if quality["status"] != "known":
            reasons.add("requested_metric_unknown")
        else:
            measured = Decimal(quality["measured_value"])
            threshold = Decimal(quality_requirement["threshold"])
            if (
                quality_requirement["direction"] == "higher_is_better"
                and measured < threshold
            ) or (
                quality_requirement["direction"] == "lower_is_better"
                and measured > threshold
            ):
                reasons.add("quality_gate_failed")
            quality_key = _quality_rank(quality["measured_value"], quality["direction"])

    cold = report["cold_start"]
    if "max_cold_start_ms" in job:
        if cold["status"] != "known":
            reasons.add("cold_start_unknown")
        elif Decimal(cold["cold_start_ms"]) > Decimal(job["max_cold_start_ms"]):
            reasons.add("cold_start_above_requirement")
    if "max_p95_latency_ms" in job and p95 > Decimal(job["max_p95_latency_ms"]):
        reasons.add("p95_latency_gate_failed")
    if "max_runner_tree_peak_rss_bytes" in job:
        if rss is None:
            reasons.add("requested_metric_unknown")
        elif rss > job["max_runner_tree_peak_rss_bytes"]:
            reasons.add("peak_rss_gate_failed")
    if "max_accelerator_process_tree_peak_bytes" in job:
        if accelerator is None:
            reasons.add("requested_metric_unknown")
        elif accelerator > job["max_accelerator_process_tree_peak_bytes"]:
            reasons.add("accelerator_memory_gate_failed")
    if "min_repeat_throughput_fps" in job and throughput < _decimal_fraction(
        job["min_repeat_throughput_fps"]
    ):
        reasons.add("repeat_throughput_gate_failed")
    if "min_sustained_fps" in job and throughput < _decimal_fraction(
        job["min_sustained_fps"]
    ):
        reasons.add("sustained_fps_gate_failed")

    policy = job["ranking_policy"]
    if p95 is None:  # pragma: no cover - validated reports always provide this
        reasons.add("ranking_metric_unknown")
    if policy in {"accuracy_first", "latency_first", "memory_first"} and rss is None:
        if "max_runner_tree_peak_rss_bytes" not in job:
            reasons.add("ranking_metric_unknown")
    if (
        policy == "memory_first"
        and job["compute_policy"] == "accelerator_required"
        and accelerator is None
        and "max_accelerator_process_tree_peak_bytes" not in job
    ):
        reasons.add("ranking_metric_unknown")
    if policy == "accuracy_first" and quality_key is None:
        if quality_requirement is None:
            reasons.add("ranking_metric_unknown")
    if reasons:
        return sorted(reasons, key=lambda item: item.encode("ascii")), None

    optional_quality = () if quality_requirement is None else (quality_key,)
    if policy == "accuracy_first":
        rank = (quality_key, p95, rss, *identity)
    elif policy == "latency_first":
        rank = (p95, *optional_quality, rss, *identity)
    elif policy == "throughput_first":
        rank = (-throughput, p95, *optional_quality, *identity)
    elif job["compute_policy"] == "accelerator_required":
        rank = (accelerator, rss, p95, *optional_quality, *identity)
    else:
        rank = (rss, p95, *optional_quality, *identity)
    return [], rank


def _trace_pass(trace: list[dict[str, Any]], step: int) -> None:
    trace.append({"step": step, "status": "pass", "reason_code": None, "detail": None})


def _trace_fail(trace: list[dict[str, Any]], step: int, reasons: list[str]) -> None:
    trace.append(
        {
            "step": step,
            "status": "failed",
            "reason_code": reasons[0],
            "detail": None,
        }
    )


def select_qualified_pipeline(
    *,
    decision_id: str,
    decided_at: str,
    job: ImageJobSpec | Mapping[str, Any],
    local_input_digest: str,
    artifact_resolver_state_digest: str,
    environment: EnvironmentProfile | Mapping[str, Any],
    workload: QualificationWorkloadProfile | Mapping[str, Any],
    protocol_fingerprint: str,
    registry: LoadedAlgorithmBundleRegistry,
    screening_observations: Mapping[str, ScreeningEligibilityObservation],
    support_profile_observations: Mapping[
        tuple[str, str], SupportProfileEligibilityObservation
    ],
    artifact_inventories: Mapping[str, LocalArtifactInventory],
    evidence_observations: Mapping[str, EvidenceEligibilityObservation],
    isolation_capabilities: Mapping[str, IsolationCapabilityObservation] | None = None,
    as_of: str | datetime | None = None,
) -> SelectionDecision:
    """Select one qualified bundle or return a complete explicit abstention."""

    if not isinstance(registry, LoadedAlgorithmBundleRegistry):
        raise TypeError("registry must be a validated LoadedAlgorithmBundleRegistry")
    if len(registry.bundles) > 128 or len(evidence_observations) > 512:
        raise ValueError("registry/evidence_limit_exceeded")
    if len(screening_observations) > 128 or len(support_profile_observations) > 256:
        raise ValueError("registry/evidence_limit_exceeded")
    if len(artifact_inventories) > 128 or len(isolation_capabilities or {}) > 128:
        raise ValueError("registry/evidence_limit_exceeded")
    evidence_bytes = 0
    for key, observation in evidence_observations.items():
        if not isinstance(observation, EvidenceEligibilityObservation):
            raise TypeError("evidence observations must already be validated")
        if key != observation.selection_key:
            raise ValueError("evidence observation map key mismatch")
        payload: dict[str, Any] = {
            "selection_key": observation.selection_key,
            "status": observation.status,
        }
        if observation.activation_record is not None:
            payload["activation_record"] = observation.activation_record.to_dict()
        if observation.qualification_report is not None:
            payload["qualification_report"] = observation.qualification_report.to_dict()
        evidence_bytes += len(canonical_json_v1(payload))
        if evidence_bytes > _MAX_SELECTION_INPUT_BYTES:
            raise ValueError("registry/evidence_limit_exceeded")

    checked_job = _as_job(job)
    checked_environment = _as_environment(environment)
    checked_workload = _as_workload(workload)
    job_record = checked_job.to_dict()
    environment_record = checked_environment.to_dict()
    workload_record = checked_workload.to_dict()
    if not _workload_matches_job(job_record, workload_record):
        raise ValueError("job/workload interface contract mismatch")
    protocol = _sha256(protocol_fingerprint, field="protocol_fingerprint")
    local_input = _sha256(local_input_digest, field="local_input_digest")
    resolver_state = _sha256(
        artifact_resolver_state_digest,
        field="artifact_resolver_state_digest",
    )
    decision_time = _utc(decided_at, field="decided_at")
    if as_of is None:
        current = decision_time
    elif isinstance(as_of, datetime):
        if as_of.tzinfo is None or as_of.utcoffset() != timezone.utc.utcoffset(as_of):
            raise ValueError("as_of: expected timezone-aware UTC datetime")
        current = as_of.replace(microsecond=0)
    else:
        current = _utc(as_of, field="as_of")
    if decision_time > current:
        raise ValueError("decided_at: future decision is invalid")

    advertised_gates_digest = compute_advertised_gates_digest(checked_job)
    bundle_by_digest = registry.by_spec_digest()
    if set(screening_observations) != set(bundle_by_digest):
        raise ValueError(
            "screening observations must cover every registry entry exactly"
        )
    unknown_artifacts = set(artifact_inventories) - set(bundle_by_digest)
    if unknown_artifacts:
        raise ValueError("artifact inventories reference unknown registry entries")
    for key in support_profile_observations:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in bundle_by_digest
            or key[1] not in {"Experimental", "Stable"}
        ):
            raise ValueError("support observations reference an unknown bundle/channel")

    prepared: list[dict[str, Any]] = []
    allowed_channels = set(job_record["allowed_maturities"])
    lifecycle = registry.lifecycle
    isolation_by_spec = isolation_capabilities or {}
    for bundle in sorted(registry.bundles, key=_identity):
        bundle_record = bundle.to_dict()
        spec_digest = bundle.spec_digest
        screening_value = screening_observations[spec_digest]
        if not isinstance(screening_value, ScreeningEligibilityObservation):
            raise TypeError("screening observations must already be validated")
        screening = validate_screening_eligibility_observation(
            screening_value.to_dict(), bundle=bundle
        ).to_dict()

        evidence_key = compute_evidence_selection_key(
            bundle_spec_digest=spec_digest,
            artifact_set_digest=bundle.artifact_set_digest,
            environment_fingerprint=checked_environment.environment_fingerprint,
            qualification_workload_fingerprint=checked_workload.workload_fingerprint,
            protocol_fingerprint=protocol,
        )
        evidence_observation = evidence_observations.get(evidence_key)
        evidence_trust = "unknown"
        evidence_identity: dict[str, Any] | None = None
        report: dict[str, Any] | None = None
        if evidence_observation is not None:
            if evidence_observation.status == "active":
                activation_object = evidence_observation.activation_record
                report_object = evidence_observation.qualification_report
                assert activation_object is not None and report_object is not None
                activation = activation_object.to_dict()
                report = report_object.to_dict()
                evidence_trust = activation["trust_domain"]
                evidence_identity = {
                    "activation_record_id": activation["event_id"],
                    "activation_record_digest": activation["event_digest"],
                    "report_id": report["report_id"],
                    "report_digest": report["report_digest"],
                    "trust_domain": evidence_trust,
                }
        support_scope = {
            "yolozu_managed": "public_qualified",
            "site_managed": "site_qualified",
        }.get(evidence_trust, "none")

        pointed = {
            channel
            for channel in _CHANNEL_ORDER
            if (
                pointer := lifecycle.channel_pointers.get(
                    (bundle_record["family_id"], channel)
                )
            )
            is not None
            and pointer["bundle_spec_digest"] == spec_digest
        }
        pointed_channels = _canonical_channels(pointed)
        considered = [
            channel
            for channel in ("Experimental", "Stable")
            if channel in pointed and channel in allowed_channels
        ]
        matching: list[str] = []
        support_by_channel: dict[str, dict[str, Any]] = {}
        step_one_reasons: set[str] = set(registry.selection_trust_reason_codes)
        state = lifecycle.bundle_states.get(spec_digest)
        if step_one_reasons:
            considered = []
        elif state is None:
            step_one_reasons.add("catalog_only")
        elif bundle_record["test_only"]:
            step_one_reasons.add("test_only")
        elif state["bundle_state"] == "revoked":
            step_one_reasons.add("bundle_revoked")
        elif state["bundle_state"] != "enabled":
            step_one_reasons.add("bundle_disabled")
        elif any(
            review["review_state"] != "approved"
            for review in state["artifact_license_reviews"]
        ):
            step_one_reasons.add("license_not_approved")
        elif bundle_record["provenance_class"] == "screened_candidate" and (
            screening["status"] != "current_pass"
            or screening["trust_domain"] != "yolozu_managed"
        ):
            step_one_reasons.add(
                "screening_untrusted"
                if screening["status"] == "untrusted"
                or screening["trust_domain"] != "yolozu_managed"
                else "screening_not_current_pass"
            )
        elif not considered:
            step_one_reasons.add("maturity_disallowed" if pointed else "catalog_only")
        else:
            for channel in considered:
                pointer = lifecycle.channel_pointers[
                    (bundle_record["family_id"], channel)
                ]
                assert pointer is not None
                observed = support_profile_observations.get((spec_digest, channel))
                if observed is None:
                    step_one_reasons.add("support_profile_mismatch")
                    continue
                if not isinstance(observed, SupportProfileEligibilityObservation):
                    raise TypeError("support observations must already be validated")
                try:
                    normalized_support = (
                        validate_support_profile_eligibility_observation(
                            observed.to_dict(),
                            expected_family_id=bundle_record["family_id"],
                            expected_spec_digest=spec_digest,
                            expected_channel=channel,
                            expected_environment_fingerprint=(
                                checked_environment.environment_fingerprint
                                if observed.to_dict()["status"] == "matching_one"
                                else None
                            ),
                            expected_workload_fingerprint=(
                                checked_workload.workload_fingerprint
                                if observed.to_dict()["status"] == "matching_one"
                                else None
                            ),
                            expected_protocol_fingerprint=(
                                protocol
                                if observed.to_dict()["status"] == "matching_one"
                                else None
                            ),
                            expected_advertised_gates_digest=(
                                advertised_gates_digest
                                if observed.to_dict()["status"] == "matching_one"
                                else None
                            ),
                            evidence_trust_domain=evidence_trust,
                            support_scope=support_scope,
                        ).to_dict()
                    )
                except ValueError:
                    step_one_reasons.add("support_profile_mismatch")
                    continue
                support_by_channel[channel] = normalized_support
                if not _support_matches_pointer(normalized_support, pointer):
                    step_one_reasons.add("support_profile_mismatch")
                    continue
                status = normalized_support["status"]
                if status == "matching_one" or (
                    status == "not_required_site"
                    and evidence_trust == "site_managed"
                    and support_scope == "site_qualified"
                ):
                    matching.append(channel)
                else:
                    step_one_reasons.add(_support_reason(status))
            if matching:
                # A failing observation on another pointed channel does not hide
                # the channel that matched this exact spec and request.
                step_one_reasons.difference_update(
                    {
                        "support_profile_mismatch",
                        "support_profile_untrusted",
                        "support_profile_conflict",
                    }
                )

        effective = (
            "Stable"
            if "Stable" in matching
            else "Experimental"
            if "Experimental" in matching
            else "Stable"
            if "Stable" in pointed
            else "Experimental"
            if "Experimental" in pointed
            else "Candidate"
        )
        support_output = support_by_channel.get(effective)
        if support_output is None:
            for channel in ("Stable", "Experimental"):
                if channel in support_by_channel:
                    support_output = support_by_channel[channel]
                    break

        evaluation: dict[str, Any] = {
            "family_id": bundle_record["family_id"],
            "bundle_id": bundle_record["bundle_id"],
            "bundle_version": bundle_record["bundle_version"],
            "spec_digest": spec_digest,
            "artifact_set_digest": bundle.artifact_set_digest,
            "effective_channel": effective,
            "pointed_channels": pointed_channels,
            "matching_channels": _canonical_channels(set(matching)),
            "screening_observation": screening,
            "support_profile_observation": support_output,
            "artifact_state_fingerprint": None,
            "class_mapping": None,
            "evidence": evidence_identity,
            "support_scope": support_scope,
            "rank_state": "excluded",
            "rank_position": None,
            "reason_codes": [],
            "reason_details": [],
            "human_summary": "Excluded because one or more required checks failed.",
            "ranking_trace": [],
        }
        trace: list[dict[str, Any]] = evaluation["ranking_trace"]
        if step_one_reasons:
            reasons = sorted(step_one_reasons, key=lambda item: item.encode("ascii"))
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 1, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 1)

        step_two: set[str] = set()
        if job_record["task"] not in bundle_record["tasks"]:
            step_two.add("task_mismatch")
        if job_record["prompt_mode"] not in bundle_record["prompt_modes"]:
            step_two.add("prompt_mode_mismatch")
        elif job_record["prompt_mode"] == "fixed_classes":
            try:
                evaluation["class_mapping"] = build_fixed_class_mapping(
                    bundle, checked_job.prompt_phrases
                )
            except ValueError:
                step_two.add("class_vocabulary_mismatch")
        if step_two:
            reasons = sorted(step_two, key=lambda item: item.encode("ascii"))
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 2, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 2)

        licenses = set(job_record["spdx_allowlist"])
        if licenses and any(
            artifact["license_expression"] not in licenses
            for artifact in bundle_record["artifacts"]
        ):
            reasons = ["license_not_allowed"]
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 3, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 3)

        if bundle_record["execution_network_required"]:
            reasons = ["network_required"]
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 4, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 4)

        isolation = isolation_by_spec.get(spec_digest)
        if isolation is not None and not isinstance(
            isolation, IsolationCapabilityObservation
        ):
            raise TypeError("isolation capabilities must already be validated")
        step_five = _runtime_and_hardware_reasons(
            bundle=bundle_record,
            job=job_record,
            environment=environment_record,
            isolation=isolation,
        )
        if step_five:
            evaluation["reason_codes"] = step_five
            _trace_fail(trace, 5, step_five)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 5)

        inventory = artifact_inventories.get(spec_digest)
        if inventory is None:
            reasons = ["artifact_member_missing"]
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 6, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        if not isinstance(inventory, LocalArtifactInventory):
            raise TypeError("artifact inventories must already be validated")
        try:
            checked_inventory = validate_local_artifact_inventory(
                inventory.to_dict(), bundle
            )
        except ValueError:
            reasons = ["artifact_member_mismatch"]
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 6, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        evaluation["artifact_state_fingerprint"] = (
            checked_inventory.artifact_state_fingerprint
        )
        _trace_pass(trace, 6)

        if evidence_observation is None:
            step_seven = ["evidence_inactive"]
        elif evidence_observation.status != "active":
            step_seven = [_EVIDENCE_REASON[evidence_observation.status]]
        elif evidence_trust not in {"yolozu_managed", "site_managed"}:
            step_seven = ["evidence_untrusted"]
        else:
            step_seven = []
        if step_seven:
            evaluation["reason_codes"] = step_seven
            _trace_fail(trace, 7, step_seven)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 7)
        assert report is not None and evidence_observation is not None
        activation_record = evidence_observation.activation_record
        assert activation_record is not None
        activation = activation_record.to_dict()

        completed = _utc(report["completed_at"], field="report.completed_at")
        report_valid_until = _utc(report["valid_until"], field="report.valid_until")
        activated = _utc(activation["activated_at"], field="activation.activated_at")
        activation_valid_until = _utc(
            activation["valid_until"], field="activation.valid_until"
        )
        step_eight: set[str] = set()
        if report["status"] != "qualified":
            step_eight.add("evidence_not_qualified")
        if completed > decision_time or activated > decision_time:
            step_eight.add("evidence_future_dated")
        if (
            decision_time >= report_valid_until
            or decision_time >= activation_valid_until
        ):
            step_eight.add("evidence_expired")
        if not completed <= activated < activation_valid_until <= report_valid_until:
            step_eight.add("evidence_inactive")
        if step_eight:
            reasons = sorted(step_eight, key=lambda item: item.encode("ascii"))
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 8, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 8)

        step_nine: set[str] = set()
        if report["bundle_spec_digest"] != spec_digest:
            step_nine.add("bundle_spec_mismatch")
        if report["artifact_set_digest"] != bundle.artifact_set_digest:
            step_nine.add("bundle_spec_mismatch")
        if (
            report["artifact_state_fingerprint"]
            != checked_inventory.artifact_state_fingerprint
        ):
            step_nine.add("artifact_state_mismatch")
        if (
            report["environment_fingerprint"]
            != checked_environment.environment_fingerprint
        ):
            step_nine.add("environment_mismatch")
        if (
            report["qualification_workload_fingerprint"]
            != checked_workload.workload_fingerprint
        ):
            step_nine.add("qualification_workload_mismatch")
        if report["protocol_fingerprint"] != protocol:
            step_nine.add("protocol_mismatch")
        if report["task"] != job_record["task"]:
            step_nine.add("task_mismatch")
        if not _pipeline_matches_report(bundle_record, report):
            step_nine.add("bundle_spec_mismatch")
        quality_requirement = job_record.get("quality_requirement")
        quality = report["quality"]
        if quality_requirement is not None and quality["status"] == "known":
            if (
                quality["metric_id"] != quality_requirement["metric_id"]
                or quality["direction"] != quality_requirement["direction"]
                or quality["threshold_context"] != quality_requirement["threshold"]
            ):
                step_nine.add("requested_metric_unknown")
            if (
                quality["evaluation_dataset_id"]
                != quality_requirement["evaluation_dataset_id"]
                or quality["evaluation_dataset_sha256"]
                != quality_requirement["evaluation_dataset_sha256"]
            ):
                step_nine.add("evaluation_dataset_mismatch")
            if (
                quality["evaluation_vocabulary_id"]
                != quality_requirement["evaluation_vocabulary_id"]
            ):
                step_nine.add("evaluation_vocabulary_mismatch")
            if (
                quality["evaluation_protocol_sha256"]
                != quality_requirement["evaluation_protocol_sha256"]
            ):
                step_nine.add("protocol_mismatch")
        if step_nine:
            reasons = sorted(step_nine, key=lambda item: item.encode("ascii"))
            evaluation["reason_codes"] = reasons
            _trace_fail(trace, 9, reasons)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 9)

        step_ten, rank = _performance_rank_and_reasons(
            report=report,
            job=job_record,
            identity=_identity(bundle),
        )
        if step_ten:
            evaluation["reason_codes"] = step_ten
            _trace_fail(trace, 10, step_ten)
            prepared.append({"evaluation": evaluation, "rank": None})
            continue
        _trace_pass(trace, 10)
        prepared.append({"evaluation": evaluation, "rank": rank})

    eligible = sorted(
        (item for item in prepared if item["rank"] is not None),
        key=lambda item: item["rank"],
    )
    for position, item in enumerate(eligible, start=1):
        evaluation = item["evaluation"]
        evaluation["rank_state"] = (
            "selected" if position == 1 else "eligible_not_selected"
        )
        evaluation["rank_position"] = position
        evaluation["reason_codes"] = []
        evaluation["reason_details"] = []
        evaluation["human_summary"] = (
            "Selected after all required checks passed."
            if position == 1
            else "Eligible, but another candidate ranked first."
        )
        _trace_pass(evaluation["ranking_trace"], 11)

    evaluations = [item["evaluation"] for item in prepared]
    evaluations.sort(
        key=lambda item: (
            item["family_id"].encode("utf-8"),
            item["bundle_id"].encode("utf-8"),
            item["bundle_version"].encode("utf-8"),
            item["spec_digest"].encode("ascii"),
        )
    )
    selected = eligible[0]["evaluation"] if eligible else None
    decision: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": decision_id,
        "status": "selected" if selected is not None else "abstained",
        "decided_at": decided_at,
        "local_job_digest": checked_job.local_job_digest,
        "local_input_digest": local_input,
        "artifact_resolver_state_digest": resolver_state,
        "environment_fingerprint": checked_environment.environment_fingerprint,
        "qualification_workload_fingerprint": checked_workload.workload_fingerprint,
        "protocol_fingerprint": protocol,
        "advertised_gates_digest": advertised_gates_digest,
        "registry_id": registry.registry.to_dict()["registry_id"],
        "registry_digest": registry.registry.registry_digest,
        "registry_trust_domain": registry.registry_trust_domain,
        "lifecycle_projection_digest": registry.lifecycle.head_digest,
        "lifecycle_trust_domain": registry.lifecycle_trust_domain,
        "ranking_policy": job_record["ranking_policy"],
        "prompt_mode": job_record["prompt_mode"],
        "registry_bundle_count": len(registry.bundles),
        "selected_bundle": None,
        "selected_evidence": None,
        "selected_artifact_state_fingerprint": None,
        "selected_class_mapping": None,
        "support_scope": "none",
        "reason_codes": ["no_eligible_candidate"] if selected is None else [],
        "human_summary": (
            "No eligible qualified bundle matched all required checks."
            if selected is None
            else "Selected one qualified bundle after all required checks passed."
        ),
        "candidate_evaluations": evaluations,
        "selection_trace": [
            {
                "rank_position": item["evaluation"]["rank_position"],
                "spec_digest": item["evaluation"]["spec_digest"],
            }
            for item in eligible
        ],
        "decision_digest": "0" * 64,
    }
    if selected is not None:
        decision["selected_bundle"] = {
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
        decision["selected_evidence"] = selected["evidence"]
        decision["selected_artifact_state_fingerprint"] = selected[
            "artifact_state_fingerprint"
        ]
        decision["selected_class_mapping"] = selected["class_mapping"]
        decision["support_scope"] = selected["support_scope"]
    decision["decision_digest"] = canonical_sha256_v1(
        decision, own_digest_field="decision_digest"
    )
    return validate_selection_decision(
        decision,
        expected_registry=registry.registry,
        as_of=current,
    )
