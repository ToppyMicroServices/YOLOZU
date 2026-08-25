"""Read-only orchestration for adaptive image-pipeline recommendation."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .artifact_resolver import ArtifactResolver, PinnedVerifiedArtifactSet
from .bundle_registry import LoadedAlgorithmBundleRegistry, load_algorithm_bundle_registry
from .bundles import (
    AlgorithmBundleSpec,
    SupportProfileProjection,
    project_support_profiles,
)
from .canonical import canonical_sha256_v1
from .contracts import (
    EnvironmentProfile,
    ImageJobSpec,
    QualificationWorkloadProfile,
    build_qualification_workload_profile,
    validate_image_job_spec,
)
from .control_records import (
    MAX_CONTROL_STREAM_BYTES,
    load_bounded_json_bytes,
    load_bounded_jsonl_bytes,
)
from .environment import build_environment_profile
from .evidence import (
    EvidenceActivationRecord,
    LocalArtifactInventory,
    QualificationReport,
    compute_artifact_state_fingerprint,
    compute_evidence_selection_key,
    load_evidence_activation_jsonl_bytes,
    project_evidence_activations,
    validate_evidence_activation_record,
    validate_local_artifact_inventory,
    validate_qualification_report,
)
from .inventory import build_decoded_input_inventory
from .isolation import _code_owned_isolated_services
from .qualification import (
    QUALIFICATION_PROTOCOL_FINGERPRINT,
    qualification_report_has_code_owned_issuer,
)
from .selection import (
    SupportProfileEligibilityObservation,
    validate_support_profile_eligibility_observation,
)
from .screening import (
    CandidateScreeningProjection,
    MAX_SCREENING_STREAM_BYTES,
    build_screening_eligibility_observation,
    load_candidate_screening_jsonl_bytes,
)
from .selector import (
    EvidenceEligibilityObservation,
    IsolationCapabilityObservation,
    compute_advertised_gates_digest,
    evidence_eligibility_from_projection,
    select_qualified_pipeline,
)

__all__ = [
    "RecommendationError",
    "recommend_image_pipeline",
]


_MAX_SUPPORT_PROFILE_BYTES = 64 * 1024 * 1024
_MAX_QUALIFICATION_REPORT_BYTES = 64 * 1024 * 1024
_MAX_CHECKSUM_MANIFEST_BYTES = 4 * 1024 * 1024
_ZERO_DIGEST = "0" * 64


class RecommendationError(ValueError):
    """A safe public recommendation failure with one stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True)
class _EvidenceLoad:
    observations: dict[str, EvidenceEligibilityObservation]
    source_kind: str
    trust_domain: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_decision_time",
            "decision time must be an exact RFC3339 UTC second",
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise RecommendationError(
            "invalid_decision_time",
            "decision time must be an exact RFC3339 UTC second",
        )
    return parsed


def _workspace_root(path: str | Path) -> Path:
    lexical = Path(os.path.abspath(Path(path)))
    if lexical.is_symlink():
        raise RecommendationError("unsafe_workspace", "workspace root is unsafe")
    try:
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise RecommendationError(
            "unsafe_workspace", "workspace root is unavailable"
        ) from exc
    if not root.is_dir():
        raise RecommendationError(
            "unsafe_workspace", "workspace root must be a directory"
        )
    return root


def _confined_directory(
    value: str | Path,
    *,
    workspace: Path,
    label: str,
) -> Path:
    lexical = _confined_lexical_path(value, workspace=workspace, label=label)
    relative = lexical.relative_to(workspace)
    current = workspace
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RecommendationError(f"unsafe_{label}", f"{label} contains a symlink")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise RecommendationError(f"unsafe_{label}", f"{label} is unavailable") from exc
    if not resolved.is_dir():
        raise RecommendationError(f"unsafe_{label}", f"{label} must be a directory")
    return resolved


def _confined_lexical_path(
    value: str | Path,
    *,
    workspace: Path,
    label: str,
) -> Path:
    candidate = Path(value)
    if ".." in candidate.parts or str(candidate).startswith("~"):
        raise RecommendationError(f"unsafe_{label}", f"{label} is outside the workspace")
    lexical = Path(os.path.abspath(candidate if candidate.is_absolute() else workspace / candidate))
    try:
        lexical.relative_to(workspace)
    except ValueError as exc:
        raise RecommendationError(f"unsafe_{label}", f"{label} is outside the workspace") from exc
    return lexical


def _read_regular_at(
    root: Path,
    parts: tuple[str, ...],
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RecommendationError("unsupported_platform", "safe local file access is unavailable")
    parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = -1
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            previous = parent_fd
            parent_fd = child
            os.close(previous)
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not a singly linked regular file")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its byte cap")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1_048_576, maximum_bytes + 1 - observed),
            )
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte cap")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        )
        if before_identity != after_identity or observed != after.st_size:
            raise ValueError(f"{label} changed while reading")
        return b"".join(chunks)
    except (OSError, ValueError) as exc:
        raise RecommendationError("invalid_evidence", f"{label} is invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _packaged_bytes(parts: tuple[str, ...], *, maximum_bytes: int, label: str) -> bytes:
    item = resources.files("yolozu.data")
    for component in parts:
        item = item.joinpath(component)
    try:
        payload = item.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise RecommendationError("invalid_evidence", f"packaged {label} is unavailable") from exc
    if len(payload) > maximum_bytes:
        raise RecommendationError("invalid_evidence", f"packaged {label} exceeds its byte cap")
    return payload


def _load_support_profiles() -> SupportProfileProjection:
    payload = _packaged_bytes(
        ("adaptive_routing", "support_profiles.jsonl"),
        maximum_bytes=_MAX_SUPPORT_PROFILE_BYTES,
        label="support-profile stream",
    )
    try:
        records = load_bounded_jsonl_bytes(
            payload,
            label="support-profile stream",
            max_records=128,
        )
        return project_support_profiles(
            records,
            source_trust_domain="yolozu_managed",
        )
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_support_profiles",
            "support-profile stream is invalid",
        ) from exc


def _validate_report_without_freshness(payload: Mapping[str, Any]) -> QualificationReport:
    completed_at = payload.get("completed_at")
    if not isinstance(completed_at, str):
        raise ValueError("qualification report completion time is invalid")
    return validate_qualification_report(payload, as_of=completed_at)


def _site_checksum_is_exact(report_bytes: bytes, checksum_bytes: bytes) -> bool:
    try:
        manifest = load_bounded_json_bytes(
            checksum_bytes,
            label="qualification checksums manifest",
        )
    except ValueError:
        return False
    expected = {
        "path": "qualification_report.json",
        "size_bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    return manifest == {
        "schema_version": 1,
        "files": [expected],
        "expected_paths": ["qualification_report.json"],
        "file_count": 1,
        "total_bytes": len(report_bytes),
    }


def _projection_status(message: str) -> str | None:
    if "future-dated" in message:
        return "future_dated"
    if "expired" in message:
        return "expired"
    if "only a qualified report" in message:
        return "not_qualified"
    if "dangling supersession" in message:
        return "superseded"
    conflict_markers = (
        "multiple active reports",
        "predecessor gap or fork",
        "sequence gap or duplicate",
        "duplicate event_id",
    )
    if any(marker in message for marker in conflict_markers):
        return "conflict"
    return None


def _load_evidence(
    *,
    evidence_root: Path | None,
    decided_at: str,
) -> _EvidenceLoad:
    if evidence_root is None:
        raw_stream = _packaged_bytes(
            ("adaptive_routing", "evidence_activation.jsonl"),
            maximum_bytes=MAX_CONTROL_STREAM_BYTES,
            label="evidence activation stream",
        )
        source_kind = "packaged_ssot"
        source_trust = "yolozu_managed"
    else:
        raw_stream = _read_regular_at(
            evidence_root,
            ("evidence_activation.jsonl",),
            maximum_bytes=MAX_CONTROL_STREAM_BYTES,
            label="evidence activation stream",
        )
        source_kind = "workspace_evidence"
        source_trust = "operator_asserted"

    try:
        raw_records = load_evidence_activation_jsonl_bytes(raw_stream)
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_evidence",
            "evidence activation stream is invalid",
        ) from exc
    if not raw_records:
        return _EvidenceLoad({}, source_kind, source_trust)

    if evidence_root is not None:
        claims = {
            str(record.get("trust_domain"))
            for record in raw_records
            if isinstance(record, Mapping)
        }
        if claims == {"site_managed"}:
            source_trust = "site_managed"
        elif claims == {"operator_asserted"}:
            source_trust = "operator_asserted"
        elif claims == {"unknown"}:
            source_trust = "unknown"
        else:
            raise RecommendationError(
                "invalid_evidence",
                "workspace evidence has mixed or unsupported trust claims",
            )

    events: list[EvidenceActivationRecord] = []
    try:
        for record in raw_records:
            events.append(
                validate_evidence_activation_record(
                    record,
                    source_trust_domain=source_trust,
                )
            )
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_evidence",
            "evidence activation authority or record is invalid",
        ) from exc

    report_identities = sorted(
        {
            (
                event.to_dict()["report_id"],
                event.to_dict()["report_digest"],
            )
            for event in events
        },
        key=lambda item: (item[0].encode("ascii"), item[1].encode("ascii")),
    )
    reports: list[QualificationReport] = []
    for report_id, report_digest in report_identities:
        if evidence_root is None:
            report_bytes = _packaged_bytes(
                (
                    "adaptive_routing",
                    "qualification_reports",
                    report_id,
                    "qualification_report.json",
                ),
                maximum_bytes=_MAX_QUALIFICATION_REPORT_BYTES,
                label="qualification report",
            )
            checksum_bytes = b""
        else:
            report_bytes = _read_regular_at(
                evidence_root,
                (
                    "qualification_reports",
                    report_id,
                    "qualification_report.json",
                ),
                maximum_bytes=_MAX_QUALIFICATION_REPORT_BYTES,
                label="qualification report",
            )
            checksum_bytes = _read_regular_at(
                evidence_root,
                ("qualification_reports", report_id, "checksums.json"),
                maximum_bytes=_MAX_CHECKSUM_MANIFEST_BYTES,
                label="qualification checksums manifest",
            )
        try:
            payload = load_bounded_json_bytes(
                report_bytes,
                label="QualificationReport",
            )
            if not isinstance(payload, Mapping):
                raise ValueError("qualification report must be an object")
            report = _validate_report_without_freshness(payload)
        except (TypeError, ValueError) as exc:
            raise RecommendationError(
                "invalid_evidence",
                "qualification report is invalid",
            ) from exc
        if report.report_id != report_id or report.report_digest != report_digest:
            raise RecommendationError(
                "invalid_evidence",
                "activation/report identity does not match",
            )
        if source_trust == "site_managed" and (
            not _site_checksum_is_exact(report_bytes, checksum_bytes)
            or not qualification_report_has_code_owned_issuer(report)
        ):
            raise RecommendationError(
                "invalid_evidence",
                "site evidence is not an exact code-owned qualifier package",
            )
        reports.append(report)

    try:
        projection = project_evidence_activations(
            events,
            reports,
            source_trust_domain=source_trust,
            as_of=decided_at,
        )
    except ValueError as exc:
        status = _projection_status(str(exc))
        if status is None:
            raise RecommendationError(
                "invalid_evidence",
                "evidence projection is corrupt or incomplete",
            ) from exc
        keys = sorted(
            {event.to_dict()["selection_key"] for event in events},
            key=lambda item: item.encode("ascii"),
        )
        return _EvidenceLoad(
            {
                key: EvidenceEligibilityObservation(key, status)
                for key in keys
            },
            source_kind,
            source_trust,
        )
    return _EvidenceLoad(
        evidence_eligibility_from_projection(projection),
        source_kind,
        source_trust,
    )


def _load_screening(
    *,
    screening_root: Path | None,
) -> tuple[CandidateScreeningProjection, str]:
    if screening_root is None:
        payload = _packaged_bytes(
            ("adaptive_routing", "candidate_screening.jsonl"),
            maximum_bytes=MAX_SCREENING_STREAM_BYTES,
            label="candidate-screening stream",
        )
        trust = "yolozu_managed"
        source = "packaged_ssot"
    else:
        try:
            payload = _read_regular_at(
                screening_root,
                ("candidate_screening.jsonl",),
                maximum_bytes=MAX_SCREENING_STREAM_BYTES,
                label="candidate-screening stream",
            )
        except RecommendationError as exc:
            if exc.code == "unsupported_platform":
                raise
            raise RecommendationError(
                "invalid_screening",
                "candidate-screening stream is invalid",
            ) from exc
        trust = "operator_asserted"
        source = "workspace_screening"
    try:
        return (
            load_candidate_screening_jsonl_bytes(
                payload,
                source_trust_domain=trust,
            ),
            source,
        )
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_screening",
            "candidate-screening stream is invalid",
        ) from exc


def _evidence_trust_for_bundle(
    *,
    bundle: AlgorithmBundleSpec,
    environment: EnvironmentProfile,
    workload: QualificationWorkloadProfile,
    evidence: Mapping[str, EvidenceEligibilityObservation],
) -> tuple[str, str]:
    key = compute_evidence_selection_key(
        bundle_spec_digest=bundle.spec_digest,
        artifact_set_digest=bundle.artifact_set_digest,
        environment_fingerprint=environment.environment_fingerprint,
        qualification_workload_fingerprint=workload.workload_fingerprint,
        protocol_fingerprint=QUALIFICATION_PROTOCOL_FINGERPRINT,
    )
    observation = evidence.get(key)
    if observation is None or observation.status != "active":
        return "unknown", "none"
    activation = observation.activation_record
    assert activation is not None
    trust = str(activation.to_dict()["trust_domain"])
    scope = {
        "yolozu_managed": "public_qualified",
        "site_managed": "site_qualified",
    }.get(trust, "none")
    return trust, scope


def _support_observations(
    *,
    registry: LoadedAlgorithmBundleRegistry,
    profiles: SupportProfileProjection,
    job: ImageJobSpec,
    environment: EnvironmentProfile,
    workload: QualificationWorkloadProfile,
    evidence: Mapping[str, EvidenceEligibilityObservation],
) -> dict[tuple[str, str], SupportProfileEligibilityObservation]:
    observations: dict[tuple[str, str], SupportProfileEligibilityObservation] = {}
    advertised_digest = compute_advertised_gates_digest(job)
    for bundle in registry.bundles:
        bundle_record = bundle.to_dict()
        evidence_trust, support_scope = _evidence_trust_for_bundle(
            bundle=bundle,
            environment=environment,
            workload=workload,
            evidence=evidence,
        )
        for channel in ("Experimental", "Stable"):
            pointer = registry.lifecycle.channel_pointers.get(
                (bundle_record["family_id"], channel)
            )
            if pointer is None or pointer["bundle_spec_digest"] != bundle.spec_digest:
                continue
            if any(
                pointer.get(field) is None
                for field in (
                    "profile_set_record_id",
                    "profile_set_record_digest",
                )
            ):
                continue
            matching: list[dict[str, Any]] = []
            for reference in pointer["profiles"]:
                profile = profiles.definitions.get(reference["profile_id"])
                if profile is None or profile.profile_digest != reference["profile_digest"]:
                    continue
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
            if evidence_trust == "site_managed" and support_scope == "site_qualified":
                status = "not_required_site"
            elif len(matching) == 1:
                status = "matching_one"
            elif len(matching) > 1:
                status = "conflict"
            else:
                status = "no_match"
            matched = matching[0] if status == "matching_one" else None
            value = {
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
                "support_profile_index_head_digest": pointer[
                    "support_profile_index_head"
                ],
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
                "advertised_gates_digest": (
                    None if matched is None else advertised_digest
                ),
                "trust_domain": (
                    "yolozu_managed" if matched is not None else "unknown"
                ),
                "observation_digest": _ZERO_DIGEST,
            }
            value["observation_digest"] = canonical_sha256_v1(
                value,
                own_digest_field="observation_digest",
            )
            observations[(bundle.spec_digest, channel)] = (
                validate_support_profile_eligibility_observation(
                    value,
                    evidence_trust_domain=evidence_trust,
                    support_scope=support_scope,
                )
            )
    return observations


def _artifact_inventory(
    *,
    pinned: PinnedVerifiedArtifactSet,
    bundle: AlgorithmBundleSpec,
    verified_at: str,
) -> LocalArtifactInventory:
    by_id = {
        item["artifact_id"]: item for item in pinned.iter_local_observations()
    }
    observations = []
    for artifact in bundle.to_dict()["artifacts"]:
        observed = by_id[artifact["artifact_id"]]
        observations.append(
            {
                "artifact_id": artifact["artifact_id"],
                "role": artifact["role"],
                "order": artifact["order"],
                "expected_size_bytes": artifact["expected_size_bytes"],
                "expected_sha256": artifact["sha256"],
                "presence_status": "present",
                "path_type_status": "regular_file",
                "read_status": "readable",
                "observed_size_bytes": observed["size_bytes"],
                "observed_sha256": observed["sha256"],
                "verified_at": verified_at,
                "error_status": "none",
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "inventory_id": f"recommend-{bundle.spec_digest[:16]}",
        "bundle_spec_digest": bundle.spec_digest,
        "artifact_set_digest": bundle.artifact_set_digest,
        "observations": observations,
        "artifact_state_fingerprint": _ZERO_DIGEST,
        "inventory_digest": _ZERO_DIGEST,
    }
    value["artifact_state_fingerprint"] = compute_artifact_state_fingerprint(value)
    value["inventory_digest"] = canonical_sha256_v1(
        value,
        own_digest_field="inventory_digest",
    )
    return validate_local_artifact_inventory(value, bundle)


def _isolation_observations(
    registry: LoadedAlgorithmBundleRegistry,
) -> dict[str, IsolationCapabilityObservation]:
    observations: dict[str, IsolationCapabilityObservation] = {}
    isolated_services = _code_owned_isolated_services()
    for bundle in registry.bundles:
        record = bundle.to_dict()
        if record["execution_binding"]["status"] != "bound":
            continue
        if record["execution_trust_class"] != "third_party_isolated":
            continue
        service = isolated_services.get(record["runner_id"])
        if service is None:
            observations[bundle.spec_digest] = IsolationCapabilityObservation(
                "unsupported"
            )
            continue
        try:
            capability = service.capability
            if capability.runner_id != record["runner_id"]:
                raise ValueError("runner mismatch")
            if capability.status != "available":
                raise ValueError("capability unavailable")
            observations[bundle.spec_digest] = IsolationCapabilityObservation(
                "supported",
                backend_id=capability.backend_id,
                backend_version=capability.backend_version,
                isolation_policy_digest=capability.policy_digest,
                image_present=capability.image_present,
            )
        except (AttributeError, TypeError, ValueError):
            observations[bundle.spec_digest] = IsolationCapabilityObservation(
                "unsupported"
            )
    return observations


def recommend_image_pipeline(
    job_spec: Mapping[str, Any],
    input_path: str,
    *,
    workspace_root: str | Path = ".",
    registry_root: str | None = None,
    screening_root: str | None = None,
    evidence_root: str | None = None,
    artifact_root: str | None = None,
    decided_at: str | None = None,
) -> dict[str, Any]:
    """Return one selected or abstained decision without execution or writes."""

    decision_time = decided_at or _utc_now()
    _utc(decision_time)
    workspace = _workspace_root(workspace_root)
    artifact_root_lexical: Path | None = None
    if artifact_root is not None:
        artifact_root_lexical = _confined_lexical_path(
            artifact_root,
            workspace=workspace,
            label="artifact_root",
        )
    try:
        job = validate_image_job_spec(job_spec)
    except (TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_job_spec",
            "job_spec does not satisfy the ImageJobSpec interface contract",
        ) from exc
    try:
        input_inventory = build_decoded_input_inventory(
            input_path,
            input_mode=job.to_dict()["input_mode"],
            workspace_root=workspace,
            max_images=job.to_dict()["max_images"],
        )
        workload = build_qualification_workload_profile(job, input_inventory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_input",
            "input_path is unsafe, unsupported, or outside the declared bounds",
        ) from exc

    try:
        environment = build_environment_profile(collected_at=decision_time)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RecommendationError(
            "environment_probe_failed",
            "the bounded local environment profile could not be built",
        ) from exc

    support_profiles = _load_support_profiles()
    custom_registry: Path | None = None
    if registry_root is not None:
        custom_registry = _confined_directory(
            registry_root,
            workspace=workspace,
            label="registry_root",
        )
    try:
        registry = load_algorithm_bundle_registry(
            workspace_root=workspace if custom_registry is not None else None,
            custom_registry_root=custom_registry,
            support_profiles=(
                support_profiles if custom_registry is None else None
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecommendationError(
            "invalid_registry",
            "bundle registry or lifecycle stream is invalid",
        ) from exc

    custom_evidence: Path | None = None
    if evidence_root is not None:
        custom_evidence = _confined_directory(
            evidence_root,
            workspace=workspace,
            label="evidence_root",
        )
    evidence_load = _load_evidence(
        evidence_root=custom_evidence,
        decided_at=decision_time,
    )

    custom_screening: Path | None = None
    if screening_root is not None:
        custom_screening = _confined_directory(
            screening_root,
            workspace=workspace,
            label="screening_root",
        )
    screening_projection, screening_source = _load_screening(
        screening_root=custom_screening,
    )

    screening = {
        bundle.spec_digest: build_screening_eligibility_observation(
            bundle,
            screening_projection,
        )
        for bundle in registry.bundles
    }
    support = _support_observations(
        registry=registry,
        profiles=support_profiles,
        job=job,
        environment=environment,
        workload=workload,
        evidence=evidence_load.observations,
    )
    isolation = _isolation_observations(registry)
    provisional_resolver_digest = canonical_sha256_v1(
        {
            "resolver_version": "artifact-resolver-v1",
            "status": "not_checked",
            "candidate_spec_digests": [],
        }
    )
    decision_seed = canonical_sha256_v1(
        {
            "decided_at": decision_time,
            "local_job_digest": job.local_job_digest,
            "local_input_digest": input_inventory.local_input_digest,
            "environment_fingerprint": environment.environment_fingerprint,
            "qualification_workload_fingerprint": workload.workload_fingerprint,
            "registry_digest": registry.registry.registry_digest,
            "lifecycle_projection_digest": registry.lifecycle.head_digest,
        }
    )
    decision_id = f"recommend-{decision_seed[:24]}"
    provisional = select_qualified_pipeline(
        decision_id=decision_id,
        decided_at=decision_time,
        job=job,
        local_input_digest=input_inventory.local_input_digest,
        artifact_resolver_state_digest=provisional_resolver_digest,
        environment=environment,
        workload=workload,
        protocol_fingerprint=QUALIFICATION_PROTOCOL_FINGERPRINT,
        registry=registry,
        screening_observations=screening,
        support_profile_observations=support,
        artifact_inventories={},
        evidence_observations=evidence_load.observations,
        isolation_capabilities=isolation,
        prefer_evidence_before_artifact_io=True,
        as_of=decision_time,
    )
    bundle_by_digest = registry.by_spec_digest()
    survivors = [
        item["spec_digest"]
        for item in provisional.to_dict()["candidate_evaluations"]
        if item["reason_codes"] == ["artifact_member_missing"]
    ]
    artifact_status = {
        item["spec_digest"]: "not_checked_due_to_prior_filter"
        for item in provisional.to_dict()["candidate_evaluations"]
        if item["spec_digest"] not in survivors
    }
    inventories: dict[str, LocalArtifactInventory] = {}
    resolver_states: list[dict[str, str]] = []
    if survivors:
        explicit_artifact_root = (
            None
            if artifact_root_lexical is None
            else _confined_directory(
                artifact_root_lexical,
                workspace=workspace,
                label="artifact_root",
            )
        )
        try:
            resolver_context = ArtifactResolver(
                workspace=workspace,
                artifact_root=explicit_artifact_root,
            )
        except (OSError, RuntimeError, ValueError):
            resolver_context = None
        if resolver_context is not None:
            with resolver_context as resolver:
                for spec_digest in sorted(survivors, key=lambda item: item.encode("ascii")):
                    bundle = bundle_by_digest[spec_digest]
                    try:
                        with resolver.pin(bundle) as pinned:
                            inventories[spec_digest] = _artifact_inventory(
                                pinned=pinned,
                                bundle=bundle,
                                verified_at=decision_time,
                            )
                            resolver_states.append(
                                {
                                    "spec_digest": spec_digest,
                                    "resolver_state_digest": pinned.artifact_resolver_state_digest,
                                }
                            )
                        artifact_status[spec_digest] = "verified"
                    except (OSError, RuntimeError, ValueError):
                        artifact_status[spec_digest] = "verification_failed"
        else:
            for spec_digest in survivors:
                artifact_status[spec_digest] = "verification_failed"

    resolver_digest = canonical_sha256_v1(
        {
            "resolver_version": "artifact-resolver-v1",
            "candidate_states": resolver_states,
            "verified_spec_digests": sorted(
                inventories,
                key=lambda item: item.encode("ascii"),
            ),
        }
    )
    decision = select_qualified_pipeline(
        decision_id=decision_id,
        decided_at=decision_time,
        job=job,
        local_input_digest=input_inventory.local_input_digest,
        artifact_resolver_state_digest=resolver_digest,
        environment=environment,
        workload=workload,
        protocol_fingerprint=QUALIFICATION_PROTOCOL_FINGERPRINT,
        registry=registry,
        screening_observations=screening,
        support_profile_observations=support,
        artifact_inventories=inventories,
        evidence_observations=evidence_load.observations,
        isolation_capabilities=isolation,
        prefer_evidence_before_artifact_io=True,
        as_of=decision_time,
    )
    decision_record = decision.to_dict()
    resolver_state_by_spec = {
        item["spec_digest"]: item["resolver_state_digest"]
        for item in resolver_states
    }
    selected_bundle = decision_record["selected_bundle"]
    ordered_status = [
        {
            "spec_digest": item["spec_digest"],
            "status": artifact_status[item["spec_digest"]],
        }
        for item in decision_record["candidate_evaluations"]
    ]
    return {
        "schema_version": 1,
        "ok": True,
        "tool": "recommend_image_pipeline",
        "summary": (
            "selected one qualified image pipeline"
            if decision_record["status"] == "selected"
            else "abstained because no qualified image pipeline matched"
        ),
        "exit_code": 0,
        "maturity": "experimental",
        "availability": "mcp_live",
        "decision": decision_record,
        "recommendation_metadata": {
            "read_only": True,
            "model_execution_performed": False,
            "network_used": False,
            "writes_performed": False,
            "registry_source": registry.source_kind,
            "screening_source": screening_source,
            "screening_trust_domain": screening_projection.source_trust_domain,
            "evidence_source": evidence_load.source_kind,
            "evidence_trust_domain": evidence_load.trust_domain,
            "input_count": input_inventory.input_count,
            "selected_artifact_resolver_state_digest": (
                None
                if selected_bundle is None
                else resolver_state_by_spec.get(selected_bundle["spec_digest"])
            ),
            "artifact_observations": ordered_status,
            "privacy": (
                "No filenames, absolute paths, per-file hashes, or raw probe output are returned."
            ),
        },
    }
