from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import _bundle_payload, _registry_payload
from tests.test_adaptive_evidence_contracts import (
    _activation,
    _redigest_report,
    _report_payload,
)
from yolozu.adaptive.bundle_registry import LoadedAlgorithmBundleRegistry
from yolozu.adaptive.bundles import (
    BundleLifecycleProjection,
    build_fixed_class_mapping,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
)
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.contracts import (
    EnvironmentProfile,
    ImageJobSpec,
    QualificationWorkloadProfile,
    build_qualification_workload_profile,
    compute_environment_fingerprint,
    validate_environment_profile,
    validate_image_job_spec,
)
from yolozu.adaptive.evidence import (
    compute_artifact_state_fingerprint,
    compute_evidence_selection_key,
    validate_evidence_activation_record,
    validate_local_artifact_inventory,
    validate_qualification_report,
)
from yolozu.adaptive.inventory import DecodedInputInventory, DecodedInputObservation
from yolozu.adaptive.selection import (
    ScreeningEligibilityObservation,
    SupportProfileEligibilityObservation,
    validate_screening_eligibility_observation,
    validate_support_profile_eligibility_observation,
)
from yolozu.adaptive.selector import (
    EvidenceEligibilityObservation,
    IsolationCapabilityObservation,
    compute_advertised_gates_digest,
    select_qualified_pipeline,
)


AS_OF = "2026-08-25T12:00:00Z"
DECIDED_AT = "2026-08-25T06:30:00Z"
PROTOCOL = "a" * 64


def _job(**updates: object) -> ImageJobSpec:
    payload: dict[str, object] = {
        "schema_version": 1,
        "task": "object_detection",
        "prompt_mode": "fixed_classes",
        "fixed_classes": ["cat"],
        "input_mode": "single_image",
        "execution_mode": "batch",
        "batch_size": 1,
        "concurrency": 1,
        "max_images": 1,
        "max_results_per_image": 100,
        "job_timeout_seconds": 60,
        "ranking_policy": "latency_first",
        "allowed_maturities": ["Experimental", "Stable"],
        "network_policy": "deny",
        "compute_policy": "auto",
        "provider_allowlist": [],
        "precision_allowlist": [],
        "spdx_allowlist": [],
        "max_cold_start_ms": "500",
        "max_p95_latency_ms": "50",
        "min_repeat_throughput_fps": "1",
    }
    for key, value in updates.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return validate_image_job_spec(payload)


def _environment() -> EnvironmentProfile:
    record = {
        "schema_version": 1,
        "collector_id": "yolozu_environment",
        "collector_version": "1",
        "collected_at": "2026-08-25T00:00:00Z",
        "os": {
            "probe_status": "present",
            "name": "Linux",
            "version": "6.8.0",
            "architecture": "x86_64",
        },
        "cpu": {
            "probe_status": "present",
            "model": "Fixture CPU",
            "logical_cores": {"probe_status": "present", "value": 8},
            "physical_cores": {"probe_status": "present", "value": 4},
        },
        "total_memory": {"probe_status": "present", "value_bytes": 16 * 1024**3},
        "accelerators": [
            {"accelerator_id": "cuda", "probe_status": "absent"},
            {"accelerator_id": "mps", "probe_status": "absent"},
        ],
        "runtimes": [
            {
                "runtime_id": "onnxruntime",
                "probe_status": "present",
                "version": "1.23.0",
                "provider_ids": ["cpu"],
            }
        ],
        "power_performance_mode": {"probe_status": "unsupported"},
        "probe_issues": [],
    }
    record["environment_fingerprint"] = compute_environment_fingerprint(record)
    return validate_environment_profile(record)


def _workload(job: ImageJobSpec) -> QualificationWorkloadProfile:
    inventory = DecodedInputInventory(
        input_mode="single_image",
        input_count=1,
        input_order="single_image_v1",
        inputs=(
            DecodedInputObservation(index=0, width=64, height=32, color_mode="RGB"),
        ),
        decoder_id="pillow",
        decoder_version="12.3.0",
        source_total_bytes=5,
        local_input_digest="1" * 64,
    )
    return build_qualification_workload_profile(job, inventory)


def _screening(bundle: dict) -> ScreeningEligibilityObservation:
    if bundle["provenance_class"] == "existing_code_owned":
        record = {
            "schema_version": 1,
            "provider_id": "no_screening_required",
            "provider_version": "1",
            "provenance_class": "existing_code_owned",
            "screening_stream_key": None,
            "source_revision": None,
            "status": "not_applicable",
            "current_record_id": None,
            "current_record_digest": None,
            "projection_head_digest": None,
            "trust_domain": "unknown",
            "observation_digest": "0" * 64,
        }
    else:
        binding = bundle["screening_binding"]
        record = {
            "schema_version": 1,
            "provider_id": "candidate-screening-projection",
            "provider_version": "1",
            "provenance_class": "screened_candidate",
            "screening_stream_key": binding["stream_key"],
            "source_revision": binding["source_revision"],
            "status": "current_pass",
            "current_record_id": binding["pass_record_id"],
            "current_record_digest": binding["pass_record_digest"],
            "projection_head_digest": "9" * 64,
            "trust_domain": "yolozu_managed",
            "observation_digest": "0" * 64,
        }
    record["observation_digest"] = canonical_sha256_v1(
        record, own_digest_field="observation_digest"
    )
    return validate_screening_eligibility_observation(record, bundle=bundle)


def _inventory(bundle: dict):
    artifact = bundle["artifacts"][0]
    record = {
        "schema_version": 1,
        "inventory_id": f"inventory-{bundle['spec_digest'][:8]}",
        "bundle_spec_digest": bundle["spec_digest"],
        "artifact_set_digest": bundle["artifact_set_digest"],
        "observations": [
            {
                "artifact_id": artifact["artifact_id"],
                "role": artifact["role"],
                "order": artifact["order"],
                "expected_size_bytes": artifact["expected_size_bytes"],
                "expected_sha256": artifact["sha256"],
                "presence_status": "present",
                "path_type_status": "regular_file",
                "read_status": "readable",
                "observed_size_bytes": artifact["expected_size_bytes"],
                "observed_sha256": artifact["sha256"],
                "verified_at": "2026-08-25T01:00:00Z",
                "error_status": "none",
            }
        ],
        "artifact_state_fingerprint": "0" * 64,
        "inventory_digest": "0" * 64,
    }
    record["artifact_state_fingerprint"] = compute_artifact_state_fingerprint(record)
    record["inventory_digest"] = canonical_sha256_v1(
        record, own_digest_field="inventory_digest"
    )
    return validate_local_artifact_inventory(
        record, validate_algorithm_bundle_spec(bundle)
    )


def _pipeline(bundle: dict) -> dict:
    return {
        "decoder": {
            "id": bundle["decoder"]["id"],
            "version": bundle["decoder"]["version"],
            "source_digest": bundle["decoder"]["digest"],
        },
        "model_input": {
            "id": "bundle_model_input_shapes",
            "version": "1",
            "source_digest": canonical_sha256_v1(bundle["model_input_shapes"]),
        },
        "preprocess": {
            "id": bundle["preprocess"]["id"],
            "version": bundle["preprocess"]["version"],
            "source_digest": bundle["preprocess"]["digest"],
        },
        "postprocess": {
            "id": bundle["postprocess"]["id"],
            "version": bundle["postprocess"]["version"],
            "source_digest": bundle["postprocess"]["digest"],
        },
    }


def _support(
    *,
    bundle: dict,
    channel: str,
    pointer: dict,
    environment: EnvironmentProfile,
    workload: QualificationWorkloadProfile,
    job: ImageJobSpec,
    status: str = "matching_one",
    evidence_trust_domain: str | None = None,
    support_scope: str | None = None,
) -> SupportProfileEligibilityObservation:
    matching = status == "matching_one"
    record = {
        "schema_version": 1,
        "provider_id": "support-profile-projection",
        "provider_version": "1",
        "family_id": bundle["family_id"],
        "bundle_spec_digest": bundle["spec_digest"],
        "channel": channel,
        "lifecycle_assignment_id": f"assignment-{channel.lower()}",
        "lifecycle_assignment_digest": pointer["lifecycle_event_digest"],
        "support_profile_index_head_digest": pointer["support_profile_index_head"],
        "profile_set_record_id": pointer["profile_set_record_id"],
        "profile_set_record_digest": pointer["profile_set_record_digest"],
        "profile_set_digest": pointer["profile_set_digest"],
        "status": status,
        "profile_id": "fixture-profile" if matching else None,
        "profile_digest": "5" * 64 if matching else None,
        "environment_fingerprint": (
            environment.environment_fingerprint if matching else None
        ),
        "qualification_workload_fingerprint": (
            workload.workload_fingerprint if matching else None
        ),
        "protocol_fingerprint": PROTOCOL if matching else None,
        "advertised_gates_digest": (
            compute_advertised_gates_digest(job) if matching else None
        ),
        "trust_domain": "yolozu_managed" if matching else "unknown",
        "observation_digest": "0" * 64,
    }
    record["observation_digest"] = canonical_sha256_v1(
        record, own_digest_field="observation_digest"
    )
    return validate_support_profile_eligibility_observation(
        record,
        evidence_trust_domain=evidence_trust_domain,
        support_scope=support_scope,
    )


def _report(
    *,
    bundle: dict,
    inventory,
    environment: EnvironmentProfile,
    workload: QualificationWorkloadProfile,
    job: ImageJobSpec,
    report_id: str,
    mutate=None,
):
    job_record = job.to_dict()
    report = _report_payload(
        report_id=report_id,
        soft_realtime=job_record["execution_mode"] == "soft_realtime",
    )
    report.update(
        {
            "task": job_record["task"],
            "execution_mode": job_record["execution_mode"],
            "bundle_spec_digest": bundle["spec_digest"],
            "artifact_set_digest": bundle["artifact_set_digest"],
            "artifact_state_fingerprint": inventory.artifact_state_fingerprint,
            "environment_fingerprint": environment.environment_fingerprint,
            "qualification_workload_fingerprint": workload.workload_fingerprint,
            "protocol_fingerprint": PROTOCOL,
            "resolved_pipeline": _pipeline(bundle),
            "source_runtime_provenance": {
                "model_source_id": bundle["model_source_id"],
                "model_revision": bundle["model_revision"],
                "runtime_id": bundle["runtime"]["runtime_id"],
                "runtime_version": bundle["runtime"]["runtime_version"],
                "provider_id": bundle["runtime"]["provider_id"],
                "provider_version": bundle["runtime"]["provider_version"],
            },
        }
    )
    quality = job_record.get("quality_requirement")
    if quality is not None:
        report["quality"] = {
            "status": "known",
            "metric_id": quality["metric_id"],
            "direction": quality["direction"],
            "measured_value": quality["threshold"],
            "threshold_context": quality["threshold"],
            "evaluation_dataset_id": quality["evaluation_dataset_id"],
            "evaluation_dataset_sha256": quality["evaluation_dataset_sha256"],
            "evaluation_protocol_sha256": quality["evaluation_protocol_sha256"],
            "evaluation_vocabulary_id": quality["evaluation_vocabulary_id"],
            "predictions_source": "same_qualification_run",
        }
    if mutate is not None:
        mutate(report)
    _redigest_report(report)
    return validate_qualification_report(report, as_of=AS_OF)


@dataclass
class _Context:
    job: ImageJobSpec
    environment: EnvironmentProfile
    workload: QualificationWorkloadProfile
    registry: LoadedAlgorithmBundleRegistry
    screening: dict[str, ScreeningEligibilityObservation]
    support: dict[tuple[str, str], SupportProfileEligibilityObservation]
    inventories: dict
    evidence: dict[str, EvidenceEligibilityObservation]


def _context(
    *bundles: dict,
    channels: dict[str, list[str]] | None = None,
    job: ImageJobSpec | None = None,
    support_status: dict[tuple[str, str], str] | None = None,
    report_mutators: dict[str, object] | None = None,
    trust: str = "yolozu_managed",
) -> _Context:
    checked_job = job or _job()
    environment = _environment()
    workload = _workload(checked_job)
    registry = validate_algorithm_bundle_registry(_registry_payload(*bundles))
    states = {}
    pointers = {}
    for bundle in bundles:
        states[bundle["spec_digest"]] = {
            "family_id": bundle["family_id"],
            "artifact_set_digest": bundle["artifact_set_digest"],
            "bundle_state": "enabled",
            "artifact_license_reviews": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "order": artifact["order"],
                    "license_expression": artifact["license_expression"],
                    "review_state": "approved",
                }
                for artifact in bundle["artifacts"]
            ],
            "event_digest": "1" * 64,
        }
        selected_channels = (
            channels.get(bundle["spec_digest"], ["Experimental"])
            if channels is not None
            else ["Experimental"]
        )
        for channel in selected_channels:
            pointers[(bundle["family_id"], channel)] = {
                "bundle_spec_digest": bundle["spec_digest"],
                "artifact_set_digest": bundle["artifact_set_digest"],
                "lifecycle_event_digest": ("2" if channel == "Experimental" else "3")
                * 64,
                "support_profile_index_head": "4" * 64,
                "profile_set_record_id": f"profile-set-{channel.lower()}",
                "profile_set_record_digest": "5" * 64,
                "profile_set_digest": "6" * 64,
                "profiles": [],
                "evidence_bindings": [],
            }
    lifecycle = BundleLifecycleProjection("f" * 64, states, pointers, ())
    loaded = LoadedAlgorithmBundleRegistry(
        registry=registry,
        bundles=tuple(sorted(registry.bundles, key=lambda item: item.spec_digest)),
        lifecycle=lifecycle,
        registry_trust_domain=trust,
        lifecycle_trust_domain=trust,
        source_kind="packaged_ssot"
        if trust == "yolozu_managed"
        else "workspace_custom",
    )
    screening = {bundle["spec_digest"]: _screening(bundle) for bundle in bundles}
    support = {}
    inventories = {}
    evidence = {}
    for bundle in bundles:
        spec = bundle["spec_digest"]
        inventory = _inventory(bundle)
        inventories[spec] = inventory
        report = _report(
            bundle=bundle,
            inventory=inventory,
            environment=environment,
            workload=workload,
            job=checked_job,
            report_id=f"report-{spec[:8]}",
            mutate=(report_mutators or {}).get(spec),
        )
        activation = _activation(report.to_dict())
        activation["event_id"] = f"activation-{spec[:8]}"
        activation["event_digest"] = canonical_sha256_v1(
            activation, own_digest_field="event_digest"
        )
        checked_activation = validate_evidence_activation_record(
            activation, source_trust_domain="yolozu_managed"
        )
        key = compute_evidence_selection_key(
            bundle_spec_digest=bundle["spec_digest"],
            artifact_set_digest=bundle["artifact_set_digest"],
            environment_fingerprint=environment.environment_fingerprint,
            qualification_workload_fingerprint=workload.workload_fingerprint,
            protocol_fingerprint=PROTOCOL,
        )
        evidence[key] = EvidenceEligibilityObservation(
            key, "active", checked_activation, report
        )
        for channel in (channels or {}).get(spec, ["Experimental"]):
            pointer = pointers[(bundle["family_id"], channel)]
            status = (support_status or {}).get((spec, channel), "matching_one")
            support[(spec, channel)] = _support(
                bundle=bundle,
                channel=channel,
                pointer=pointer,
                environment=environment,
                workload=workload,
                job=checked_job,
                status=status,
            )
    return _Context(
        checked_job,
        environment,
        workload,
        loaded,
        screening,
        support,
        inventories,
        evidence,
    )


def _select(
    context: _Context,
    *,
    isolation: dict[str, IsolationCapabilityObservation] | None = None,
):
    return select_qualified_pipeline(
        decision_id="decision-1",
        decided_at=DECIDED_AT,
        job=context.job,
        local_input_digest="1" * 64,
        artifact_resolver_state_digest="2" * 64,
        environment=context.environment,
        workload=context.workload,
        protocol_fingerprint=PROTOCOL,
        registry=context.registry,
        screening_observations=context.screening,
        support_profile_observations=context.support,
        artifact_inventories=context.inventories,
        evidence_observations=context.evidence,
        isolation_capabilities=isolation,
        as_of=AS_OF,
    )


class TestAdaptiveSelector(unittest.TestCase):
    def test_empty_registry_abstains_and_evidence_cap_fails_before_selection(
        self,
    ) -> None:
        empty = _context()
        decision = _select(empty).to_dict()
        self.assertEqual(decision["status"], "abstained")
        self.assertEqual(decision["registry_bundle_count"], 0)
        self.assertEqual(decision["candidate_evaluations"], [])

        empty.evidence = {
            (
                key := canonical_sha256_v1({"index": index})
            ): EvidenceEligibilityObservation(key, "absent")
            for index in range(513)
        }
        with self.assertRaisesRegex(ValueError, "registry/evidence_limit_exceeded"):
            _select(empty)

    def test_selects_one_exact_bundle_without_any_provider_file_io(self) -> None:
        bundle = _bundle_payload()
        context = _context(bundle)
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")):
            decision = _select(context).to_dict()
        self.assertEqual(decision["status"], "selected")
        self.assertEqual(
            decision["selected_bundle"]["spec_digest"], bundle["spec_digest"]
        )
        self.assertEqual(decision["candidate_evaluations"][0]["rank_state"], "selected")
        self.assertEqual(
            [
                item["step"]
                for item in decision["candidate_evaluations"][0]["ranking_trace"]
            ],
            list(range(1, 12)),
        )

    def test_noncurrent_catalog_entry_is_complete_and_abstains(self) -> None:
        bundle = _bundle_payload()
        context = _context(bundle, channels={bundle["spec_digest"]: []})
        decision = _select(context).to_dict()
        candidate = decision["candidate_evaluations"][0]
        self.assertEqual(decision["status"], "abstained")
        self.assertEqual(candidate["reason_codes"], ["catalog_only"])
        self.assertEqual(candidate["pointed_channels"], [])
        self.assertIsNone(candidate["support_profile_observation"])

    def test_channel_collapse_prefers_matching_stable_only_after_profile_gate(
        self,
    ) -> None:
        bundle = _bundle_payload()
        spec = bundle["spec_digest"]
        both = {spec: ["Experimental", "Stable"]}
        selected = _select(_context(bundle, channels=both)).to_dict()
        candidate = selected["candidate_evaluations"][0]
        self.assertEqual(candidate["pointed_channels"], ["Experimental", "Stable"])
        self.assertEqual(candidate["matching_channels"], ["Experimental", "Stable"])
        self.assertEqual(candidate["effective_channel"], "Stable")

        experimental = _select(
            _context(
                bundle,
                channels=both,
                support_status={(spec, "Stable"): "no_match"},
            )
        ).to_dict()["candidate_evaluations"][0]
        self.assertEqual(experimental["matching_channels"], ["Experimental"])
        self.assertEqual(experimental["effective_channel"], "Experimental")
        self.assertEqual(experimental["rank_state"], "selected")

    def test_untrusted_registry_and_lifecycle_never_select(self) -> None:
        bundle = _bundle_payload()
        candidate = _select(_context(bundle, trust="operator_asserted")).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(
            candidate["reason_codes"],
            ["lifecycle_untrusted", "registry_untrusted"],
        )

    def test_maturity_lifecycle_and_support_projection_reasons_are_exact(self) -> None:
        bundle = _bundle_payload()
        spec = bundle["spec_digest"]
        maturity = _select(
            _context(bundle, job=_job(allowed_maturities=["Stable"]))
        ).to_dict()["candidate_evaluations"][0]
        self.assertEqual(maturity["reason_codes"], ["maturity_disallowed"])

        for state, reason in (
            ("disabled", "bundle_disabled"),
            ("revoked", "bundle_revoked"),
        ):
            with self.subTest(state=state):
                context = _context(bundle)
                context.registry.lifecycle.bundle_states[spec]["bundle_state"] = state
                candidate = _select(context).to_dict()["candidate_evaluations"][0]
                self.assertEqual(candidate["reason_codes"], [reason])

        for status, reason in (
            ("no_match", "support_profile_mismatch"),
            ("absent", "support_profile_mismatch"),
            ("untrusted", "support_profile_untrusted"),
            ("conflict", "support_profile_conflict"),
        ):
            with self.subTest(status=status):
                candidate = _select(
                    _context(bundle, support_status={(spec, "Experimental"): status})
                ).to_dict()["candidate_evaluations"][0]
                self.assertEqual(candidate["reason_codes"], [reason])

    def test_screened_candidate_hold_excludes_promoted_bundle(self) -> None:
        bundle = _bundle_payload()
        bundle["provenance_class"] = "screened_candidate"
        bundle["screening_binding"] = {
            "stream_key": "screen-1",
            "pass_record_id": "pass-1",
            "pass_record_digest": "7" * 64,
            "source_revision": "revision-1",
        }
        bundle["spec_digest"] = canonical_sha256_v1(
            bundle, own_digest_field="spec_digest"
        )
        context = _context(bundle)
        observed = context.screening[bundle["spec_digest"]].to_dict()
        observed["status"] = "current_hold"
        observed["observation_digest"] = canonical_sha256_v1(
            observed, own_digest_field="observation_digest"
        )
        context.screening[bundle["spec_digest"]] = (
            validate_screening_eligibility_observation(observed, bundle=bundle)
        )
        candidate = _select(context).to_dict()["candidate_evaluations"][0]
        self.assertEqual(candidate["reason_codes"], ["screening_not_current_pass"])

    def test_exact_class_license_and_provider_gates_fail_closed(self) -> None:
        bundle = _bundle_payload()
        class_job = _job(fixed_classes=["bird"])
        class_candidate = _select(_context(bundle, job=class_job)).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(class_candidate["reason_codes"], ["class_vocabulary_mismatch"])

        license_job = _job(spdx_allowlist=["MIT"])
        license_candidate = _select(_context(bundle, job=license_job)).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(license_candidate["reason_codes"], ["license_not_allowed"])

        provider_job = _job(provider_allowlist=["cuda"])
        provider_candidate = _select(_context(bundle, job=provider_job)).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(provider_candidate["reason_codes"], ["provider_not_allowed"])

        network_bundle = _bundle_payload()
        network_bundle["execution_network_required"] = True
        network_bundle["spec_digest"] = canonical_sha256_v1(
            network_bundle, own_digest_field="spec_digest"
        )
        network_candidate = _select(_context(network_bundle)).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(network_candidate["reason_codes"], ["network_required"])

    def test_missing_artifact_and_changed_artifact_state_abstain(self) -> None:
        bundle = _bundle_payload()
        context = _context(bundle)
        context.inventories.clear()
        missing = _select(context).to_dict()["candidate_evaluations"][0]
        self.assertEqual(missing["reason_codes"], ["artifact_member_missing"])

        context = _context(
            bundle,
            report_mutators={
                bundle["spec_digest"]: lambda report: report.__setitem__(
                    "artifact_state_fingerprint", "9" * 64
                )
            },
        )
        changed = _select(context).to_dict()["candidate_evaluations"][0]
        self.assertEqual(changed["reason_codes"], ["artifact_state_mismatch"])

    def test_each_terminal_or_untrusted_evidence_state_has_an_exact_reason(
        self,
    ) -> None:
        bundle = _bundle_payload()
        expected = {
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
        for status, reason in expected.items():
            with self.subTest(status=status):
                context = _context(bundle)
                key = next(iter(context.evidence))
                context.evidence[key] = EvidenceEligibilityObservation(key, status)
                candidate = _select(context).to_dict()["candidate_evaluations"][0]
                self.assertEqual(candidate["reason_codes"], [reason])

    def test_site_managed_evidence_selects_only_site_scope_without_public_claim(
        self,
    ) -> None:
        bundle = _bundle_payload()
        context = _context(bundle)
        key = next(iter(context.evidence))
        current = context.evidence[key]
        assert current.activation_record is not None
        assert current.qualification_report is not None
        activation = current.activation_record.to_dict()
        activation.update(
            {
                "reviewer_role_id": "site_operator",
                "review_reference": {"kind": "site_local_status", "status": "present"},
                "issuer_claim": "site_source",
                "trust_domain": "site_managed",
            }
        )
        activation["event_digest"] = canonical_sha256_v1(
            activation, own_digest_field="event_digest"
        )
        checked_activation = validate_evidence_activation_record(
            activation, source_trust_domain="site_managed"
        )
        context.evidence[key] = EvidenceEligibilityObservation(
            key,
            "active",
            checked_activation,
            current.qualification_report,
        )
        pointer = context.registry.lifecycle.channel_pointers[
            (bundle["family_id"], "Experimental")
        ]
        assert pointer is not None
        context.support[(bundle["spec_digest"], "Experimental")] = _support(
            bundle=bundle,
            channel="Experimental",
            pointer=pointer,
            environment=context.environment,
            workload=context.workload,
            job=context.job,
            status="not_required_site",
            evidence_trust_domain="site_managed",
            support_scope="site_qualified",
        )
        decision = _select(context).to_dict()
        self.assertEqual(decision["status"], "selected")
        self.assertEqual(decision["support_scope"], "site_qualified")
        self.assertEqual(
            decision["candidate_evaluations"][0]["support_profile_observation"][
                "status"
            ],
            "not_required_site",
        )

    def test_third_party_loader_requires_exact_live_isolation_capability(self) -> None:
        bundle = _bundle_payload()
        bundle.update(
            {
                "execution_trust_class": "third_party_isolated",
                "execution_isolation_policy_digest": "f" * 64,
                "loader_format": "python_archive",
                "unsafe_deserialization_required": True,
            }
        )
        bundle["spec_digest"] = canonical_sha256_v1(
            bundle, own_digest_field="spec_digest"
        )
        context = _context(bundle)
        spec = bundle["spec_digest"]
        required = _select(context).to_dict()["candidate_evaluations"][0]
        self.assertEqual(required["reason_codes"], ["isolation_required"])

        missing_image = IsolationCapabilityObservation(
            status="supported",
            backend_id="fixture-isolator",
            backend_version="1",
            isolation_policy_digest="f" * 64,
            image_present=False,
        )
        missing = _select(context, isolation={spec: missing_image}).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(missing["reason_codes"], ["isolation_image_missing"])

        mismatch = IsolationCapabilityObservation(
            status="supported",
            backend_id="fixture-isolator",
            backend_version="1",
            isolation_policy_digest="e" * 64,
            image_present=True,
        )
        mismatched = _select(context, isolation={spec: mismatch}).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(mismatched["reason_codes"], ["isolation_policy_mismatch"])

        supported = IsolationCapabilityObservation(
            status="supported",
            backend_id="fixture-isolator",
            backend_version="1",
            isolation_policy_digest="f" * 64,
            image_present=True,
        )
        self.assertEqual(
            _select(context, isolation={spec: supported}).to_dict()["status"],
            "selected",
        )

    def test_hard_performance_gates_use_exact_mode_specific_aggregates(self) -> None:
        bundle = _bundle_payload()
        cases = (
            ({"max_cold_start_ms": "99"}, "cold_start_above_requirement"),
            ({"max_p95_latency_ms": "21"}, "p95_latency_gate_failed"),
            ({"min_repeat_throughput_fps": "67"}, "repeat_throughput_gate_failed"),
            ({"max_runner_tree_peak_rss_bytes": 1299}, "peak_rss_gate_failed"),
        )
        for updates, reason in cases:
            with self.subTest(reason=reason):
                candidate = _select(_context(bundle, job=_job(**updates))).to_dict()[
                    "candidate_evaluations"
                ][0]
                self.assertIn(reason, candidate["reason_codes"])
                self.assertEqual(candidate["ranking_trace"][-1]["step"], 10)

    def test_soft_realtime_uses_sustained_not_repeat_throughput(self) -> None:
        bundle = _bundle_payload()
        passing_job = _job(
            execution_mode="soft_realtime",
            ranking_policy="throughput_first",
            min_sustained_fps="1600",
            min_repeat_throughput_fps=None,
        )
        # Remove the batch-only field instead of passing a null governed value.
        payload = passing_job.to_dict()
        self.assertNotIn("min_repeat_throughput_fps", payload)
        self.assertEqual(
            _select(_context(bundle, job=passing_job)).to_dict()["status"], "selected"
        )

        failing_job = _job(
            execution_mode="soft_realtime",
            ranking_policy="throughput_first",
            min_sustained_fps="1667",
            min_repeat_throughput_fps=None,
        )
        failed = _select(_context(bundle, job=failing_job)).to_dict()[
            "candidate_evaluations"
        ][0]
        self.assertEqual(failed["reason_codes"], ["sustained_fps_gate_failed"])

    def test_direction_aware_quality_ranking_handles_higher_and_lower_better(
        self,
    ) -> None:
        first = _bundle_payload(version="1.0-rc01")
        second = _bundle_payload(version="1.00")
        channels = {
            first["spec_digest"]: ["Experimental"],
            second["spec_digest"]: ["Stable"],
        }

        def quality(direction: str, threshold: str) -> dict:
            return {
                "metric_id": "fixture_metric",
                "direction": direction,
                "threshold": threshold,
                "evaluation_dataset_id": "fixture-dataset",
                "evaluation_dataset_sha256": "7" * 64,
                "evaluation_protocol_sha256": "8" * 64,
                "evaluation_vocabulary_id": "fixture-vocabulary",
            }

        def measured(value: str):
            return lambda report: report["quality"].__setitem__("measured_value", value)

        higher_job = _job(
            ranking_policy="accuracy_first",
            quality_requirement=quality("higher_is_better", "0.3"),
        )
        higher = _select(
            _context(
                first,
                second,
                channels=channels,
                job=higher_job,
                report_mutators={
                    first["spec_digest"]: measured("0.4"),
                    second["spec_digest"]: measured("0.6"),
                },
            )
        ).to_dict()
        self.assertEqual(higher["selected_bundle"]["bundle_version"], "1.00")

        lower_job = _job(
            ranking_policy="accuracy_first",
            quality_requirement=quality("lower_is_better", "3"),
        )
        lower = _select(
            _context(
                first,
                second,
                channels=channels,
                job=lower_job,
                report_mutators={
                    first["spec_digest"]: measured("2"),
                    second["spec_digest"]: measured("1"),
                },
            )
        ).to_dict()
        self.assertEqual(lower["selected_bundle"]["bundle_version"], "1.00")

    def test_input_order_and_version_text_do_not_change_canonical_tie_break(
        self,
    ) -> None:
        first = _bundle_payload(version="1.0-rc01")
        second = _bundle_payload(version="1.00")
        channels = {
            first["spec_digest"]: ["Experimental"],
            second["spec_digest"]: ["Stable"],
        }
        forward = _select(_context(first, second, channels=channels)).to_dict()
        reverse = _select(_context(second, first, channels=channels)).to_dict()
        self.assertEqual(forward["selected_bundle"], reverse["selected_bundle"])
        self.assertEqual(forward["selected_bundle"]["bundle_version"], "1.0-rc01")
        self.assertEqual(
            [item["spec_digest"] for item in forward["selection_trace"]],
            [item["spec_digest"] for item in reverse["selection_trace"]],
        )

    def test_fixed_class_mapping_is_selected_and_exact(self) -> None:
        bundle = _bundle_payload()
        decision = _select(_context(bundle)).to_dict()
        self.assertEqual(
            decision["selected_class_mapping"],
            build_fixed_class_mapping(validate_algorithm_bundle_spec(bundle), ["cat"]),
        )


if __name__ == "__main__":
    unittest.main()
