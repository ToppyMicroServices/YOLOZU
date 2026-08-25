from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import _bundle_payload
from tests.test_adaptive_image_contracts import _schema_accepts
from yolozu.adaptive.bundles import ZERO_DIGEST, validate_algorithm_bundle_spec
from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.evidence import (
    LATENCY_PHASES,
    compute_artifact_state_fingerprint,
    compute_evidence_selection_key,
    load_evidence_activation_jsonl_bytes,
    project_evidence_activations,
    validate_evidence_activation_record,
    validate_local_artifact_inventory,
    validate_qualification_report,
)


AS_OF = "2026-08-25T00:00:00Z"


def _memory(value: int) -> dict[str, Any]:
    return {
        "status": "known",
        "value_bytes": value,
        "collector_id": "process_tree_memory",
        "collector_version": "1",
        "collector_source_digest": "8" * 64,
        "scope": "runner_process_tree",
        "covered_processes": "all",
        "covered_devices": "not_applicable",
    }


def _accelerator_not_applicable() -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "value_bytes": None,
        "collector_id": "accelerator_memory",
        "collector_version": "1",
        "collector_source_digest": "9" * 64,
        "scope": "not_applicable",
        "covered_processes": "not_applicable",
        "covered_devices": "not_applicable",
    }


def _repeat(index: int) -> dict[str, Any]:
    duration = index * 1_000_000_000
    return {
        "repeat_index": index,
        "status": "completed",
        "failure_code": None,
        "sample_count": 200,
        "duration_ns": duration,
        "p50_latency_ms": str(9 + index),
        "p95_latency_ms": str(19 + index),
        "p99_latency_ms": str(29 + index),
        "throughput_processed_count": 200,
        "throughput_duration_ns": duration,
        "input_coverage_counts": [200],
        "runner_tree_peak_rss": _memory(1_000 + index * 100),
        "accelerator_process_tree_peak": _accelerator_not_applicable(),
    }


def _report_payload(*, report_id: str = "report-1", soft_realtime: bool = False) -> dict[str, Any]:
    repeats = [_repeat(index) for index in range(1, 4)]
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_id": report_id,
        "report_digest": "a" * 64,
        "collector": {
            "id": "yolozu_qualifier",
            "version": "1",
            "source_digest": "1" * 64,
        },
        "issuer": {
            "id": "yolozu_qualification_workflow",
            "version": "1",
            "source_digest": "2" * 64,
        },
        "status": "qualified",
        "task": "object_detection",
        "execution_mode": "soft_realtime" if soft_realtime else "batch",
        "bundle_spec_digest": "3" * 64,
        "artifact_set_digest": "4" * 64,
        "artifact_state_fingerprint": "5" * 64,
        "environment_fingerprint": "6" * 64,
        "qualification_workload_fingerprint": "7" * 64,
        "protocol_fingerprint": "a" * 64,
        "latency_interval": {
            "interval_id": "image_e2e_validated_handoff_v1",
            "handoff_id": "image_result_mask_handoff_v1",
            "handoff_version": 1,
            "included_phases": list(LATENCY_PHASES),
            "publication_boundary": "managed_output_transaction_after_interval",
        },
        "started_at": "2026-08-24T00:00:00Z",
        "completed_at": "2026-08-24T00:20:00Z",
        "valid_until": "2026-10-01T00:20:00Z",
        "repeats": repeats,
        "conservative_aggregates": {
            "repeat_throughput_source_index": 3,
            "repeat_throughput_processed_count": 200,
            "repeat_throughput_duration_ns": 3_000_000_000,
            "p50_latency_ms": "12",
            "p95_latency_ms": "22",
            "p99_latency_ms": "32",
            "runner_tree_peak_rss": _memory(1_300),
            "accelerator_process_tree_peak": _accelerator_not_applicable(),
        },
        "cold_start": {
            "status": "known",
            "cold_start_ms": "100",
            "failure_code": None,
            "fresh_runner": True,
            "os_cache_state": "uncontrolled",
            "interval_id": "image_e2e_validated_handoff_v1",
        },
        "warmup": {
            "status": "completed",
            "iteration_count": 20,
            "failure_code": None,
        },
        "lifetime_memory": {
            "interval_scope": "fresh_runner_creation_through_close",
            "runner_tree_peak_rss": _memory(1_500),
            "accelerator_process_tree_peak": _accelerator_not_applicable(),
        },
        "sustained_section": {
            "status": "not_required",
            "reason": "batch_profile",
        },
        "quality": {
            "status": "not_required",
            "reason": "request_has_no_quality_requirement",
        },
        "resolved_pipeline": {
            name: {"id": name, "version": "1", "source_digest": character * 64}
            for name, character in (
                ("decoder", "b"),
                ("model_input", "c"),
                ("preprocess", "d"),
                ("postprocess", "e"),
            )
        },
        "source_runtime_provenance": {
            "model_source_id": "repository-model-card",
            "model_revision": "revision-1",
            "runtime_id": "onnxruntime",
            "runtime_version": "1.23.0",
            "provider_id": "cpu",
            "provider_version": "1",
        },
        "limitations": ["Representative inputs do not cover every image distribution."],
        "failures": [],
    }
    if soft_realtime:
        report["sustained_section"] = {
            "status": "completed",
            "failure_code": None,
            "schedule_reset_index": 0,
            "duration_ns": 600_000_000_000,
            "processed_count": 1_000_000,
            "sample_count": 1_000_000,
            "max_sustained_samples": 1_000_000,
            "sample_storage_bytes": 8_000_000,
            "aggregation_method": "exact_nearest_rank_all_samples",
            "p95_latency_ms": "20",
            "p99_latency_ms": "30",
            "throughput_processed_count": 1_000_000,
            "throughput_duration_ns": 600_000_000_000,
            "runner_tree_peak_rss": _memory(1_600),
            "accelerator_process_tree_peak": _accelerator_not_applicable(),
            "queue_status": "not_applicable",
            "drop_status": "not_applicable",
            "power_observation": {"status": "unknown", "value": None},
            "thermal_observation": {"status": "known", "value": "not_throttled"},
            "warmup_excluded": True,
            "cold_start_excluded": True,
            "repeat_samples_excluded": True,
        }
    report["report_digest"] = canonical_sha256_v1(
        report, own_digest_field="report_digest"
    )
    return report


def _redigest_report(report: dict[str, Any]) -> None:
    report["report_digest"] = canonical_sha256_v1(
        report, own_digest_field="report_digest"
    )


def _activation(
    report: dict[str, Any],
    *,
    sequence: int = 1,
    previous: str = ZERO_DIGEST,
    state: str = "active",
    replacement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = compute_evidence_selection_key(
        bundle_spec_digest=report["bundle_spec_digest"],
        artifact_set_digest=report["artifact_set_digest"],
        environment_fingerprint=report["environment_fingerprint"],
        qualification_workload_fingerprint=report[
            "qualification_workload_fingerprint"
        ],
        protocol_fingerprint=report["protocol_fingerprint"],
    )
    event: dict[str, Any] = {
        "schema_version": 1,
        "stream_id": key,
        "selection_key": key,
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_id": f"event-{sequence}",
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "state": state,
        "replacement_report_id": None,
        "replacement_report_digest": None,
        "activated_at": f"2026-08-24T0{sequence}:00:00Z",
        "valid_until": "2026-09-30T00:00:00Z",
        "reviewer_role_id": "repo_maintainer",
        "review_reference": {"kind": "public_repository_id", "value": "gh-1"},
        "issuer_claim": "repository_source",
        "trust_domain": "yolozu_managed",
        "reason": "Reviewed qualification evidence.",
        "event_digest": "f" * 64,
    }
    if replacement is not None:
        event["replacement_report_id"] = replacement["report_id"]
        event["replacement_report_digest"] = replacement["report_digest"]
    event["event_digest"] = canonical_sha256_v1(
        event, own_digest_field="event_digest"
    )
    return event


def _redigest_event(event: dict[str, Any]) -> None:
    event["event_digest"] = canonical_sha256_v1(
        event, own_digest_field="event_digest"
    )


class TestAdaptiveEvidenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.schemas = {
            name: json.loads(
                (root / "docs" / "schemas" / f"{name}.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in (
                "local_artifact_inventory",
                "qualification_report",
                "evidence_activation_record",
            )
        }

    def _inventory_payload(self) -> tuple[Any, dict[str, Any]]:
        bundle = validate_algorithm_bundle_spec(_bundle_payload())
        artifact = bundle.to_dict()["artifacts"][0]
        inventory: dict[str, Any] = {
            "schema_version": 1,
            "inventory_id": "inventory-1",
            "bundle_spec_digest": bundle.spec_digest,
            "artifact_set_digest": bundle.artifact_set_digest,
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
                    "verified_at": "2026-08-24T00:00:00Z",
                    "error_status": "none",
                }
            ],
            "artifact_state_fingerprint": "a" * 64,
            "inventory_digest": "b" * 64,
        }
        inventory["artifact_state_fingerprint"] = compute_artifact_state_fingerprint(
            inventory
        )
        inventory["inventory_digest"] = canonical_sha256_v1(
            inventory, own_digest_field="inventory_digest"
        )
        return bundle, inventory

    def test_inventory_validates_exact_bundle_and_separates_fingerprints(self) -> None:
        bundle, payload = self._inventory_payload()
        validated = validate_local_artifact_inventory(payload, bundle)
        self.assertTrue(
            _schema_accepts(
                validated.to_dict(),
                self.schemas["local_artifact_inventory"],
                root=self.schemas["local_artifact_inventory"],
            )
        )
        changed = copy.deepcopy(payload)
        changed["observations"][0]["verified_at"] = "2026-08-24T00:00:01Z"
        changed["inventory_digest"] = canonical_sha256_v1(
            changed, own_digest_field="inventory_digest"
        )
        second = validate_local_artifact_inventory(changed, bundle)
        self.assertEqual(
            validated.artifact_state_fingerprint,
            second.artifact_state_fingerprint,
        )
        self.assertNotEqual(validated.inventory_digest, second.inventory_digest)

    def test_inventory_rejects_missing_member_contradiction_and_path_field(self) -> None:
        bundle, payload = self._inventory_payload()
        cases: list[dict[str, Any]] = []
        missing = copy.deepcopy(payload)
        missing["observations"] = []
        cases.append(missing)
        mismatch = copy.deepcopy(payload)
        mismatch["observations"][0]["observed_size_bytes"] += 1
        cases.append(mismatch)
        path = copy.deepcopy(payload)
        path["observations"][0]["absolute_path"] = "/private/model.onnx"
        cases.append(path)
        for case in cases:
            case["artifact_state_fingerprint"] = compute_artifact_state_fingerprint(case)
            case["inventory_digest"] = canonical_sha256_v1(
                case, own_digest_field="inventory_digest"
            )
            with self.assertRaises(ValueError):
                validate_local_artifact_inventory(case, bundle)

    def test_batch_qualification_validates_schema_and_exact_aggregates(self) -> None:
        payload = _report_payload()
        validated = validate_qualification_report(payload, as_of=AS_OF)
        self.assertTrue(
            _schema_accepts(
                validated.to_dict(),
                self.schemas["qualification_report"],
                root=self.schemas["qualification_report"],
            )
        )
        self.assertEqual(validated.to_dict()["quality"]["status"], "not_required")

    def test_qualification_rejects_incomplete_repeat_coverage(self) -> None:
        for coverage in ([199], [1, 199], [100, 99]):
            payload = _report_payload()
            payload["repeats"][0]["input_coverage_counts"] = coverage
            _redigest_report(payload)
            with self.assertRaises(ValueError):
                validate_qualification_report(payload, as_of=AS_OF)

    def test_qualification_rejects_timing_memory_and_freshness_errors(self) -> None:
        cases: list[dict[str, Any]] = []
        phase = _report_payload()
        phase["latency_interval"]["included_phases"].pop()
        cases.append(phase)
        percentile = _report_payload()
        percentile["repeats"][0]["p95_latency_ms"] = "9"
        cases.append(percentile)
        conservative = _report_payload()
        conservative["conservative_aggregates"]["p95_latency_ms"] = "21"
        cases.append(conservative)
        partial_memory = _report_payload()
        partial_memory["repeats"][0]["runner_tree_peak_rss"]["covered_processes"] = "unknown"
        cases.append(partial_memory)
        warmup = _report_payload()
        warmup["warmup"] = {
            "status": "failed",
            "iteration_count": None,
            "failure_code": "warmup_failed",
        }
        cases.append(warmup)
        lifetime = _report_payload()
        lifetime["lifetime_memory"]["interval_scope"] = "repeat_only"
        cases.append(lifetime)
        too_long = _report_payload()
        too_long["valid_until"] = "2026-11-22T00:20:01Z"
        cases.append(too_long)
        for case in cases:
            _redigest_report(case)
            with self.assertRaises(ValueError):
                validate_qualification_report(case, as_of=AS_OF)
        with self.assertRaises(ValueError):
            validate_qualification_report(_report_payload(), as_of="2026-10-01T00:20:00Z")

    def test_smoke_cannot_masquerade_as_qualified(self) -> None:
        payload = _report_payload()
        payload["status"] = "smoke"
        _redigest_report(payload)
        with self.assertRaises(ValueError):
            validate_qualification_report(payload, as_of=AS_OF)

    def test_soft_realtime_exact_cap_passes_and_over_early_or_approximate_fails(self) -> None:
        payload = _report_payload(soft_realtime=True)
        validate_qualification_report(payload, as_of=AS_OF)
        for name, value in (
            ("sample_count", 1_000_001),
            ("duration_ns", 599_999_999_999),
            ("aggregation_method", "reservoir_sampling"),
            ("warmup_excluded", False),
        ):
            invalid = copy.deepcopy(payload)
            invalid["sustained_section"][name] = value
            _redigest_report(invalid)
            with self.assertRaises(ValueError):
                validate_qualification_report(invalid, as_of=AS_OF)

    def test_known_quality_requires_same_run_and_unknown_has_no_metric_claim(self) -> None:
        payload = _report_payload()
        payload["quality"] = {
            "status": "known",
            "metric_id": "coco_map",
            "direction": "higher_is_better",
            "measured_value": "0.5",
            "threshold_context": "0.4",
            "evaluation_dataset_id": "public-eval",
            "evaluation_dataset_sha256": "a" * 64,
            "evaluation_protocol_sha256": "b" * 64,
            "evaluation_vocabulary_id": "coco-80",
            "predictions_source": "same_qualification_run",
        }
        _redigest_report(payload)
        validate_qualification_report(payload, as_of=AS_OF)
        payload["quality"]["predictions_source"] = "model_card"
        _redigest_report(payload)
        with self.assertRaises(ValueError):
            validate_qualification_report(payload, as_of=AS_OF)

    def test_activation_projects_complete_chain_and_terminal_revoke(self) -> None:
        first = _report_payload(report_id="report-1")
        second = _report_payload(report_id="report-2")
        active = _activation(first)
        superseded = _activation(
            first,
            sequence=2,
            previous=active["event_digest"],
            state="superseded",
            replacement=second,
        )
        replacement = _activation(
            second,
            sequence=3,
            previous=superseded["event_digest"],
        )
        projection = project_evidence_activations(
            [active, superseded, replacement],
            [first, second],
            source_trust_domain="yolozu_managed",
            as_of=AS_OF,
        )
        self.assertEqual(
            next(iter(projection.active_by_selection_key.values())).report_id,
            "report-2",
        )
        revoked = _activation(
            second,
            sequence=4,
            previous=replacement["event_digest"],
            state="revoked",
        )
        projection = project_evidence_activations(
            [active, superseded, replacement, revoked],
            [first, second],
            source_trust_domain="yolozu_managed",
            as_of=AS_OF,
        )
        self.assertEqual(projection.active_by_selection_key, {})
        self.assertEqual(
            next(iter(projection.terminal_reason_by_selection_key.values())),
            "evidence_revoked",
        )

    def test_activation_rejects_gap_fork_conflict_dangling_and_reactivation(self) -> None:
        first = _report_payload(report_id="report-1")
        second = _report_payload(report_id="report-2")
        active = _activation(first)
        conflict = _activation(
            second,
            sequence=2,
            previous=active["event_digest"],
        )
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [active, conflict],
                [first, second],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )
        gap = copy.deepcopy(conflict)
        gap["sequence"] = 3
        _redigest_event(gap)
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [active, gap],
                [first, second],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )
        fork = copy.deepcopy(conflict)
        fork["previous_event_digest"] = "1" * 64
        _redigest_event(fork)
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [active, fork],
                [first, second],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )
        superseded = _activation(
            first,
            sequence=2,
            previous=active["event_digest"],
            state="superseded",
            replacement=second,
        )
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [active, superseded],
                [first, second],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )
        revoked = _activation(
            first,
            sequence=2,
            previous=active["event_digest"],
            state="revoked",
        )
        reactivated = _activation(
            first,
            sequence=3,
            previous=revoked["event_digest"],
        )
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [active, revoked, reactivated],
                [first],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )

    def test_activation_trust_time_and_schema_are_strict(self) -> None:
        report = _report_payload()
        event = _activation(report)
        validated = validate_evidence_activation_record(
            event, source_trust_domain="yolozu_managed"
        )
        self.assertTrue(
            _schema_accepts(
                validated.to_dict(),
                self.schemas["evidence_activation_record"],
                root=self.schemas["evidence_activation_record"],
            )
        )
        personal = copy.deepcopy(event)
        personal["reviewer_role_id"] = "alice"
        _redigest_event(personal)
        with self.assertRaises(ValueError):
            validate_evidence_activation_record(
                personal, source_trust_domain="yolozu_managed"
            )
        with self.assertRaises(ValueError):
            validate_evidence_activation_record(
                event, source_trust_domain="operator_asserted"
            )
        future = copy.deepcopy(event)
        future["activated_at"] = "2026-08-26T00:00:00Z"
        _redigest_event(future)
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [future],
                [report],
                source_trust_domain="yolozu_managed",
                as_of=AS_OF,
            )
        with self.assertRaises(ValueError):
            project_evidence_activations(
                [event],
                [report],
                source_trust_domain="yolozu_managed",
                as_of="2026-09-30T00:00:00Z",
            )

    def test_activation_stream_limits_and_complete_suffix(self) -> None:
        report = _report_payload()
        event = _activation(report)
        encoded = json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
        with patch(
            "yolozu.adaptive.evidence.MAX_EVIDENCE_ACTIVATION_BYTES", len(encoded)
        ):
            self.assertEqual(len(load_evidence_activation_jsonl_bytes(encoded)), 1)
        with patch(
            "yolozu.adaptive.evidence.MAX_EVIDENCE_ACTIVATION_BYTES",
            len(encoded) - 1,
        ):
            with self.assertRaises(ValueError):
                load_evidence_activation_jsonl_bytes(encoded)
        with self.assertRaises(ValueError):
            load_evidence_activation_jsonl_bytes(encoded[:-1])
        with patch("yolozu.adaptive.evidence.MAX_EVIDENCE_ACTIVATION_RECORDS", 1):
            with self.assertRaises(ValueError):
                load_evidence_activation_jsonl_bytes(encoded + encoded)


if __name__ == "__main__":
    unittest.main()
