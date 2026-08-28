from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tests.test_adaptive_bundle_contracts import (
    _bundle_payload,
    _lifecycle_event,
    _registry_payload,
    _support_record,
)
from tests.test_adaptive_evidence_contracts import (
    _activation,
    _redigest_event,
    _redigest_report,
    _report_payload,
)
from tests.test_adaptive_candidate_screening import (
    _finalize as _finalize_screening,
    _record as _screening_record,
)
from tests.test_adaptive_image_contracts import _schema_accepts
from tests.test_adaptive_lifecycle_governance import (
    _Workspace as _LifecycleWorkspace,
    _target_rollback_args,
)
from tests.test_adaptive_qualification import _blocking_runner_factory
from tests.test_adaptive_selector import (
    _context as _selection_context,
    _environment as _selection_environment,
    _select as _select_pipeline,
)
from tests.test_adaptive_support_profile_governance import _profile
from yolozu.adaptive.artifact_resolver import ArtifactResolver
from yolozu.adaptive.bundles import (
    EMPTY_PROFILE_SET_DIGEST,
    ZERO_DIGEST,
    validate_algorithm_bundle_spec,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.inventory import pin_decoded_inputs
from yolozu.adaptive.lifecycle import _load_state, update_image_pipeline_lifecycle
from yolozu.adaptive.promotion import (
    _compare_stable_reports,
    promote_image_pipeline,
)
from yolozu.adaptive.qualification import QualificationError, _ForkedRunnerSession
from yolozu.adaptive.screening import validate_candidate_screening_record
from yolozu.adaptive.selection import validate_screening_eligibility_observation
from yolozu.cli_entry import main as cli_main


def _jsonl(*records: dict) -> bytes:
    return b"".join(canonical_json_v1(record) + b"\n" for record in records)


class _ScreeningPass:
    def to_dict(self) -> dict:
        return {"status": "current_pass", "trust_domain": "yolozu_managed"}


class _PromotionWorkspace:
    def __init__(self, *, target_channel: str, baseline: bool = False) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "yolozu" / "data" / "adaptive_routing"
        self.data.mkdir(parents=True)
        (self.data / "qualification_reports").mkdir()

        self.candidate = _bundle_payload(data=b"candidate", version="2.0")
        bundles = [self.candidate]
        self.baseline = None
        if baseline:
            self.baseline = _bundle_payload(data=b"baseline", version="1.0")
            bundles.append(self.baseline)
        registry = _registry_payload(*bundles)
        (self.data / "bundle_specs.json").write_bytes(
            canonical_json_v1(registry) + b"\n"
        )

        profile = _profile()
        profile["advertised_constraints"].update(
            {
                "max_p99_latency_ms": "60",
                "max_runner_tree_peak_rss_bytes": 2_000,
                "quality_requirement": {
                    "metric_id": "map50",
                    "direction": "higher_is_better",
                    "threshold": "0.5",
                    "evaluation_dataset_id": "fixture-set",
                    "evaluation_dataset_sha256": "a" * 64,
                    "evaluation_protocol_sha256": "b" * 64,
                    "evaluation_vocabulary_id": "fixture-vocabulary",
                },
            }
        )
        profile["profile_digest"] = canonical_sha256_v1(
            profile, own_digest_field="profile_digest"
        )
        self.profile = profile
        self.refs = [
            {
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
            }
        ]
        definition = _support_record(
            sequence=1,
            previous=ZERO_DIGEST,
            record_id="define-cpu-batch",
            kind="profile_definition",
            variant={"profile": profile},
        )
        records = [definition]

        def assignment(channel: str, record_id: str) -> dict:
            item = _support_record(
                sequence=len(records) + 1,
                previous=records[-1]["record_digest"],
                record_id=record_id,
                kind="profile_set_assignment",
                variant={
                    "family_id": "example-detector",
                    "channel": channel,
                    "profiles": self.refs,
                    "profile_set_digest": canonical_sha256_v1(self.refs),
                },
            )
            records.append(item)
            return item

        self.experimental_assignment = assignment("Experimental", "assign-experimental")
        self.stable_assignment = (
            assignment("Stable", "assign-stable")
            if target_channel == "Stable"
            else None
        )
        self.assignment = (
            self.experimental_assignment
            if target_channel == "Experimental"
            else self.stable_assignment
        )
        assert self.assignment is not None
        self.support_head = records[-1]["record_digest"]
        (self.data / "support_profiles.jsonl").write_bytes(_jsonl(*records))

        reviews = [{"artifact_id": "model", "review_state": "approved"}]
        events: list[dict] = []

        def add(scope: str, event_type: str, variant: dict) -> dict:
            event = _lifecycle_event(
                sequence=len(events) + 1,
                previous=ZERO_DIGEST if not events else events[-1]["event_digest"],
                scope=scope,
                event_type=event_type,
                variant=variant,
            )
            events.append(event)
            return event

        add(
            "bundle_global",
            "register_global",
            {
                "family_id": "example-detector",
                "bundle_spec_digest": self.candidate["spec_digest"],
                "artifact_set_digest": self.candidate["artifact_set_digest"],
                "bundle_state": "enabled",
                "artifact_license_reviews": reviews,
            },
        )
        self.candidate_pointer = add(
            "channel_assignment",
            "candidate_registration",
            {
                "family_id": "example-detector",
                "channel": "Candidate",
                "target_bundle_spec_digest": self.candidate["spec_digest"],
                "target_artifact_set_digest": self.candidate["artifact_set_digest"],
                "target_artifact_license_reviews": reviews,
                "support_profile_index_head": self.support_head,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            },
        )
        self.experimental_pointer = None
        if target_channel == "Stable":
            self.experimental_pointer = add(
                "channel_assignment",
                "public_assignment",
                {
                    "family_id": "example-detector",
                    "channel": "Experimental",
                    "target_bundle_spec_digest": self.candidate["spec_digest"],
                    "target_artifact_set_digest": self.candidate["artifact_set_digest"],
                    "target_artifact_license_reviews": reviews,
                    "support_profile_index_head": self.experimental_assignment[
                        "record_digest"
                    ],
                    "profile_set_record_id": self.experimental_assignment["record_id"],
                    "profile_set_record_digest": self.experimental_assignment[
                        "record_digest"
                    ],
                    "profile_set_digest": canonical_sha256_v1(self.refs),
                    "profiles": self.refs,
                    "evidence_bindings": [self._placeholder_binding("candidate")],
                },
            )
        self.stable_pointer = None
        if self.baseline is not None:
            add(
                "bundle_global",
                "register_global",
                {
                    "family_id": "example-detector",
                    "bundle_spec_digest": self.baseline["spec_digest"],
                    "artifact_set_digest": self.baseline["artifact_set_digest"],
                    "bundle_state": "enabled",
                    "artifact_license_reviews": reviews,
                },
            )
            add(
                "channel_assignment",
                "candidate_registration",
                {
                    "family_id": "example-detector",
                    "channel": "Candidate",
                    "target_bundle_spec_digest": self.baseline["spec_digest"],
                    "target_artifact_set_digest": self.baseline["artifact_set_digest"],
                    "target_artifact_license_reviews": reviews,
                    "support_profile_index_head": self.support_head,
                    "profile_set_record_id": None,
                    "profile_set_record_digest": None,
                    "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                    "profiles": [],
                    "evidence_bindings": [],
                },
            )
            assert self.stable_assignment is not None
            self.stable_pointer = add(
                "channel_assignment",
                "public_assignment",
                {
                    "family_id": "example-detector",
                    "channel": "Stable",
                    "target_bundle_spec_digest": self.baseline["spec_digest"],
                    "target_artifact_set_digest": self.baseline["artifact_set_digest"],
                    "target_artifact_license_reviews": reviews,
                    "support_profile_index_head": self.support_head,
                    "profile_set_record_id": self.stable_assignment["record_id"],
                    "profile_set_record_digest": self.stable_assignment[
                        "record_digest"
                    ],
                    "profile_set_digest": canonical_sha256_v1(self.refs),
                    "profiles": self.refs,
                    "evidence_bindings": [self._placeholder_binding("baseline")],
                },
            )
        self.events = events
        self.lifecycle = self.data / "bundle_lifecycle.jsonl"
        self.lifecycle.write_bytes(_jsonl(*events))
        (self.data / "candidate_screening.jsonl").write_bytes(b"")
        (self.data / "evidence_activation.jsonl").write_bytes(b"")
        self.install_evidence(self.candidate, "candidate-report", "0.8")
        if self.baseline is not None:
            self.install_evidence(self.baseline, "baseline-report", "0.8")
        self.write_bindings()

    def _placeholder_binding(self, name: str) -> dict:
        return {
            "profile_id": self.profile["profile_id"],
            "profile_digest": self.profile["profile_digest"],
            "activation_id": f"{name}-activation",
            "activation_digest": "f" * 64,
            "trust_domain_claim": "yolozu_managed",
        }

    @property
    def head(self) -> str:
        return self.events[-1]["event_digest"]

    def _report(self, bundle: dict, report_id: str, quality: str) -> dict:
        report = _report_payload(report_id=report_id)
        report["bundle_spec_digest"] = bundle["spec_digest"]
        report["artifact_set_digest"] = bundle["artifact_set_digest"]
        report["environment_fingerprint"] = self.profile["environment_fingerprint"]
        report["qualification_workload_fingerprint"] = self.profile[
            "qualification_workload_fingerprint"
        ]
        report["protocol_fingerprint"] = self.profile["protocol_fingerprint"]
        report["quality"] = {
            "status": "known",
            "metric_id": "map50",
            "direction": "higher_is_better",
            "measured_value": quality,
            "threshold_context": "0.5",
            "evaluation_dataset_id": "fixture-set",
            "evaluation_dataset_sha256": "a" * 64,
            "evaluation_protocol_sha256": "b" * 64,
            "evaluation_vocabulary_id": "fixture-vocabulary",
            "predictions_source": "same_qualification_run",
        }
        report["resolved_pipeline"] = {
            name: {
                "id": bundle[name]["id"],
                "version": bundle[name]["version"],
                "source_digest": bundle[name]["digest"],
            }
            for name in ("decoder", "preprocess", "postprocess")
        }
        report["resolved_pipeline"]["model_input"] = {
            "id": "bundle_model_input_shapes",
            "version": "1",
            "source_digest": canonical_sha256_v1(bundle["model_input_shapes"]),
        }
        runtime = bundle["runtime"]
        report["source_runtime_provenance"] = {
            "model_source_id": bundle["model_source_id"],
            "model_revision": bundle["model_revision"],
            "runtime_id": runtime["runtime_id"],
            "runtime_version": runtime["runtime_version"],
            "provider_id": runtime["provider_id"],
            "provider_version": runtime["provider_version"],
        }
        _redigest_report(report)
        return report

    def install_evidence(self, bundle: dict, report_id: str, quality: str) -> None:
        report = self._report(bundle, report_id, quality)
        activation = _activation(report)
        activation["event_id"] = f"{report_id}-activation"
        existing = (self.data / "evidence_activation.jsonl").read_bytes()
        if existing:
            prior = existing.strip().splitlines()[-1]
            import json

            previous_event = json.loads(prior)
            activation["sequence"] = 1
            activation["previous_event_digest"] = ZERO_DIGEST
            # Independent selection keys each start their own sequence-1 chain.
            if activation["stream_id"] == previous_event["stream_id"]:
                raise AssertionError(
                    "fixture bundles must have distinct selection keys"
                )
        _redigest_event(activation)
        with (self.data / "evidence_activation.jsonl").open("ab") as stream:
            stream.write(canonical_json_v1(activation) + b"\n")
        report_root = self.data / "qualification_reports" / report_id
        report_root.mkdir()
        (report_root / "qualification_report.json").write_bytes(
            canonical_json_v1(report) + b"\n"
        )

    def write_bindings(self) -> None:
        import json

        records = [
            json.loads(line)
            for line in (self.data / "evidence_activation.jsonl")
            .read_text()
            .splitlines()
        ]
        candidate_event = next(
            item for item in records if item["report_id"] == "candidate-report"
        )
        self.binding = {
            "profile_id": self.profile["profile_id"],
            "profile_digest": self.profile["profile_digest"],
            "activation_id": candidate_event["event_id"],
            "activation_digest": candidate_event["event_digest"],
            "trust_domain_claim": "yolozu_managed",
        }
        self.bindings = self.root / "promotion-bindings.json"
        self.bindings.write_bytes(
            canonical_json_v1({"schema_version": 1, "bindings": [self.binding]}) + b"\n"
        )

    def write_drill(self, *, automated: str = "ci-promotion-drill") -> Path:
        value = {
            "schema_version": 1,
            "family_id": "example-detector",
            "bundle_spec_digest": self.candidate["spec_digest"],
            "lifecycle_head_digest": self.head,
            "profile_set_digest": canonical_sha256_v1(self.refs),
            "status": "passed",
            "failure_codes": [
                "artifact_hash_mismatch",
                "runtime_unavailable",
                "out_of_memory_or_timeout",
                "metric_regression",
                "license_failure",
                "interface_contract_failure",
            ],
            "public_repository_reference": "gh-drill-1",
            "automated_pass_reference": automated,
            "report_digest": ZERO_DIGEST,
        }
        value["report_digest"] = canonical_sha256_v1(
            value, own_digest_field="report_digest"
        )
        path = self.root / "failure-drill.json"
        path.write_bytes(canonical_json_v1(value) + b"\n")
        return path

    def args(
        self, *, target_channel: str, approve: bool = False, **updates: object
    ) -> dict:
        source_pointer = (
            self.candidate_pointer
            if target_channel == "Experimental"
            else self.experimental_pointer
        )
        target_pointer = self.stable_pointer if target_channel == "Stable" else None
        failure_drill = updates.pop("failure_drill_report_path", None)
        if target_channel == "Stable" and failure_drill is None:
            failure_drill = self.write_drill().relative_to(self.root)
        values: dict[str, object] = {
            "workspace_root": self.root,
            "family_id": "example-detector",
            "source_channel": "Candidate"
            if target_channel == "Experimental"
            else "Experimental",
            "target_channel": target_channel,
            "bundle_spec_digest": self.candidate["spec_digest"],
            "expected_source_pointer_digest": source_pointer["event_digest"],
            "expected_target_pointer_digest": (
                "none" if target_pointer is None else target_pointer["event_digest"]
            ),
            "expected_lifecycle_head_digest": self.head,
            "expected_support_profile_index_head": self.support_head,
            "expected_profile_set_record_digest": self.assignment["record_digest"],
            "expected_profile_set_digest": canonical_sha256_v1(self.refs),
            "profiles": self.refs,
            "evidence_bindings_path": self.bindings.relative_to(self.root),
            "rollback_target": "none" if target_pointer is None else "prior",
            "approver_role_id": "repo_maintainer",
            "public_review_id": "gh-promotion-1",
            "reason": "Review one exact evidence-bound public promotion.",
            "failure_drill_report_path": failure_drill,
            "approve": approve,
            "occurred_at": "2026-08-26T00:00:00Z",
        }
        values.update(updates)
        return values

    def cleanup(self) -> None:
        self.temporary.cleanup()


class PromotionServiceTests(unittest.TestCase):
    @patch(
        "yolozu.adaptive.promotion.build_screening_eligibility_observation",
        return_value=_ScreeningPass(),
    )
    def test_first_experimental_dry_run_then_atomic_append(self, _screening) -> None:
        workspace = _PromotionWorkspace(target_channel="Experimental")
        self.addCleanup(workspace.cleanup)
        immutable = {
            name: (workspace.data / name).read_bytes()
            for name in (
                "bundle_specs.json",
                "support_profiles.jsonl",
                "evidence_activation.jsonl",
                "candidate_screening.jsonl",
            )
        }
        before = workspace.lifecycle.read_bytes()
        dry_run = promote_image_pipeline(
            **workspace.args(target_channel="Experimental")
        )
        self.assertEqual(dry_run.status, "dry_run_ready")
        self.assertEqual(workspace.lifecycle.read_bytes(), before)
        self.assertEqual(
            dry_run.planned_record["rollback_target_status"], "none_abstention"
        )
        import json

        schema = json.loads(
            Path("docs/schemas/bundle_lifecycle_record.schema.json").read_text()
        )
        self.assertTrue(_schema_accepts(dry_run.planned_record, schema, root=schema))

        applied = promote_image_pipeline(
            **workspace.args(target_channel="Experimental", approve=True)
        )
        self.assertEqual(applied.status, "applied")
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))
        for name, expected in immutable.items():
            self.assertEqual((workspace.data / name).read_bytes(), expected)

    @patch(
        "yolozu.adaptive.promotion.build_screening_eligibility_observation",
        return_value=_ScreeningPass(),
    )
    def test_stale_subset_site_and_atomic_failure_do_not_mutate(
        self, _screening
    ) -> None:
        cases = (
            ({"expected_lifecycle_head_digest": "f" * 64}, "lifecycle_head_stale"),
            ({"profiles": []}, "profiles_invalid"),
            (
                {"expected_support_profile_index_head": "e" * 64},
                "support_profile_head_stale",
            ),
        )
        for updates, code in cases:
            with self.subTest(code=code):
                workspace = _PromotionWorkspace(target_channel="Experimental")
                try:
                    before = workspace.lifecycle.read_bytes()
                    outcome = promote_image_pipeline(
                        **workspace.args(
                            target_channel="Experimental", approve=True, **updates
                        )
                    )
                    self.assertEqual(outcome.status, "apply_failed")
                    self.assertIn(code, [gate.code for gate in outcome.gates])
                    self.assertEqual(workspace.lifecycle.read_bytes(), before)
                finally:
                    workspace.cleanup()

        workspace = _PromotionWorkspace(target_channel="Experimental")
        self.addCleanup(workspace.cleanup)
        events = (
            workspace.data.joinpath("evidence_activation.jsonl")
            .read_text()
            .splitlines()
        )
        import json

        site = json.loads(events[0])
        site["trust_domain"] = "site_managed"
        _redigest_event(site)
        workspace.data.joinpath("evidence_activation.jsonl").write_bytes(_jsonl(site))
        workspace.write_bindings()
        blocked = promote_image_pipeline(
            **workspace.args(target_channel="Experimental", approve=True)
        )
        self.assertIn(
            "promotion_evidence_invalid", [gate.code for gate in blocked.gates]
        )

        atomic_workspace = _PromotionWorkspace(target_channel="Experimental")
        self.addCleanup(atomic_workspace.cleanup)
        before = atomic_workspace.lifecycle.read_bytes()

        def fail(stage: str) -> None:
            if stage == "before_replace":
                raise OSError("fixture interruption")

        failed = promote_image_pipeline(
            **atomic_workspace.args(
                target_channel="Experimental", approve=True, fault_hook=fail
            )
        )
        self.assertEqual(failed.status, "apply_failed")
        self.assertEqual(atomic_workspace.lifecycle.read_bytes(), before)

    def test_first_and_replacement_stable_require_human_drill_and_exact_set(
        self,
    ) -> None:
        first = _PromotionWorkspace(target_channel="Stable")
        self.addCleanup(first.cleanup)
        ready = promote_image_pipeline(**first.args(target_channel="Stable"))
        self.assertEqual(ready.status, "dry_run_ready")
        self.assertEqual(
            ready.planned_record["stable_comparator_status"],
            "comparator_not_applicable_first_assignment",
        )
        same_reference = first.write_drill(automated="gh-promotion-1")
        blocked = promote_image_pipeline(
            **first.args(
                target_channel="Stable",
                failure_drill_report_path=same_reference.relative_to(first.root),
            )
        )
        self.assertIn(
            "human_approval_not_distinct", [gate.code for gate in blocked.gates]
        )

        replacement = _PromotionWorkspace(target_channel="Stable", baseline=True)
        self.addCleanup(replacement.cleanup)
        ready = promote_image_pipeline(**replacement.args(target_channel="Stable"))
        self.assertEqual(ready.status, "dry_run_ready")
        self.assertEqual(
            ready.planned_record["rollback_target_status"], "prior_assignment"
        )
        reordered = [*replacement.refs, copy.deepcopy(replacement.refs[0])]
        blocked = promote_image_pipeline(
            **replacement.args(target_channel="Stable", profiles=reordered)
        )
        self.assertIn(
            "profile_set_echo_mismatch", [gate.code for gate in blocked.gates]
        )

    def test_zero_tolerance_comparison_covers_both_quality_directions_and_metrics(
        self,
    ) -> None:
        workspace = _PromotionWorkspace(target_channel="Stable")
        self.addCleanup(workspace.cleanup)
        baseline = workspace._report(workspace.candidate, "base", "0.8")
        candidate = copy.deepcopy(baseline)
        self.assertEqual(_compare_stable_reports(candidate, baseline), [])

        for field, value, expected in (
            ("p95_latency_ms", "23", "stable p95_latency_ms regressed"),
            ("p99_latency_ms", "33", "stable p99_latency_ms regressed"),
        ):
            changed = copy.deepcopy(candidate)
            changed["conservative_aggregates"][field] = value
            self.assertIn(expected, _compare_stable_reports(changed, baseline))
        changed = copy.deepcopy(candidate)
        changed["conservative_aggregates"]["repeat_throughput_duration_ns"] += 1
        self.assertIn(
            "stable throughput regressed", _compare_stable_reports(changed, baseline)
        )
        changed = copy.deepcopy(candidate)
        changed["conservative_aggregates"]["runner_tree_peak_rss"]["status"] = "unknown"
        changed["conservative_aggregates"]["runner_tree_peak_rss"]["value_bytes"] = None
        self.assertIn(
            "stable runner_tree_peak_rss is unknown",
            _compare_stable_reports(changed, baseline),
        )
        changed = copy.deepcopy(candidate)
        changed["quality"]["measured_value"] = "0.7"
        self.assertIn(
            "stable quality regressed", _compare_stable_reports(changed, baseline)
        )
        lower_baseline = copy.deepcopy(baseline)
        lower_baseline["quality"]["direction"] = "lower_is_better"
        lower_candidate = copy.deepcopy(lower_baseline)
        lower_candidate["quality"]["measured_value"] = "0.9"
        self.assertIn(
            "stable quality regressed",
            _compare_stable_reports(lower_candidate, lower_baseline),
        )


class PromotionFailureDrillTests(unittest.TestCase):
    fixture_root = Path("tests/fixtures/adaptive_promotion_failure_drills")

    def _load_fixtures(self) -> list[dict]:
        manifest_path = self.fixture_root / "checksums.json"
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        self.assertEqual(manifest_raw, canonical_json_v1(manifest) + b"\n")
        self.assertNotIn("checksums.json", {
            item["path"] for item in manifest["declared_files"]
        })
        self.assertEqual(manifest["expected_file_count"], 1)
        self.assertEqual(len(manifest["declared_files"]), 1)

        declared = manifest["declared_files"][0]
        fixture_path = self.fixture_root / declared["path"]
        fixture_raw = fixture_path.read_bytes()
        self.assertEqual(len(fixture_raw), declared["size_bytes"])
        self.assertEqual(hashlib.sha256(fixture_raw).hexdigest(), declared["sha256"])
        self.assertEqual(manifest["expected_total_bytes"], len(fixture_raw))
        self.assertEqual(
            sorted(path.name for path in self.fixture_root.iterdir()),
            ["checksums.json", "fixtures.jsonl"],
        )

        fixtures = [json.loads(line) for line in fixture_raw.splitlines()]
        for fixture, line in zip(fixtures, fixture_raw.splitlines(), strict=True):
            self.assertEqual(line, canonical_json_v1(fixture))
        self.assertEqual(
            [item["aggregate_failure_code"] for item in fixtures],
            [
                "artifact_hash_mismatch",
                "runtime_unavailable",
                "out_of_memory_or_timeout",
                "metric_regression",
                "license_failure",
                "interface_contract_failure",
            ],
        )
        return fixtures

    def _artifact_mismatch_reason(self) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "weights" / "model.onnx"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"tampered")
            bundle = validate_algorithm_bundle_spec(
                _bundle_payload(data=b"expected")
            )
            with ArtifactResolver(
                workspace=root, artifact_root=artifact_root
            ) as resolver, self.assertRaisesRegex(
                ValueError, "SHA-256 mismatch"
            ) as raised:
                resolver.pin(bundle)
        return [str(raised.exception)]

    def _timeout_reason(self) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            artifact_root = root / "artifacts"
            artifact_path = artifact_root / "weights" / "model.onnx"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"model")
            bundle = validate_algorithm_bundle_spec(_bundle_payload())
            with (
                pin_decoded_inputs(
                    image_path,
                    input_mode="single_image",
                    workspace_root=root,
                    max_images=1,
                ) as inputs,
                ArtifactResolver(
                    workspace=root, artifact_root=artifact_root
                ) as resolver,
                resolver.pin(bundle) as artifacts,
            ):
                session = _ForkedRunnerSession(
                    factory=_blocking_runner_factory,
                    bundle=bundle,
                    environment=_selection_environment(),
                    artifacts=artifacts,
                    inputs=inputs,
                    labels=("cat",),
                    outer_deadline_ns=time.monotonic_ns() + 10_000_000_000,
                )
                with self.assertRaises(QualificationError) as raised:
                    session.warmup(0, 0)
                self.assertFalse(session._process.is_alive())
        return [raised.exception.code]

    def _screening_reason(self, *, interface: bool) -> list[str]:
        record = _screening_record()
        if interface:
            record["mechanical_checks"]["predictions_interface_mapping"].update(
                {
                    "result": "fail",
                    "interface_contract_id": None,
                    "mapping_revision": None,
                }
            )
        else:
            record["mechanical_checks"]["weight_license"].update(
                {"result": "unknown", "spdx_id": None}
            )
        checked = validate_candidate_screening_record(
            _finalize_screening(record),
            source_trust_domain="yolozu_managed",
        ).to_dict()
        return checked["reason_codes"]

    def _observed_reasons(self, failure_code: str) -> list[str]:
        if failure_code == "artifact_hash_mismatch":
            return self._artifact_mismatch_reason()
        if failure_code == "runtime_unavailable":
            _status, _selected, reasons = self._selection_result(failure_code)
            return reasons
        if failure_code == "out_of_memory_or_timeout":
            return self._timeout_reason()
        if failure_code == "metric_regression":
            workspace = _PromotionWorkspace(target_channel="Stable")
            try:
                baseline = workspace._report(workspace.candidate, "base", "0.8")
                candidate = copy.deepcopy(baseline)
                candidate["conservative_aggregates"]["p95_latency_ms"] = "23"
                return _compare_stable_reports(candidate, baseline)
            finally:
                workspace.cleanup()
        if failure_code == "license_failure":
            return self._screening_reason(interface=False)
        if failure_code == "interface_contract_failure":
            return self._screening_reason(interface=True)
        raise AssertionError(f"unknown failure fixture: {failure_code}")

    @staticmethod
    def _regress_p95(report: dict) -> None:
        for repeat in report["repeats"]:
            repeat["p95_latency_ms"] = "100"
            repeat["p99_latency_ms"] = "100"
        report["conservative_aggregates"]["p95_latency_ms"] = "100"
        report["conservative_aggregates"]["p99_latency_ms"] = "100"

    def _selection_result(self, failure_code: str) -> tuple[str, str | None, list[str]]:
        baseline = _bundle_payload(data=b"baseline", version="1.0")
        candidate = _bundle_payload(data=b"candidate", version="2.0")
        if failure_code == "runtime_unavailable":
            candidate["runtime"]["runtime_version"] = "9.0"
            candidate["spec_digest"] = canonical_sha256_v1(
                candidate, own_digest_field="spec_digest"
            )
        if failure_code == "interface_contract_failure":
            candidate["provenance_class"] = "screened_candidate"
            candidate["screening_binding"] = {
                "stream_key": "screen-fixture",
                "pass_record_id": "screen-pass",
                "pass_record_digest": "7" * 64,
                "source_revision": "revision-1",
            }
            candidate["spec_digest"] = canonical_sha256_v1(
                candidate, own_digest_field="spec_digest"
            )
            context = _selection_context(
                candidate,
                channels={candidate["spec_digest"]: ["Experimental"]},
            )
            observed = context.screening[candidate["spec_digest"]].to_dict()
            observed["status"] = "current_reject"
            observed["observation_digest"] = canonical_sha256_v1(
                observed, own_digest_field="observation_digest"
            )
            context.screening[candidate["spec_digest"]] = (
                validate_screening_eligibility_observation(
                    observed, bundle=candidate
                )
            )
        else:
            report_mutators = (
                {candidate["spec_digest"]: self._regress_p95}
                if failure_code == "metric_regression"
                else None
            )
            context = _selection_context(
                baseline,
                candidate,
                channels={
                    baseline["spec_digest"]: ["Stable"],
                    candidate["spec_digest"]: ["Experimental"],
                },
                report_mutators=report_mutators,
            )
            if failure_code == "artifact_hash_mismatch":
                context.inventories.pop(candidate["spec_digest"])
            elif failure_code == "out_of_memory_or_timeout":
                context.evidence = {
                    key: value
                    for key, value in context.evidence.items()
                    if value.qualification_report.to_dict()["bundle_spec_digest"]
                    != candidate["spec_digest"]
                }
            elif failure_code == "license_failure":
                context.registry.lifecycle.bundle_states[candidate["spec_digest"]][
                    "artifact_license_reviews"
                ][0]["review_state"] = "unknown"

        decision = _select_pipeline(context).to_dict()
        candidate_evaluation = next(
            item
            for item in decision["candidate_evaluations"]
            if item["spec_digest"] == candidate["spec_digest"]
        )
        selected = decision["selected_bundle"]
        return (
            decision["status"],
            None if selected is None else selected["bundle_version"],
            candidate_evaluation["reason_codes"],
        )

    def _failed_drill_path(
        self, workspace: _PromotionWorkspace, failure_code: str
    ) -> Path:
        path = workspace.write_drill()
        report = json.loads(path.read_bytes())
        report["status"] = "failed"
        report["failure_codes"] = [failure_code]
        report["report_digest"] = canonical_sha256_v1(
            report, own_digest_field="report_digest"
        )
        path.write_bytes(canonical_json_v1(report) + b"\n")
        return path

    @patch(
        "yolozu.adaptive.promotion.build_screening_eligibility_observation",
        return_value=_ScreeningPass(),
    )
    def test_six_offline_failure_fixtures_cannot_promote(self, _screening) -> None:
        for fixture in self._load_fixtures():
            failure_code = fixture["aggregate_failure_code"]
            with self.subTest(fixture=fixture["fixture_id"]):
                self.assertEqual(
                    self._observed_reasons(failure_code),
                    fixture["expected_observed_reason_codes"],
                )
                status, selected_version, candidate_reasons = (
                    self._selection_result(failure_code)
                )
                self.assertEqual(
                    candidate_reasons,
                    fixture["expected_selection_candidate_reasons"],
                )
                if fixture["selection_outcome"] == "last_known_good":
                    self.assertEqual((status, selected_version), ("selected", "1.0"))
                else:
                    self.assertEqual((status, selected_version), ("abstained", None))

                workspace = _PromotionWorkspace(
                    target_channel="Stable", baseline=True
                )
                try:
                    immutable_paths = (
                        "bundle_specs.json",
                        "bundle_lifecycle.jsonl",
                        "support_profiles.jsonl",
                        "candidate_screening.jsonl",
                        "evidence_activation.jsonl",
                    )
                    before = {
                        name: (workspace.data / name).read_bytes()
                        for name in immutable_paths
                    }
                    stable_before = _load_state(workspace.root).lifecycle.channel_pointers[
                        ("example-detector", "Stable")
                    ]["bundle_spec_digest"]
                    failed_report = self._failed_drill_path(workspace, failure_code)
                    outcome = promote_image_pipeline(
                        **workspace.args(
                            target_channel="Stable",
                            approve=True,
                            failure_drill_report_path=failed_report.relative_to(
                                workspace.root
                            ),
                        )
                    )
                    self.assertEqual(outcome.status, "apply_failed")
                    self.assertIn(
                        fixture["expected_promotion_gate"],
                        [gate.code for gate in outcome.gates],
                    )
                    self.assertFalse(outcome.lifecycle_changed)
                    for name, expected in before.items():
                        self.assertEqual((workspace.data / name).read_bytes(), expected)
                    stable_after = _load_state(workspace.root).lifecycle.channel_pointers[
                        ("example-detector", "Stable")
                    ]["bundle_spec_digest"]
                    self.assertEqual(stable_after, stable_before)
                finally:
                    workspace.cleanup()

    def test_reviewed_exact_last_known_good_rollback_succeeds(self) -> None:
        workspace = _LifecycleWorkspace(stable_pointer=True)
        self.addCleanup(workspace.cleanup)
        arguments = _target_rollback_args(workspace)
        immutable = {
            name: (workspace.data / name).read_bytes()
            for name in (
                "bundle_specs.json",
                "support_profiles.jsonl",
                "candidate_screening.jsonl",
                "evidence_activation.jsonl",
            )
        }
        stable_before = _load_state(workspace.root).lifecycle.channel_pointers[
            ("example-detector", "Stable")
        ]["lifecycle_event_digest"]
        outcome = update_image_pipeline_lifecycle(
            **{**arguments, "approve": True}
        )
        self.assertEqual(outcome.status, "applied", outcome.to_dict())
        projected = _load_state(workspace.root).lifecycle
        self.assertEqual(
            projected.channel_pointers[("example-detector", "Experimental")][
                "bundle_spec_digest"
            ],
            workspace.target["spec_digest"],
        )
        self.assertEqual(
            projected.channel_pointers[("example-detector", "Stable")][
                "lifecycle_event_digest"
            ],
            stable_before,
        )
        for name, expected in immutable.items():
            self.assertEqual((workspace.data / name).read_bytes(), expected)


class PromotionCliTests(unittest.TestCase):
    def test_help_and_missing_gate_json(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as help_exit, redirect_stdout(stdout):
            cli_main(["promote-image-pipeline", "--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertIn("--expected-source-pointer-digest", stdout.getvalue())

        workspace = _PromotionWorkspace(target_channel="Experimental")
        self.addCleanup(workspace.cleanup)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(
                ["promote-image-pipeline", "--workspace", str(workspace.root)]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn('"status": "dry_run_blocked"', stdout.getvalue())

    @patch(
        "yolozu.adaptive.promotion.build_screening_eligibility_observation",
        return_value=_ScreeningPass(),
    )
    def test_cli_approve_appends_one_assignment(self, _screening) -> None:
        workspace = _PromotionWorkspace(target_channel="Experimental")
        self.addCleanup(workspace.cleanup)
        profile = workspace.refs[0]
        argv = [
            "promote-image-pipeline",
            "--workspace",
            str(workspace.root),
            "--family-id",
            "example-detector",
            "--source-channel",
            "Candidate",
            "--target-channel",
            "Experimental",
            "--bundle-spec-digest",
            workspace.candidate["spec_digest"],
            "--expected-source-pointer-digest",
            workspace.candidate_pointer["event_digest"],
            "--expected-target-pointer-digest",
            "none",
            "--expected-lifecycle-head-digest",
            workspace.head,
            "--expected-support-profile-index-head",
            workspace.support_head,
            "--expected-profile-set-record-digest",
            workspace.assignment["record_digest"],
            "--expected-profile-set-digest",
            canonical_sha256_v1(workspace.refs),
            "--profile",
            f"{profile['profile_id']}={profile['profile_digest']}",
            "--evidence-bindings",
            str(workspace.bindings.relative_to(workspace.root)),
            "--rollback-target",
            "none",
            "--approver-role-id",
            "repo_maintainer",
            "--public-review-id",
            "gh-promotion-cli",
            "--reason",
            "Review one exact CLI promotion.",
            "--approve",
        ]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli_main(argv)
        self.assertEqual(exit_code, 0)
        self.assertIn('"status": "applied"', stdout.getvalue())
