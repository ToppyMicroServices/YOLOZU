from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import (
    _bundle_payload,
    _lifecycle_event,
    _registry_payload,
    _support_record,
)
from tests.test_adaptive_selector import _environment, _job, _workload
from yolozu.adaptive.bundle_registry import LoadedAlgorithmBundleRegistry
from yolozu.adaptive.bundles import (
    EMPTY_PROFILE_SET_DIGEST,
    ZERO_DIGEST,
    project_bundle_lifecycle,
    project_support_profiles,
    validate_algorithm_bundle_registry,
    validate_algorithm_bundle_spec,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.qualification import QUALIFICATION_PROTOCOL_FINGERPRINT
from yolozu.adaptive.processing import (
    ProcessingError,
    _revalidate_support_profile_before_execution,
)
from yolozu.adaptive.recommendation import _support_observations
from yolozu.adaptive.support_profiles import (
    build_support_profile_eligibility_observation,
    load_support_profile_jsonl_bytes,
    review_image_pipeline_support_profiles,
)


ROOT = Path(__file__).resolve().parents[1]


def _profile(
    profile_id: str = "cpu-batch",
    *,
    environment_fingerprint: str = "1" * 64,
    workload_fingerprint: str = "2" * 64,
    limitation: str = "Exact measured public fixture configuration only.",
) -> dict:
    value = {
        "schema_version": 1,
        "profile_id": profile_id,
        "profile_digest": ZERO_DIGEST,
        "task": "object_detection",
        "environment_fingerprint": environment_fingerprint,
        "qualification_workload_fingerprint": workload_fingerprint,
        "protocol_fingerprint": QUALIFICATION_PROTOCOL_FINGERPRINT,
        "advertised_constraints": {
            "execution_mode": "batch",
            "max_cold_start_ms": "500",
            "max_p95_latency_ms": "50",
            "min_repeat_throughput_fps": "1",
        },
        "public_limitations": [limitation],
    }
    value["profile_digest"] = canonical_sha256_v1(
        value,
        own_digest_field="profile_digest",
    )
    return value


def _proposal(family_id: str, channel: str, *profiles: dict) -> bytes:
    value = {
        "schema_version": 1,
        "family_id": family_id,
        "channel": channel,
        "complete_profile_ids": [item["profile_id"] for item in profiles],
        "profiles": list(profiles),
    }
    return canonical_json_v1(value) + b"\n"


def _workspace() -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    data = root / "yolozu" / "data" / "adaptive_routing"
    data.mkdir(parents=True)
    stream = data / "support_profiles.jsonl"
    stream.write_bytes(b"")
    return temporary, root, stream


def _review(root: Path, *, approve: bool = False, **updates: object):
    arguments: dict[str, object] = {
        "proposal_path": "proposal.json",
        "family_id": "example-detector",
        "channel": "Experimental",
        "workspace_root": root,
        "expected_head_digest": ZERO_DIGEST,
        "expected_current_profile_set_record_digest": None,
        "expected_current_profile_set_digest": None,
        "expect_no_current_profile_set": True,
        "reviewer_role_id": "repo_maintainer",
        "public_review_id": "gh-279",
        "reason": "Review one complete dormant public target scope.",
        "approve": approve,
        "occurred_at": "2026-08-26T00:00:00Z",
    }
    arguments.update(updates)
    return review_image_pipeline_support_profiles(**arguments)


class SupportProfileReviewTests(unittest.TestCase):
    def test_dry_run_is_zero_write_and_approval_appends_exact_ssot(self) -> None:
        temporary, root, stream = _workspace()
        self.addCleanup(temporary.cleanup)
        profile = _profile()
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", profile)
        )
        before = stream.read_bytes()

        dry_run = _review(root)
        self.assertEqual(dry_run.status, "dry_run_ready")
        self.assertEqual(stream.read_bytes(), before)
        self.assertEqual(len(dry_run.planned_records), 2)
        self.assertFalse(dry_run.to_dict()["support_state_changed"])

        applied = _review(root, approve=True)
        self.assertEqual(applied.status, "applied")
        self.assertEqual(len(applied.applied_record_digests), 2)
        projection = load_support_profile_jsonl_bytes(
            stream.read_bytes(),
            source_trust_domain="yolozu_managed",
        )
        assigned = projection.assignments[("example-detector", "Experimental")]
        self.assertEqual(assigned["profiles"][0]["profile_id"], "cpu-batch")
        self.assertEqual(projection.head_digest, applied.observed_head_digest)
        self.assertEqual(list(root.rglob("*.jsonl")), [stream])

    def test_replacement_reuses_immutable_definitions_and_keeps_old_bytes(self) -> None:
        temporary, root, stream = _workspace()
        self.addCleanup(temporary.cleanup)
        first = _profile()
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", first)
        )
        initial = _review(root, approve=True)
        old_bytes = stream.read_bytes()
        second = _profile("cpu-batch-low-latency")
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", first, second)
        )
        replacement = _review(
            root,
            approve=True,
            expected_head_digest=initial.observed_head_digest,
            expected_current_profile_set_record_digest=(
                initial.observed_current_profile_set_record_digest
            ),
            expected_current_profile_set_digest=(
                initial.observed_current_profile_set_digest
            ),
            expect_no_current_profile_set=False,
            occurred_at="2026-08-26T00:01:00Z",
        )
        self.assertEqual(replacement.status, "applied")
        self.assertTrue(stream.read_bytes().startswith(old_bytes))
        self.assertEqual(
            [item["kind"] for item in replacement.planned_records],
            ["profile_definition", "profile_set_assignment"],
        )

        changed = copy.deepcopy(first)
        changed["advertised_constraints"]["max_p95_latency_ms"] = "49"
        changed["profile_digest"] = canonical_sha256_v1(
            changed,
            own_digest_field="profile_digest",
        )
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", changed)
        )
        unchanged = stream.read_bytes()
        rejected = _review(
            root,
            approve=True,
            expected_head_digest=replacement.observed_head_digest,
            expected_current_profile_set_record_digest=(
                replacement.observed_current_profile_set_record_digest
            ),
            expected_current_profile_set_digest=(
                replacement.observed_current_profile_set_digest
            ),
            expect_no_current_profile_set=False,
            occurred_at="2026-08-26T00:02:00Z",
        )
        self.assertIn("profile_id_reused", [item.code for item in rejected.gates])
        self.assertEqual(stream.read_bytes(), unchanged)

    def test_stale_incomplete_private_and_noncanonical_inputs_fail_closed(self) -> None:
        cases = []
        good = _profile()
        incomplete = json.loads(_proposal("example-detector", "Experimental", good))
        incomplete["complete_profile_ids"].append("missing-profile")
        cases.append((canonical_json_v1(incomplete) + b"\n", {}, "proposal_invalid"))

        duplicate = json.loads(_proposal("example-detector", "Experimental", good))
        duplicate["complete_profile_ids"] = ["cpu-batch", "cpu-batch"]
        duplicate["profiles"] = [good, good]
        cases.append((canonical_json_v1(duplicate) + b"\n", {}, "proposal_invalid"))

        private = _profile(limitation="Contact owner@example.com from /Users/owner/data.")
        cases.append(
            (
                _proposal("example-detector", "Experimental", private),
                {},
                "proposal_invalid",
            )
        )
        cases.append(
            (
                json.dumps(
                    json.loads(_proposal("example-detector", "Experimental", good)),
                    indent=2,
                ).encode()
                + b"\n",
                {},
                "proposal_invalid",
            )
        )
        cases.append(
            (
                _proposal("example-detector", "Experimental", good),
                {"expected_head_digest": "9" * 64},
                "stale_head",
            )
        )

        for raw, updates, expected_gate in cases:
            with self.subTest(expected_gate=expected_gate):
                temporary, root, stream = _workspace()
                try:
                    (root / "proposal.json").write_bytes(raw)
                    outcome = _review(root, approve=True, **updates)
                    self.assertEqual(outcome.status, "apply_failed")
                    self.assertIn(expected_gate, [item.code for item in outcome.gates])
                    self.assertEqual(stream.read_bytes(), b"")
                finally:
                    temporary.cleanup()

    def test_write_interruption_before_replace_preserves_stream(self) -> None:
        temporary, root, stream = _workspace()
        self.addCleanup(temporary.cleanup)
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", _profile())
        )

        def fail(step: str) -> None:
            if step == "before_replace":
                raise OSError("injected write interruption")

        outcome = _review(root, approve=True, fault_hook=fail)
        self.assertEqual(outcome.status, "apply_failed")
        self.assertEqual(stream.read_bytes(), b"")
        self.assertFalse(any("stage" in item.name for item in stream.parent.iterdir()))

    def test_readback_mismatch_never_claims_success(self) -> None:
        temporary, root, _stream = _workspace()
        self.addCleanup(temporary.cleanup)
        (root / "proposal.json").write_bytes(
            _proposal("example-detector", "Experimental", _profile())
        )
        from yolozu.adaptive import support_profiles as module

        original = module._read_regular
        calls = 0

        def mismatch(*args: object, **kwargs: object) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 4:
                return b""
            return original(*args, **kwargs)

        with patch.object(module, "_read_regular", side_effect=mismatch):
            outcome = _review(root, approve=True)
        self.assertEqual(outcome.status, "apply_failed")
        self.assertEqual(outcome.applied_record_digests, ())
        self.assertIn("atomic_write_failed", [item.code for item in outcome.gates])

    def test_cli_help_and_dry_run_json(self) -> None:
        help_result = subprocess.run(
            [
                sys.executable,
                "tools/review_image_pipeline_support_profiles.py",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--expect-no-current-profile-set", help_result.stdout)


def _provider_fixture(*, include_newer_dormant_review: bool = False):
    job = _job()
    environment = _environment()
    workload = _workload(job)
    profile = _profile(
        environment_fingerprint=environment.environment_fingerprint,
        workload_fingerprint=workload.workload_fingerprint,
    )
    definition = _support_record(
        sequence=1,
        previous=ZERO_DIGEST,
        record_id="define-cpu-batch",
        kind="profile_definition",
        variant={"profile": profile},
    )
    refs = [
        {
            "profile_id": profile["profile_id"],
            "profile_digest": profile["profile_digest"],
        }
    ]
    assignment = _support_record(
        sequence=2,
        previous=definition["record_digest"],
        record_id="assign-cpu-batch",
        kind="profile_set_assignment",
        variant={
            "family_id": "example-detector",
            "channel": "Experimental",
            "profiles": refs,
            "profile_set_digest": canonical_sha256_v1(refs),
        },
    )
    support_records = [definition, assignment]
    if include_newer_dormant_review:
        later = _support_record(
            sequence=3,
            previous=assignment["record_digest"],
            record_id="assign-later-dormant",
            kind="profile_set_assignment",
            variant={
                "family_id": "example-detector",
                "channel": "Experimental",
                "profiles": refs,
                "profile_set_digest": canonical_sha256_v1(refs),
            },
        )
        support_records.append(later)
    support = project_support_profiles(
        support_records,
        source_trust_domain="yolozu_managed",
    )
    bundle_payload = _bundle_payload()
    bundle = validate_algorithm_bundle_spec(bundle_payload)
    registry = validate_algorithm_bundle_registry(_registry_payload(bundle_payload))
    reviews = [{"artifact_id": "model", "review_state": "approved"}]
    global_event = _lifecycle_event(
        sequence=1,
        previous=ZERO_DIGEST,
        scope="bundle_global",
        event_type="register_global",
        variant={
            "family_id": "example-detector",
            "bundle_spec_digest": bundle.spec_digest,
            "artifact_set_digest": bundle.artifact_set_digest,
            "bundle_state": "enabled",
            "artifact_license_reviews": reviews,
        },
    )
    candidate = _lifecycle_event(
        sequence=2,
        previous=global_event["event_digest"],
        scope="channel_assignment",
        event_type="candidate_registration",
        variant={
            "family_id": "example-detector",
            "channel": "Candidate",
            "target_bundle_spec_digest": bundle.spec_digest,
            "target_artifact_set_digest": bundle.artifact_set_digest,
            "target_artifact_license_reviews": reviews,
            "support_profile_index_head": support.head_digest,
            "profile_set_record_id": None,
            "profile_set_record_digest": None,
            "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
            "profiles": [],
            "evidence_bindings": [],
        },
    )
    public = _lifecycle_event(
        sequence=3,
        previous=candidate["event_digest"],
        scope="channel_assignment",
        event_type="public_assignment",
        variant={
            "family_id": "example-detector",
            "channel": "Experimental",
            "target_bundle_spec_digest": bundle.spec_digest,
            "target_artifact_set_digest": bundle.artifact_set_digest,
            "target_artifact_license_reviews": reviews,
            "support_profile_index_head": support.head_digest,
            "profile_set_record_id": assignment["record_id"],
            "profile_set_record_digest": assignment["record_digest"],
            "profile_set_digest": canonical_sha256_v1(refs),
            "profiles": refs,
            "evidence_bindings": [
                {
                    "profile_id": profile["profile_id"],
                    "profile_digest": profile["profile_digest"],
                    "activation_id": "activation-fixture",
                    "activation_digest": "8" * 64,
                    "trust_domain_claim": "yolozu_managed",
                }
            ],
        },
    )
    lifecycle = project_bundle_lifecycle(
        registry,
        [global_event, candidate, public],
        source_trust_domain="yolozu_managed",
        support_profiles=support,
    )
    loaded = LoadedAlgorithmBundleRegistry(
        registry=registry,
        bundles=(bundle,),
        lifecycle=lifecycle,
        registry_trust_domain="yolozu_managed",
        lifecycle_trust_domain="yolozu_managed",
        source_kind="packaged_ssot",
    )
    return loaded, support, bundle, job, environment, workload, assignment


class SupportProfileProviderTests(unittest.TestCase):
    def test_exact_match_site_behavior_and_historical_snapshot(self) -> None:
        for newer in (False, True):
            with self.subTest(newer_dormant_review=newer):
                registry, support, bundle, job, environment, workload, assignment = (
                    _provider_fixture(include_newer_dormant_review=newer)
                )
                observed = build_support_profile_eligibility_observation(
                    registry=registry,
                    profiles=support,
                    bundle=bundle,
                    channel="Experimental",
                    job=job,
                    environment=environment,
                    workload=workload,
                    evidence_trust_domain="yolozu_managed",
                    support_scope="public_qualified",
                )
                self.assertIsNotNone(observed)
                self.assertEqual(observed.to_dict()["status"], "matching_one")
                self.assertEqual(
                    observed.to_dict()["profile_set_record_digest"],
                    assignment["record_digest"],
                )
                site = build_support_profile_eligibility_observation(
                    registry=registry,
                    profiles=support,
                    bundle=bundle,
                    channel="Experimental",
                    job=job,
                    environment=environment,
                    workload=workload,
                    evidence_trust_domain="site_managed",
                    support_scope="site_qualified",
                )
                self.assertEqual(site.to_dict()["status"], "not_required_site")

                with patch(
                    "yolozu.adaptive.recommendation._evidence_trust_for_bundle",
                    return_value=("yolozu_managed", "public_qualified"),
                ):
                    integrated = _support_observations(
                        registry=registry,
                        profiles=support,
                        job=job,
                        environment=environment,
                        workload=workload,
                        evidence={},
                    )
                self.assertEqual(
                    integrated[(bundle.spec_digest, "Experimental")].to_dict(),
                    observed.to_dict(),
                )

    def test_no_match_untrusted_absent_and_conflict_fail_closed(self) -> None:
        registry, support, bundle, job, environment, workload, _assignment = (
            _provider_fixture()
        )
        different_job = _job(max_p95_latency_ms="49")
        no_match = build_support_profile_eligibility_observation(
            registry=registry,
            profiles=support,
            bundle=bundle,
            channel="Experimental",
            job=different_job,
            environment=environment,
            workload=_workload(different_job),
            evidence_trust_domain="yolozu_managed",
            support_scope="public_qualified",
        )
        self.assertEqual(no_match.to_dict()["status"], "no_match")

        raw = b"".join(
            canonical_json_v1(item.to_dict()) + b"\n"
            for item in support.record_by_digest.values()
        )
        untrusted_support = load_support_profile_jsonl_bytes(
            raw,
            source_trust_domain="operator_asserted",
        )
        untrusted = build_support_profile_eligibility_observation(
            registry=registry,
            profiles=untrusted_support,
            bundle=bundle,
            channel="Experimental",
            job=job,
            environment=environment,
            workload=workload,
            evidence_trust_domain="yolozu_managed",
            support_scope="public_qualified",
        )
        self.assertEqual(untrusted.to_dict()["status"], "untrusted")

        pointer = registry.lifecycle.channel_pointers[("example-detector", "Experimental")]
        assert pointer is not None
        original = pointer["profile_set_record_digest"]
        pointer["profile_set_record_digest"] = "9" * 64
        absent = build_support_profile_eligibility_observation(
            registry=registry,
            profiles=support,
            bundle=bundle,
            channel="Experimental",
            job=job,
            environment=environment,
            workload=workload,
            evidence_trust_domain="yolozu_managed",
            support_scope="public_qualified",
        )
        self.assertEqual(absent.to_dict()["status"], "absent")
        pointer["profile_set_record_digest"] = original
        pointer["profile_set_digest"] = "7" * 64
        conflict = build_support_profile_eligibility_observation(
            registry=registry,
            profiles=support,
            bundle=bundle,
            channel="Experimental",
            job=job,
            environment=environment,
            workload=workload,
            evidence_trust_domain="yolozu_managed",
            support_scope="public_qualified",
        )
        self.assertEqual(conflict.to_dict()["status"], "conflict")

    def test_execution_preflight_rejects_support_pointer_tamper(self) -> None:
        registry, support, bundle, job, environment, workload, _assignment = (
            _provider_fixture()
        )
        observed = build_support_profile_eligibility_observation(
            registry=registry,
            profiles=support,
            bundle=bundle,
            channel="Experimental",
            job=job,
            environment=environment,
            workload=workload,
            evidence_trust_domain="yolozu_managed",
            support_scope="public_qualified",
        )
        assert observed is not None
        selected = {"spec_digest": bundle.spec_digest}
        pinned = {
            "registry_digest": registry.registry.registry_digest,
            "lifecycle_projection_digest": registry.lifecycle.head_digest,
        }
        evaluation = {
            "effective_channel": "Experimental",
            "evidence": {"trust_domain": "yolozu_managed"},
            "support_scope": "public_qualified",
            "support_profile_observation": observed.to_dict(),
        }
        route = object()
        with (
            patch(
                "yolozu.adaptive.processing._load_support_profiles",
                return_value=support,
            ),
            patch(
                "yolozu.adaptive.processing.load_algorithm_bundle_registry",
                return_value=registry,
            ),
            patch(
                "yolozu.adaptive.processing._resolve_execution_route",
                return_value=route,
            ),
        ):
            current_bundle, current_route = (
                _revalidate_support_profile_before_execution(
                    selected=selected,
                    pinned_record=pinned,
                    selected_evaluation=evaluation,
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    workload=workload,
                )
            )
            self.assertEqual(current_bundle.spec_digest, bundle.spec_digest)
            self.assertIs(current_route, route)

            pointer = registry.lifecycle.channel_pointers[
                ("example-detector", "Experimental")
            ]
            assert pointer is not None
            pointer["profile_set_digest"] = "7" * 64
            with self.assertRaises(ProcessingError) as rejected:
                _revalidate_support_profile_before_execution(
                    selected=selected,
                    pinned_record=pinned,
                    selected_evaluation=evaluation,
                    bundle=bundle,
                    job=job,
                    environment=environment,
                    workload=workload,
                )
            self.assertEqual(rejected.exception.code, "selection_stale")

    def test_projection_rejects_chain_gap_and_changed_definition(self) -> None:
        _registry, support, _bundle, _job_value, _environment_value, _workload_value, _ = (
            _provider_fixture()
        )
        records = [item.to_dict() for item in support.record_by_digest.values()]
        gap = copy.deepcopy(records)
        gap[1]["sequence"] = 3
        gap[1]["record_digest"] = canonical_sha256_v1(
            gap[1],
            own_digest_field="record_digest",
        )
        with self.assertRaises(ValueError):
            project_support_profiles(gap, source_trust_domain="yolozu_managed")
