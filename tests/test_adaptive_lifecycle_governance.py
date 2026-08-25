from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.test_adaptive_bundle_contracts import (
    _bundle_payload,
    _lifecycle_event,
    _registry_payload,
    _support_record,
)
from tests.test_adaptive_evidence_contracts import (
    _activation,
    _redigest_report,
    _report_payload,
)
from tests.test_adaptive_selector import _context, _job, _select
from tests.test_adaptive_support_profile_governance import _profile
from yolozu.adaptive.bundles import (
    EMPTY_PROFILE_SET_DIGEST,
    ZERO_DIGEST,
    project_bundle_lifecycle,
    validate_bundle_lifecycle_record,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.lifecycle import _load_state, update_image_pipeline_lifecycle
from yolozu.cli_entry import main as cli_main


def _jsonl(*records: dict) -> bytes:
    return b"".join(canonical_json_v1(record) + b"\n" for record in records)


class _Workspace:
    def __init__(
        self,
        *,
        stable_pointer: bool = False,
        target_family: str = "example-detector",
        target_was_assigned: bool = True,
    ) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "yolozu" / "data" / "adaptive_routing"
        self.data.mkdir(parents=True)

        self.current = _bundle_payload(data=b"current", version="2.0")
        self.target = _bundle_payload(data=b"target", version="1.0")
        if target_family != "example-detector":
            self.target["family_id"] = target_family
            self.target["bundle_id"] = f"{target_family}-onnx"
            self.target["spec_digest"] = canonical_sha256_v1(
                self.target,
                own_digest_field="spec_digest",
            )
        registry = _registry_payload(self.current, self.target)
        (self.data / "bundle_specs.json").write_bytes(
            canonical_json_v1(registry) + b"\n"
        )

        profile = _profile()
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
        self.profile = profile
        self.refs = refs
        self.support_assignment = assignment
        self.experimental_support_head = assignment["record_digest"]
        support_records = [definition, assignment]
        if stable_pointer:
            stable_assignment = _support_record(
                sequence=3,
                previous=assignment["record_digest"],
                record_id="assign-cpu-batch-stable",
                kind="profile_set_assignment",
                variant={
                    "family_id": "example-detector",
                    "channel": "Stable",
                    "profiles": refs,
                    "profile_set_digest": canonical_sha256_v1(refs),
                },
            )
            support_records.append(stable_assignment)
        else:
            stable_assignment = None
        self.support_head = support_records[-1]["record_digest"]
        (self.data / "support_profiles.jsonl").write_bytes(_jsonl(*support_records))

        current_reviews = [{"artifact_id": "model", "review_state": "approved"}]
        target_reviews = [{"artifact_id": "model", "review_state": "approved"}]
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
                "bundle_spec_digest": self.current["spec_digest"],
                "artifact_set_digest": self.current["artifact_set_digest"],
                "bundle_state": "enabled",
                "artifact_license_reviews": current_reviews,
            },
        )
        add(
            "channel_assignment",
            "candidate_registration",
            {
                "family_id": "example-detector",
                "channel": "Candidate",
                "target_bundle_spec_digest": self.current["spec_digest"],
                "target_artifact_set_digest": self.current["artifact_set_digest"],
                "target_artifact_license_reviews": current_reviews,
                "support_profile_index_head": self.experimental_support_head,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            },
        )
        self.target_global = add(
            "bundle_global",
            "register_global",
            {
                "family_id": target_family,
                "bundle_spec_digest": self.target["spec_digest"],
                "artifact_set_digest": self.target["artifact_set_digest"],
                "bundle_state": "enabled",
                "artifact_license_reviews": target_reviews,
            },
        )
        add(
            "channel_assignment",
            "candidate_registration",
            {
                "family_id": target_family,
                "channel": "Candidate",
                "target_bundle_spec_digest": self.target["spec_digest"],
                "target_artifact_set_digest": self.target["artifact_set_digest"],
                "target_artifact_license_reviews": target_reviews,
                "support_profile_index_head": self.support_head,
                "profile_set_record_id": None,
                "profile_set_record_digest": None,
                "profile_set_digest": EMPTY_PROFILE_SET_DIGEST,
                "profiles": [],
                "evidence_bindings": [],
            },
        )
        self.target_experimental = None
        if target_was_assigned:
            self.target_experimental = add(
                "channel_assignment",
                "public_assignment",
                {
                    "family_id": target_family,
                    "channel": "Experimental",
                    "target_bundle_spec_digest": self.target["spec_digest"],
                    "target_artifact_set_digest": self.target[
                        "artifact_set_digest"
                    ],
                    "target_artifact_license_reviews": target_reviews,
                    "support_profile_index_head": self.experimental_support_head,
                    "profile_set_record_id": assignment["record_id"],
                    "profile_set_record_digest": assignment["record_digest"],
                    "profile_set_digest": canonical_sha256_v1(refs),
                    "profiles": refs,
                    "evidence_bindings": [
                        {
                            "profile_id": profile["profile_id"],
                            "profile_digest": profile["profile_digest"],
                            "activation_id": "target-historical-activation",
                            "activation_digest": "7" * 64,
                            "trust_domain_claim": "yolozu_managed",
                        }
                    ],
                },
            )
        self.experimental = add(
            "channel_assignment",
            "public_assignment",
            {
                "family_id": "example-detector",
                "channel": "Experimental",
                "target_bundle_spec_digest": self.current["spec_digest"],
                "target_artifact_set_digest": self.current["artifact_set_digest"],
                "target_artifact_license_reviews": current_reviews,
                "support_profile_index_head": self.experimental_support_head,
                "profile_set_record_id": assignment["record_id"],
                "profile_set_record_digest": assignment["record_digest"],
                "profile_set_digest": canonical_sha256_v1(refs),
                "profiles": refs,
                "evidence_bindings": [
                    {
                        "profile_id": profile["profile_id"],
                        "profile_digest": profile["profile_digest"],
                        "activation_id": "current-activation",
                        "activation_digest": "8" * 64,
                        "trust_domain_claim": "yolozu_managed",
                    }
                ],
            },
        )
        if stable_pointer:
            assert stable_assignment is not None
            self.stable = add(
                "channel_assignment",
                "public_assignment",
                {
                    **{
                        key: copy.deepcopy(value)
                        for key, value in self.experimental.items()
                        if key
                        not in {
                            "sequence",
                            "previous_event_digest",
                            "event_digest",
                            "occurred_at",
                        }
                    },
                    "channel": "Stable",
                    "support_profile_index_head": self.support_head,
                    "profile_set_record_id": stable_assignment["record_id"],
                    "profile_set_record_digest": stable_assignment["record_digest"],
                },
            )
        else:
            self.stable = None
        self.events = events
        self.lifecycle = self.data / "bundle_lifecycle.jsonl"
        self.lifecycle.write_bytes(_jsonl(*events))
        (self.data / "candidate_screening.jsonl").write_bytes(b"")
        (self.data / "evidence_activation.jsonl").write_bytes(b"")
        (self.data / "qualification_reports").mkdir()

    @property
    def head(self) -> str:
        return self.events[-1]["event_digest"]

    def cleanup(self) -> None:
        self.temporary.cleanup()


def _global_args(workspace: _Workspace, operation: str, **updates: object) -> dict:
    state_event = workspace.events[0]["event_digest"]
    values: dict[str, object] = {
        "operation": operation,
        "workspace_root": workspace.root,
        "family_id": "example-detector",
        "bundle_spec_digest": workspace.current["spec_digest"],
        "artifact_set_digest": workspace.current["artifact_set_digest"],
        "expected_lifecycle_head_digest": workspace.head,
        "expected_bundle_state_event_digest": state_event,
        "actor_role_id": "repo_maintainer",
        "public_review_id": "gh-rollback-review",
        "review_status": "approved",
        "reason": "Reviewed lifecycle maintenance without changing immutable assets.",
        "occurred_at": "2026-08-26T01:00:00Z",
    }
    values.update(updates)
    return values


def _none_rollback_args(workspace: _Workspace, **updates: object) -> dict:
    values: dict[str, object] = {
        "operation": "rollback-channel",
        "workspace_root": workspace.root,
        "family_id": "example-detector",
        "bundle_spec_digest": workspace.current["spec_digest"],
        "artifact_set_digest": workspace.current["artifact_set_digest"],
        "expected_lifecycle_head_digest": workspace.head,
        "channel": "Experimental",
        "expected_current_pointer_digest": workspace.experimental["event_digest"],
        "expected_prior_assignment_digest": workspace.experimental["event_digest"],
        "expected_support_profile_index_head": workspace.support_head,
        "expected_prior_support_profile_index_head": workspace.experimental_support_head,
        "expected_prior_profile_set_record_digest": workspace.support_assignment[
            "record_digest"
        ],
        "expected_prior_profile_set_digest": canonical_sha256_v1(workspace.refs),
        "target_bundle_spec_digest": "none",
        "actor_role_id": "repo_maintainer",
        "public_review_id": "gh-rollback-review",
        "review_status": "approved",
        "reason": "Explicitly restore the abstaining channel state.",
        "occurred_at": "2026-08-26T01:00:00Z",
    }
    values.update(updates)
    return values


def _install_target_evidence(
    workspace: _Workspace,
    *,
    environment_fingerprint: str | None = None,
) -> tuple[Path, dict]:
    report = _report_payload(report_id="target-profile-report")
    report["bundle_spec_digest"] = workspace.target["spec_digest"]
    report["artifact_set_digest"] = workspace.target["artifact_set_digest"]
    report["environment_fingerprint"] = (
        workspace.profile["environment_fingerprint"]
        if environment_fingerprint is None
        else environment_fingerprint
    )
    report["qualification_workload_fingerprint"] = workspace.profile[
        "qualification_workload_fingerprint"
    ]
    report["protocol_fingerprint"] = workspace.profile["protocol_fingerprint"]
    _redigest_report(report)
    activation = _activation(report)
    activation["event_id"] = "target-profile-activation"
    activation["event_digest"] = canonical_sha256_v1(
        activation,
        own_digest_field="event_digest",
    )
    (workspace.data / "evidence_activation.jsonl").write_bytes(_jsonl(activation))
    report_root = workspace.data / "qualification_reports" / report["report_id"]
    report_root.mkdir()
    (report_root / "qualification_report.json").write_bytes(
        canonical_json_v1(report) + b"\n"
    )
    binding = {
        "profile_id": workspace.profile["profile_id"],
        "profile_digest": workspace.profile["profile_digest"],
        "activation_id": activation["event_id"],
        "activation_digest": activation["event_digest"],
        "trust_domain_claim": "yolozu_managed",
    }
    proposal = workspace.root / "rollback-bindings.json"
    proposal.write_bytes(
        canonical_json_v1({"schema_version": 1, "bindings": [binding]}) + b"\n"
    )
    return proposal, binding


def _target_rollback_args(workspace: _Workspace, **updates: object) -> dict:
    proposal, _binding = _install_target_evidence(workspace)
    values = _none_rollback_args(
        workspace,
        target_bundle_spec_digest=workspace.target["spec_digest"],
        target_artifact_set_digest=workspace.target["artifact_set_digest"],
        evidence_bindings_path=proposal.relative_to(workspace.root),
    )
    values.update(updates)
    return values


class LifecycleGlobalOperationTests(unittest.TestCase):
    def test_maintenance_record_requires_approved_repository_audit_fields(
        self,
    ) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        outcome = update_image_pipeline_lifecycle(
            **_global_args(workspace, "disable")
        )
        self.assertEqual(outcome.status, "dry_run_ready")
        base = outcome.planned_records[0]
        registry = _load_state(workspace.root).registry
        cases = {
            "unapproved": {"review_status": "pending"},
            "non_repository_actor": {
                "actor_role_id": "automation",
                "reviewer_role_id": "automation",
            },
            "site_review": {
                "review_reference": {
                    "kind": "site_local_status",
                    "status": "present",
                }
            },
            "site_source": {"issuer_claim": "site_source"},
        }
        for name, updates in cases.items():
            with self.subTest(name=name):
                invalid = {**base, **updates}
                invalid["event_digest"] = canonical_sha256_v1(
                    invalid,
                    own_digest_field="event_digest",
                )
                with self.assertRaises(ValueError):
                    validate_bundle_lifecycle_record(
                        invalid,
                        registry=registry,
                        source_trust_domain="yolozu_managed",
                    )

        missing_audit = dict(base)
        del missing_audit["artifact_members"]
        del missing_audit["existing_runs_reproducible"]
        missing_audit["event_digest"] = canonical_sha256_v1(
            missing_audit,
            own_digest_field="event_digest",
        )
        with self.assertRaises(ValueError):
            validate_bundle_lifecycle_record(
                missing_audit,
                registry=registry,
                source_trust_domain="yolozu_managed",
            )

    def test_dry_run_then_disable_preserves_immutable_inputs(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        registry_before = (workspace.data / "bundle_specs.json").read_bytes()
        support_before = (workspace.data / "support_profiles.jsonl").read_bytes()

        dry_run = update_image_pipeline_lifecycle(
            **_global_args(workspace, "disable")
        )
        self.assertEqual(dry_run.status, "dry_run_ready")
        self.assertEqual(workspace.lifecycle.read_bytes(), before)
        self.assertEqual(dry_run.planned_records[0]["maintenance_operation"], "disable")
        self.assertTrue(dry_run.planned_records[0]["existing_runs_reproducible"])
        self.assertEqual(
            dry_run.planned_records[0]["artifact_members"][0]["sha256"],
            workspace.current["artifacts"][0]["sha256"],
        )

        applied = update_image_pipeline_lifecycle(
            **_global_args(workspace, "disable", approve=True)
        )
        self.assertEqual(applied.status, "applied")
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))
        self.assertEqual(
            (workspace.data / "bundle_specs.json").read_bytes(), registry_before
        )
        self.assertEqual(
            (workspace.data / "support_profiles.jsonl").read_bytes(), support_before
        )
        self.assertFalse(applied.to_dict()["artifacts_changed"])

    def test_stale_review_license_and_terminal_revoke_fail_closed(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        stale = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "disable",
                expected_lifecycle_head_digest="9" * 64,
                approve=True,
            )
        )
        self.assertIn("stale_lifecycle_head", [gate.code for gate in stale.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        unchanged = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "review-license",
                license_reviews=[
                    {"artifact_id": "model", "review_state": "approved"}
                ],
                approve=True,
            )
        )
        self.assertIn("license_review_unchanged", [gate.code for gate in unchanged.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        revoked = update_image_pipeline_lifecycle(
            **_global_args(workspace, "revoke", approve=True)
        )
        self.assertEqual(revoked.status, "applied")
        after_revoke = workspace.lifecycle.read_bytes()
        illegal = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "enable",
                expected_lifecycle_head_digest=revoked.observed_lifecycle_head_digest,
                expected_bundle_state_event_digest=revoked.applied_record_digests[0],
                approve=True,
            )
        )
        self.assertIn("revoked_terminal", [gate.code for gate in illegal.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), after_revoke)

    def test_interrupted_write_claims_no_success(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()

        def fail(step: str) -> None:
            if step == "before_replace":
                raise OSError("injected interruption")

        outcome = update_image_pipeline_lifecycle(
            **_global_args(workspace, "disable", approve=True, fault_hook=fail)
        )
        self.assertEqual(outcome.status, "apply_failed")
        self.assertEqual(outcome.applied_record_digests, ())
        self.assertIsNone(outcome.lifecycle_changed)
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

    def test_review_license_disable_and_enable_append_versioned_history(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        reviewed = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "review-license",
                license_reviews=[
                    {"artifact_id": "model", "review_state": "blocked"}
                ],
                approve=True,
            )
        )
        self.assertEqual(reviewed.status, "applied", reviewed.to_dict())
        disabled = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "disable",
                expected_lifecycle_head_digest=reviewed.observed_lifecycle_head_digest,
                expected_bundle_state_event_digest=reviewed.applied_record_digests[0],
                approve=True,
            )
        )
        self.assertEqual(disabled.status, "applied", disabled.to_dict())
        enabled = update_image_pipeline_lifecycle(
            **_global_args(
                workspace,
                "enable",
                expected_lifecycle_head_digest=disabled.observed_lifecycle_head_digest,
                expected_bundle_state_event_digest=disabled.applied_record_digests[0],
                approve=True,
            )
        )
        self.assertEqual(enabled.status, "applied", enabled.to_dict())
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))
        state = _load_state(workspace.root).lifecycle.bundle_states[
            workspace.current["spec_digest"]
        ]
        self.assertEqual(state["bundle_state"], "enabled")
        self.assertEqual(
            state["artifact_license_reviews"],
            [{"artifact_id": "model", "review_state": "blocked"}],
        )

    def test_readback_failure_never_reports_success(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        original = _load_state
        calls = 0

        def fail_readback(root: Path):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected readback failure")
            return original(root)

        with patch("yolozu.adaptive.lifecycle._load_state", side_effect=fail_readback):
            outcome = update_image_pipeline_lifecycle(
                **_global_args(workspace, "disable", approve=True)
            )
        self.assertEqual(outcome.status, "apply_failed")
        self.assertEqual(outcome.applied_record_digests, ())
        self.assertIsNone(outcome.lifecycle_changed)
        self.assertIn("atomic_write_failed", [gate.code for gate in outcome.gates])

    def test_global_revoke_excludes_both_channel_pointers_without_deletion(self) -> None:
        workspace = _Workspace(stable_pointer=True)
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        revoked = update_image_pipeline_lifecycle(
            **_global_args(workspace, "revoke", approve=True)
        )
        self.assertEqual(revoked.status, "applied", revoked.to_dict())
        projected = _load_state(workspace.root).lifecycle
        self.assertFalse(
            projected.is_lifecycle_eligible(
                family_id="example-detector",
                channel="Experimental",
            )
        )
        self.assertFalse(
            projected.is_lifecycle_eligible(
                family_id="example-detector",
                channel="Stable",
            )
        )
        self.assertEqual(
            projected.channel_pointers[("example-detector", "Experimental")][
                "bundle_spec_digest"
            ],
            workspace.current["spec_digest"],
        )
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))


class LifecycleRollbackTests(unittest.TestCase):
    def test_tighter_future_job_abstains_without_pointer_mutation(self) -> None:
        bundle = _bundle_payload()
        selected = _select(_context(bundle)).to_dict()
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(
            selected["selected_bundle"],
            {
                "family_id": bundle["family_id"],
                "bundle_id": bundle["bundle_id"],
                "bundle_version": bundle["bundle_version"],
                "spec_digest": bundle["spec_digest"],
                "artifact_set_digest": bundle["artifact_set_digest"],
                "effective_channel": "Experimental",
            },
        )

        tighter = _context(bundle, job=_job(max_p95_latency_ms="1"))
        pointer_before = copy.deepcopy(tighter.registry.lifecycle.channel_pointers)
        abstained = _select(tighter).to_dict()
        self.assertEqual(abstained["status"], "abstained")
        self.assertEqual(abstained["reason_codes"], ["no_eligible_candidate"])
        self.assertEqual(
            abstained["candidate_evaluations"][0]["reason_codes"],
            ["p95_latency_gate_failed"],
        )
        self.assertEqual(abstained["selected_bundle"], None)
        self.assertEqual(
            tighter.registry.lifecycle.channel_pointers,
            pointer_before,
        )

    def test_rollback_to_none_is_explicit_and_preserves_other_channel(self) -> None:
        workspace = _Workspace(stable_pointer=True)
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        dry_run = update_image_pipeline_lifecycle(**_none_rollback_args(workspace))
        self.assertEqual(dry_run.status, "dry_run_ready")
        planned = dry_run.planned_records[0]
        self.assertEqual(planned["event_scope"], "channel_none")
        self.assertEqual(planned["profiles"], [])
        self.assertEqual(planned["evidence_bindings"], [])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        applied = update_image_pipeline_lifecycle(
            **_none_rollback_args(workspace, approve=True)
        )
        self.assertEqual(applied.status, "applied")
        self.assertEqual(applied.observed_channel_pointer_digest, planned["event_digest"])
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))
        projected = _load_state(workspace.root).lifecycle
        self.assertIsNone(
            projected.channel_pointers[("example-detector", "Experimental")]
        )
        self.assertEqual(
            projected.channel_pointers[("example-detector", "Stable")][
                "bundle_spec_digest"
            ],
            workspace.current["spec_digest"],
        )

    def test_stale_snapshot_subset_arguments_and_cross_family_fail_closed(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        stale = update_image_pipeline_lifecycle(
            **_none_rollback_args(
                workspace,
                expected_prior_profile_set_digest="9" * 64,
                approve=True,
            )
        )
        self.assertIn("stale_prior_set", [gate.code for gate in stale.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        forbidden = update_image_pipeline_lifecycle(
            **_none_rollback_args(
                workspace,
                target_artifact_set_digest=workspace.target["artifact_set_digest"],
                approve=True,
            )
        )
        self.assertIn("none_target_arguments", [gate.code for gate in forbidden.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        cross_family = _Workspace(
            target_family="other-detector",
            target_was_assigned=False,
        )
        self.addCleanup(cross_family.cleanup)
        proposal, _ = _install_target_evidence(cross_family)
        rejected = update_image_pipeline_lifecycle(
            **{
                **_none_rollback_args(
                    cross_family,
                    target_bundle_spec_digest=cross_family.target["spec_digest"],
                    target_artifact_set_digest=cross_family.target[
                        "artifact_set_digest"
                    ],
                    evidence_bindings_path=proposal.relative_to(cross_family.root),
                ),
                "approve": True,
            }
        )
        self.assertIn("cross_family_target", [gate.code for gate in rejected.gates])
        self.assertEqual(
            cross_family.lifecycle.read_bytes(),
            _jsonl(*cross_family.events),
        )

        never_assigned = _Workspace(target_was_assigned=False)
        self.addCleanup(never_assigned.cleanup)
        proposal, _ = _install_target_evidence(never_assigned)
        rejected = update_image_pipeline_lifecycle(
            **{
                **_none_rollback_args(
                    never_assigned,
                    target_bundle_spec_digest=never_assigned.target["spec_digest"],
                    target_artifact_set_digest=never_assigned.target[
                        "artifact_set_digest"
                    ],
                    evidence_bindings_path=proposal.relative_to(
                        never_assigned.root
                    ),
                ),
                "approve": True,
            }
        )
        self.assertIn(
            "target_not_prior_assignment",
            [gate.code for gate in rejected.gates],
        )
        self.assertEqual(
            never_assigned.lifecycle.read_bytes(),
            _jsonl(*never_assigned.events),
        )

    def test_exact_last_known_good_binding_reassigns_only_one_pointer(self) -> None:
        workspace = _Workspace(stable_pointer=True)
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        arguments = _target_rollback_args(workspace)
        dry_run = update_image_pipeline_lifecycle(**arguments)
        self.assertEqual(dry_run.status, "dry_run_ready", dry_run.to_dict())
        planned = dry_run.planned_records[0]
        self.assertEqual(planned["target_bundle_spec_digest"], workspace.target["spec_digest"])
        self.assertEqual(planned["profiles"], workspace.refs)
        self.assertEqual(len(planned["evidence_bindings"]), 1)
        self.assertEqual(
            planned["profile_set_record_digest"],
            workspace.support_assignment["record_digest"],
        )
        self.assertEqual(planned["support_profile_index_head"], workspace.support_head)
        self.assertEqual(
            planned["rollback_target_prior_assignment_digest"],
            workspace.target_experimental["event_digest"],
        )
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        forged = {**planned, "rollback_target_prior_assignment_digest": "9" * 64}
        forged["event_digest"] = canonical_sha256_v1(
            forged,
            own_digest_field="event_digest",
        )
        loaded = _load_state(workspace.root)
        with self.assertRaises(ValueError):
            project_bundle_lifecycle(
                loaded.registry,
                [*workspace.events, forged],
                source_trust_domain="yolozu_managed",
                support_profiles=loaded.support_profiles,
            )

        applied = update_image_pipeline_lifecycle(**{**arguments, "approve": True})
        self.assertEqual(applied.status, "applied", applied.to_dict())
        self.assertTrue(workspace.lifecycle.read_bytes().startswith(before))
        self.assertEqual(applied.target_bundle_spec_digest, workspace.target["spec_digest"])
        projected = _load_state(workspace.root).lifecycle
        self.assertEqual(
            projected.channel_pointers[("example-detector", "Experimental")][
                "bundle_spec_digest"
            ],
            workspace.target["spec_digest"],
        )
        self.assertEqual(
            projected.bundle_states[workspace.current["spec_digest"]]["bundle_state"],
            "enabled",
        )

    def test_evidence_is_revalidated_immediately_before_append(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        arguments = _target_rollback_args(workspace)
        from yolozu.adaptive import lifecycle as lifecycle_module

        original = lifecycle_module._load_evidence
        calls = 0

        def change_before_append(state, *, as_of):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("evidence changed before append")
            return original(state, as_of=as_of)

        with patch(
            "yolozu.adaptive.lifecycle._load_evidence",
            side_effect=change_before_append,
        ):
            outcome = update_image_pipeline_lifecycle(
                **{**arguments, "approve": True}
            )
        self.assertEqual(outcome.status, "apply_failed")
        self.assertIsNone(outcome.lifecycle_changed)
        self.assertIn("atomic_write_failed", [gate.code for gate in outcome.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

    def test_missing_extra_cross_environment_and_untrusted_bindings_fail_closed(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        arguments = _target_rollback_args(workspace)
        proposal = workspace.root / Path(arguments["evidence_bindings_path"])
        payload = load_json(proposal)
        payload["bindings"].append(copy.deepcopy(payload["bindings"][0]))
        proposal.write_bytes(canonical_json_v1(payload) + b"\n")
        duplicate = update_image_pipeline_lifecycle(**{**arguments, "approve": True})
        self.assertIn("evidence_bindings_mismatch", [gate.code for gate in duplicate.gates])
        self.assertEqual(workspace.lifecycle.read_bytes(), before)

        other = _Workspace()
        self.addCleanup(other.cleanup)
        proposal, _ = _install_target_evidence(
            other,
            environment_fingerprint="9" * 64,
        )
        cross_environment = update_image_pipeline_lifecycle(
            **{
                **_none_rollback_args(
                    other,
                    target_bundle_spec_digest=other.target["spec_digest"],
                    target_artifact_set_digest=other.target["artifact_set_digest"],
                    evidence_bindings_path=proposal.relative_to(other.root),
                ),
                "approve": True,
            }
        )
        self.assertIn(
            "target_evidence_invalid",
            [gate.code for gate in cross_environment.gates],
        )
        self.assertEqual(other.lifecycle.read_bytes(), _jsonl(*other.events))

        untrusted = _Workspace()
        self.addCleanup(untrusted.cleanup)
        untrusted_args = _target_rollback_args(untrusted)
        untrusted_proposal = untrusted.root / Path(
            untrusted_args["evidence_bindings_path"]
        )
        untrusted_payload = load_json(untrusted_proposal)
        untrusted_payload["bindings"][0]["trust_domain_claim"] = "site_managed"
        untrusted_proposal.write_bytes(
            canonical_json_v1(untrusted_payload) + b"\n"
        )
        rejected = update_image_pipeline_lifecycle(
            **{**untrusted_args, "approve": True}
        )
        self.assertIn(
            "evidence_bindings_invalid",
            [gate.code for gate in rejected.gates],
        )
        self.assertEqual(untrusted.lifecycle.read_bytes(), _jsonl(*untrusted.events))

    def test_site_operator_expired_and_revoked_evidence_fail_closed(self) -> None:
        for trust_domain, issuer_claim in (
            ("site_managed", "site_source"),
            ("operator_asserted", "operator_source"),
        ):
            with self.subTest(trust_domain=trust_domain):
                workspace = _Workspace()
                self.addCleanup(workspace.cleanup)
                arguments = _target_rollback_args(workspace)
                activation = load_json(workspace.data / "evidence_activation.jsonl")
                activation.update(
                    {
                        "reviewer_role_id": "site_operator",
                        "review_reference": {
                            "kind": "site_local_status",
                            "status": "present",
                        },
                        "issuer_claim": issuer_claim,
                        "trust_domain": trust_domain,
                    }
                )
                activation["event_digest"] = canonical_sha256_v1(
                    activation,
                    own_digest_field="event_digest",
                )
                (workspace.data / "evidence_activation.jsonl").write_bytes(
                    _jsonl(activation)
                )
                rejected = update_image_pipeline_lifecycle(
                    **{**arguments, "approve": True}
                )
                self.assertIn(
                    "target_evidence_invalid",
                    [gate.code for gate in rejected.gates],
                )
                self.assertEqual(
                    workspace.lifecycle.read_bytes(),
                    _jsonl(*workspace.events),
                )

        expired = _Workspace()
        self.addCleanup(expired.cleanup)
        expired_arguments = _target_rollback_args(expired)
        rejected = update_image_pipeline_lifecycle(
            **{
                **expired_arguments,
                "occurred_at": "2027-01-01T00:00:00Z",
                "approve": True,
            }
        )
        self.assertIn(
            "target_evidence_invalid",
            [gate.code for gate in rejected.gates],
        )
        self.assertEqual(expired.lifecycle.read_bytes(), _jsonl(*expired.events))

        revoked = _Workspace()
        self.addCleanup(revoked.cleanup)
        revoked_arguments = _target_rollback_args(revoked)
        active = load_json(revoked.data / "evidence_activation.jsonl")
        report = load_json(
            revoked.data
            / "qualification_reports"
            / active["report_id"]
            / "qualification_report.json"
        )
        terminal = _activation(
            report,
            sequence=2,
            previous=active["event_digest"],
            state="revoked",
        )
        (revoked.data / "evidence_activation.jsonl").write_bytes(
            _jsonl(active, terminal)
        )
        rejected = update_image_pipeline_lifecycle(
            **{**revoked_arguments, "approve": True}
        )
        self.assertIn(
            "target_evidence_invalid",
            [gate.code for gate in rejected.gates],
        )
        self.assertEqual(revoked.lifecycle.read_bytes(), _jsonl(*revoked.events))


class LifecycleCliTests(unittest.TestCase):
    def test_cli_dry_run_emits_json_and_writes_nothing(self) -> None:
        workspace = _Workspace()
        self.addCleanup(workspace.cleanup)
        before = workspace.lifecycle.read_bytes()
        output = io.StringIO()
        with redirect_stdout(output):
            status = cli_main(
                [
                    "update-image-pipeline-lifecycle",
                    "disable",
                    "--family-id",
                    "example-detector",
                    "--bundle-spec-digest",
                    workspace.current["spec_digest"],
                    "--artifact-set-digest",
                    workspace.current["artifact_set_digest"],
                    "--expected-lifecycle-head-digest",
                    workspace.head,
                    "--expected-bundle-state-event-digest",
                    workspace.events[0]["event_digest"],
                    "--actor-role-id",
                    "repo_maintainer",
                    "--public-review-id",
                    "gh-cli-review",
                    "--review-status",
                    "approved",
                    "--reason",
                    "Review the exact dry-run lifecycle transition.",
                    "--workspace",
                    str(workspace.root),
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "dry_run_ready")
        self.assertEqual(workspace.lifecycle.read_bytes(), before)


def load_json(path: Path) -> dict:
    return json.loads(path.read_bytes())
