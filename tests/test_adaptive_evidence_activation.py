from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from tests.test_adaptive_bundle_contracts import (
    _bundle_payload,
    _lifecycle_event,
    _registry_payload,
)
from tests.test_adaptive_evidence_contracts import _redigest_report, _report_payload
from yolozu.adaptive.activation import activate_qualification_evidence
from yolozu.adaptive.bundle_registry import LoadedAlgorithmBundleRegistry
from yolozu.adaptive.bundles import (
    project_bundle_lifecycle,
    validate_algorithm_bundle_registry,
)
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.evidence import compute_evidence_selection_key
from yolozu.cli_entry import main as cli_main


AS_OF = "2026-08-25T00:00:00Z"


def _code_owned_report(*, report_id: str = "report-1") -> dict[str, Any]:
    bundle = _bundle_payload()
    report = _report_payload(report_id=report_id)
    report["collector"] = {
        "id": "yolozu_qualifier",
        "version": "1",
        "source_digest": canonical_sha256_v1(
            {"module": "yolozu.adaptive.qualification", "interface_version": 1}
        ),
    }
    report["issuer"] = {
        "id": "yolozu_qualification_workflow",
        "version": "1",
        "source_digest": canonical_sha256_v1(
            {"workflow": "local_unactivated_qualification", "interface_version": 1}
        ),
    }
    report["bundle_spec_digest"] = bundle["spec_digest"]
    report["artifact_set_digest"] = bundle["artifact_set_digest"]
    _redigest_report(report)
    return report


def _managed_registry(*, state: str = "enabled") -> LoadedAlgorithmBundleRegistry:
    bundle_payload = _bundle_payload()
    registry = validate_algorithm_bundle_registry(_registry_payload(bundle_payload))
    reviews = [{"artifact_id": "model", "review_state": "approved"}]
    registered = _lifecycle_event(
        sequence=1,
        previous="0" * 64,
        scope="bundle_global",
        event_type="register_global",
        variant={
            "family_id": "example-detector",
            "bundle_spec_digest": bundle_payload["spec_digest"],
            "artifact_set_digest": bundle_payload["artifact_set_digest"],
            "bundle_state": state,
            "artifact_license_reviews": reviews,
        },
    )
    candidate = _lifecycle_event(
        sequence=2,
        previous=registered["event_digest"],
        scope="channel_assignment",
        event_type="candidate_registration",
        variant={
            "family_id": "example-detector",
            "channel": "Candidate",
            "target_bundle_spec_digest": bundle_payload["spec_digest"],
            "target_artifact_set_digest": bundle_payload["artifact_set_digest"],
            "target_artifact_license_reviews": reviews,
            "support_profile_index_head": "0" * 64,
            "profile_set_record_id": None,
            "profile_set_record_digest": None,
            "profile_set_digest": canonical_sha256_v1([]),
            "profiles": [],
            "evidence_bindings": [],
        },
    )
    lifecycle = project_bundle_lifecycle(
        registry,
        [registered, candidate],
        source_trust_domain="yolozu_managed",
    )
    return LoadedAlgorithmBundleRegistry(
        registry=registry,
        bundles=registry.bundles,
        lifecycle=lifecycle,
        registry_trust_domain="yolozu_managed",
        lifecycle_trust_domain="yolozu_managed",
        source_kind="packaged_ssot",
    )


def _write_site_report(root: Path, report: dict[str, Any], name: str) -> Path:
    directory = root / name
    directory.mkdir()
    report_bytes = canonical_json_v1(report)
    report_path = directory / "qualification_report.json"
    report_path.write_bytes(report_bytes)
    entry = {
        "path": "qualification_report.json",
        "size_bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    manifest = {
        "schema_version": 1,
        "files": [entry],
        "expected_paths": ["qualification_report.json"],
        "file_count": 1,
        "total_bytes": len(report_bytes),
    }
    directory.joinpath("checksums.json").write_bytes(canonical_json_v1(manifest))
    return report_path


def _key(report: dict[str, Any]) -> str:
    return compute_evidence_selection_key(
        bundle_spec_digest=report["bundle_spec_digest"],
        artifact_set_digest=report["artifact_set_digest"],
        environment_fingerprint=report["environment_fingerprint"],
        qualification_workload_fingerprint=report[
            "qualification_workload_fingerprint"
        ],
        protocol_fingerprint=report["protocol_fingerprint"],
    )


class TestAdaptiveEvidenceActivation(unittest.TestCase):
    def _call(
        self,
        *,
        workspace: Path,
        report_path: Path,
        report: dict[str, Any],
        stream: Path,
        operation: str = "activate",
        expected_head: str = "0" * 64,
        expected_current: str = "none",
        approve: bool = False,
        prior: tuple[Path, ...] = (),
        supersede: str | None = None,
        revoke: str | None = None,
        registry: LoadedAlgorithmBundleRegistry | None = None,
        fault_hook: Any = None,
    ) -> Any:
        return activate_qualification_evidence(
            operation=operation,
            report_path=report_path,
            report_id=report["report_id"],
            report_digest=report["report_digest"],
            selection_key=_key(report),
            stream_path=stream,
            workspace_root=workspace,
            expected_head_digest=expected_head,
            expected_current_activation_id=expected_current,
            reviewer_role_id="site_operator",
            reason="Reviewed exact local qualification evidence.",
            approve=approve,
            site_local_review_present=True,
            supersede_activation_id=supersede,
            revoke_activation_id=revoke,
            prior_report_paths=prior,
            as_of=AS_OF,
            registry=_managed_registry() if registry is None else registry,
            fault_hook=fault_hook,
        )

    def test_default_dry_run_then_activate_candidate_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            stream = workspace / "site" / "evidence.jsonl"
            stream.parent.mkdir()
            report = _code_owned_report()
            report_path = _write_site_report(workspace, report, "report-1")

            dry_run = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
            )
            self.assertEqual(dry_run.status, "dry_run_ready")
            self.assertFalse(stream.exists())
            self.assertEqual(dry_run.source_trust_domain, "site_managed")
            self.assertEqual(dry_run.support_scope, "site_qualified")
            self.assertEqual(len(dry_run.planned_records), 1)

            applied = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                approve=True,
            )
            self.assertEqual(applied.status, "applied")
            self.assertTrue(stream.read_bytes().endswith(b"\n"))

    def test_supersede_then_terminal_revoke_and_illegal_reactivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            stream = workspace / "site" / "evidence.jsonl"
            stream.parent.mkdir()
            first = _code_owned_report(report_id="report-1")
            second = _code_owned_report(report_id="report-2")
            third = _code_owned_report(report_id="report-3")
            first_path = _write_site_report(workspace, first, "report-1")
            second_path = _write_site_report(workspace, second, "report-2")
            third_path = _write_site_report(workspace, third, "report-3")
            active = self._call(
                workspace=workspace,
                report_path=first_path,
                report=first,
                stream=stream,
                approve=True,
            )
            active_id = active.observed_current_activation_id
            self.assertIsNotNone(active_id)
            superseded = self._call(
                workspace=workspace,
                report_path=second_path,
                report=second,
                stream=stream,
                operation="supersede",
                expected_head=active.observed_head_digest,
                expected_current=str(active_id),
                approve=True,
                prior=(first_path,),
                supersede=str(active_id),
            )
            self.assertEqual(superseded.status, "applied")
            self.assertEqual(len(superseded.applied_record_digests), 2)
            replacement_id = superseded.observed_current_activation_id
            revoked = self._call(
                workspace=workspace,
                report_path=second_path,
                report=second,
                stream=stream,
                operation="revoke",
                expected_head=superseded.observed_head_digest,
                expected_current=str(replacement_id),
                approve=True,
                prior=(first_path,),
                revoke=str(replacement_id),
            )
            self.assertEqual(revoked.status, "applied")
            self.assertIsNone(revoked.observed_current_activation_id)

            illegal = self._call(
                workspace=workspace,
                report_path=third_path,
                report=third,
                stream=stream,
                expected_head=revoked.observed_head_digest,
                expected_current="none",
                prior=(first_path, second_path),
            )
            self.assertEqual(illegal.status, "dry_run_blocked")
            self.assertIn(
                "planned_transition_invalid",
                {gate.code for gate in illegal.gates},
            )

    def test_stale_head_untrusted_input_and_blocked_lifecycle_write_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            stream = workspace / "site" / "evidence.jsonl"
            stream.parent.mkdir()
            report = _code_owned_report()
            report_path = _write_site_report(workspace, report, "report-1")
            stale = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                expected_head="f" * 64,
                approve=True,
            )
            self.assertEqual(stale.status, "apply_failed")
            self.assertIn("stale_head", {gate.code for gate in stale.gates})
            self.assertFalse(stream.exists())

            report_path.parent.joinpath("checksums.json").unlink()
            untrusted = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                approve=True,
            )
            self.assertIn("source_not_managed", {gate.code for gate in untrusted.gates})
            self.assertFalse(stream.exists())

            report_path = _write_site_report(workspace, report, "report-2")
            blocked = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                approve=True,
                registry=_managed_registry(state="disabled"),
            )
            self.assertIn("bundle_blocked", {gate.code for gate in blocked.gates})
            self.assertFalse(stream.exists())

    def test_interrupted_atomic_write_preserves_old_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            stream = workspace / "site" / "evidence.jsonl"
            stream.parent.mkdir()
            report = _code_owned_report()
            report_path = _write_site_report(workspace, report, "report-1")

            def interrupt(step: str) -> None:
                if step == "before_replace":
                    raise ValueError("fixture interruption")

            outcome = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                approve=True,
                fault_hook=interrupt,
            )
            self.assertEqual(outcome.status, "apply_failed")
            self.assertFalse(stream.exists())
            self.assertEqual(list(stream.parent.glob(".*.stage.*")), [])

            active = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                approve=True,
            )
            before_revoke = stream.read_bytes()
            active_id = str(active.observed_current_activation_id)
            interrupted_revoke = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
                operation="revoke",
                expected_head=active.observed_head_digest,
                expected_current=active_id,
                approve=True,
                revoke=active_id,
                fault_hook=interrupt,
            )
            self.assertEqual(interrupted_revoke.status, "apply_failed")
            self.assertEqual(stream.read_bytes(), before_revoke)
            self.assertEqual(list(stream.parent.glob(".*.stage.*")), [])

    def test_unknown_issuer_and_personal_actor_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            stream = workspace / "site" / "evidence.jsonl"
            stream.parent.mkdir()
            report = _code_owned_report()
            report["issuer"] = copy.deepcopy(report["issuer"])
            report["issuer"]["id"] = "unknown_workflow"
            _redigest_report(report)
            report_path = _write_site_report(workspace, report, "report-1")
            unknown = self._call(
                workspace=workspace,
                report_path=report_path,
                report=report,
                stream=stream,
            )
            self.assertIn("issuer_unknown", {gate.code for gate in unknown.gates})

            valid = _code_owned_report()
            valid_path = _write_site_report(workspace, valid, "report-2")
            personal = activate_qualification_evidence(
                operation="activate",
                report_path=valid_path,
                report_id=valid["report_id"],
                report_digest=valid["report_digest"],
                selection_key=_key(valid),
                stream_path=stream,
                workspace_root=workspace,
                expected_head_digest="0" * 64,
                expected_current_activation_id="none",
                reviewer_role_id="alice",
                reason="Reviewed.",
                site_local_review_present=True,
                as_of=AS_OF,
                registry=_managed_registry(),
            )
            self.assertIn(
                "planned_transition_invalid", {gate.code for gate in personal.gates}
            )

    def test_cli_default_is_no_write_and_reports_all_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            site = workspace / "site"
            site.mkdir()
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli_main(
                    [
                        "activate-qualification-evidence",
                        "--workspace",
                        str(workspace),
                        "--activation-stream",
                        str(site / "evidence.jsonl"),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(payload["status"], "dry_run_blocked")
            self.assertEqual(
                {
                    "report_path_missing",
                    "report_id_missing",
                    "report_digest_missing",
                    "selection_key_missing",
                    "expected_head_missing",
                    "expected_current_missing",
                    "reviewer_role_missing",
                    "reason_missing",
                },
                {gate["code"] for gate in payload["gates"]},
            )
            self.assertFalse((site / "evidence.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
