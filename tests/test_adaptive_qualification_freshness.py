from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main, mock

from tests.test_adaptive_evidence_activation import _code_owned_report, _managed_registry
from tests.test_adaptive_evidence_contracts import _redigest_report
from yolozu.adaptive.canonical import canonical_json_v1, canonical_sha256_v1
from yolozu.adaptive.evidence import compute_evidence_selection_key
from yolozu.adaptive.freshness import (
    QualificationFreshnessError,
    check_qualification_freshness,
    render_qualification_freshness_issue_body,
    write_qualification_freshness_report,
)


AS_OF = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _selection_key(report: dict) -> str:
    return compute_evidence_selection_key(
        bundle_spec_digest=report["bundle_spec_digest"],
        artifact_set_digest=report["artifact_set_digest"],
        environment_fingerprint=report["environment_fingerprint"],
        qualification_workload_fingerprint=report[
            "qualification_workload_fingerprint"
        ],
        protocol_fingerprint=report["protocol_fingerprint"],
    )


def _activation(
    report: dict,
    *,
    sequence: int = 1,
    previous: str = "0" * 64,
    event_id: str = "activation-1",
    activated_at: str = "2026-08-25T00:00:00Z",
) -> dict:
    key = _selection_key(report)
    event = {
        "schema_version": 1,
        "stream_id": key,
        "selection_key": key,
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_id": event_id,
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "state": "active",
        "replacement_report_id": None,
        "replacement_report_digest": None,
        "activated_at": activated_at,
        "valid_until": report["valid_until"],
        "reviewer_role_id": "site_operator",
        "review_reference": {"kind": "site_local_status", "status": "present"},
        "issuer_claim": "site_source",
        "trust_domain": "site_managed",
        "reason": "Reviewed site qualification evidence.",
        "event_digest": "0" * 64,
    }
    event["event_digest"] = canonical_sha256_v1(
        event, own_digest_field="event_digest"
    )
    return event


def _write_evidence(root: Path, reports: list[dict], events: list[dict]) -> Path:
    evidence = root / "evidence"
    report_root = evidence / "qualification_reports"
    report_root.mkdir(parents=True)
    evidence.joinpath("evidence_activation.jsonl").write_bytes(
        b"".join(canonical_json_v1(event) + b"\n" for event in events)
    )
    for report in reports:
        directory = report_root / report["report_id"]
        directory.mkdir()
        report_bytes = canonical_json_v1(report)
        directory.joinpath("qualification_report.json").write_bytes(report_bytes)
        entry = {
            "path": "qualification_report.json",
            "size_bytes": len(report_bytes),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        }
        directory.joinpath("checksums.json").write_bytes(
            canonical_json_v1(
                {
                    "schema_version": 1,
                    "files": [entry],
                    "expected_paths": ["qualification_report.json"],
                    "file_count": 1,
                    "total_bytes": len(report_bytes),
                }
            )
        )
    return evidence


def _report_with_validity(valid_until: datetime, *, report_id: str = "report-1") -> dict:
    report = _code_owned_report(report_id=report_id)
    report["valid_until"] = valid_until.strftime("%Y-%m-%dT%H:%M:%SZ")
    _redigest_report(report)
    return report


class TestQualificationFreshness(TestCase):
    def _check(self, report: dict, *, registry=None, activated_at: str = "2026-08-25T00:00:00Z") -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            evidence = _write_evidence(
                workspace, [report], [_activation(report, activated_at=activated_at)]
            )
            return check_qualification_freshness(
                workspace_root=workspace,
                evidence_root=evidence,
                as_of=AS_OF,
                registry=_managed_registry() if registry is None else registry,
            )

    def test_frozen_expiry_boundaries(self) -> None:
        cases = (
            (31 * 86400, "ok", 31),
            (30 * 86400, "due_30", 30),
            (14 * 86400, "due_14", 14),
            (7 * 86400, "due_7", 7),
            (12 * 3600, "due_7", 0),
            (0, "expired", 0),
            (-1, "expired", 0),
        )
        for seconds, state, days in cases:
            with self.subTest(seconds=seconds):
                report = _report_with_validity(AS_OF + timedelta(seconds=seconds))
                result = self._check(report)
                self.assertEqual(result["rows"][0]["state"], state)
                self.assertEqual(result["rows"][0]["days_remaining"], days)

    def test_runtime_and_lifecycle_drift_are_separate(self) -> None:
        runtime = _report_with_validity(AS_OF + timedelta(days=31))
        runtime["source_runtime_provenance"]["runtime_version"] = "changed"
        _redigest_report(runtime)
        runtime_result = self._check(runtime)
        self.assertEqual(runtime_result["rows"][0]["state"], "runtime_drift")
        self.assertIn(
            "runtime_version_changed", runtime_result["rows"][0]["reason_codes"]
        )

        lifecycle = _report_with_validity(AS_OF + timedelta(days=31))
        lifecycle_result = self._check(
            lifecycle, registry=_managed_registry(state="disabled")
        )
        self.assertEqual(
            lifecycle_result["rows"][0]["state"], "artifact_or_bundle_drift"
        )
        self.assertIn(
            "bundle_disabled_or_revoked",
            lifecycle_result["rows"][0]["reason_codes"],
        )

    def test_future_clock_and_conflicting_active_events_fail_closed(self) -> None:
        report = _report_with_validity(AS_OF + timedelta(days=31))
        future = self._check(report, activated_at="2026-08-30T00:00:00Z")
        self.assertEqual(future["rows"][0]["state"], "unknown")
        self.assertEqual(future["rows"][0]["reason_codes"], ["clock_invalid"])

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            second = copy.deepcopy(report)
            second["report_id"] = "report-2"
            _redigest_report(second)
            first_event = _activation(report)
            second_event = _activation(
                second,
                sequence=2,
                previous=first_event["event_digest"],
                event_id="activation-2",
            )
            evidence = _write_evidence(
                workspace, [report, second], [first_event, second_event]
            )
            result = check_qualification_freshness(
                workspace_root=workspace,
                evidence_root=evidence,
                as_of=AS_OF,
                registry=_managed_registry(),
            )
            self.assertEqual(result["rows"][0]["state"], "conflict")

    def test_site_mode_is_network_free_local_only_and_managed_output(self) -> None:
        report = _report_with_validity(AS_OF + timedelta(days=30))
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            evidence = _write_evidence(workspace, [report], [_activation(report)])
            with mock.patch("socket.create_connection", side_effect=AssertionError):
                result = check_qualification_freshness(
                    workspace_root=workspace,
                    evidence_root=evidence,
                    as_of=AS_OF,
                    missed_run_dates=("2026-08-24",),
                    registry=_managed_registry(),
                )
            self.assertFalse(result["upload_eligible"])
            self.assertEqual(result["missed_expected_run_dates"], ["2026-08-24"])
            path = write_qualification_freshness_report(
                result,
                workspace_root=workspace,
                output="reports/freshness",
            )
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_name("checksums.json").is_file())
            body = render_qualification_freshness_issue_body(result)
            self.assertIn("state=due_30", body)
            self.assertNotIn(str(workspace), body)

    def test_invalid_as_of_and_missed_date_are_rejected(self) -> None:
        with self.assertRaisesRegex(QualificationFreshnessError, "clock_invalid"):
            check_qualification_freshness(as_of="2026-02-30T00:00:00Z")
        with self.assertRaisesRegex(QualificationFreshnessError, "must precede"):
            check_qualification_freshness(
                as_of=AS_OF, missed_run_dates=("2026-08-29",)
            )

    def test_canonical_and_packaged_schemas_match(self) -> None:
        canonical = Path("docs/schemas/qualification_freshness_report.schema.json")
        packaged = Path("yolozu/data/schemas/qualification_freshness_report.schema.json")
        self.assertEqual(canonical.read_bytes(), packaged.read_bytes())
        schema = json.loads(canonical.read_text(encoding="utf-8"))
        self.assertIn("rows", schema["required"])
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    main()
