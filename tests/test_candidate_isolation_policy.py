import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from yolozu.adaptive.canonical import canonical_sha256_v1
from yolozu.adaptive.isolation_policy import (
    CANDIDATE_ISOLATION_DECISION,
    candidate_isolation_policy,
    probe_candidate_isolation,
)


class TestCandidateIsolationPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_policy_is_canonical_none_supported_and_deny_by_default(self) -> None:
        policy = candidate_isolation_policy()
        self.assertEqual(CANDIDATE_ISOLATION_DECISION, "none_supported")
        self.assertEqual(policy["decision"], "none_supported")
        self.assertEqual(policy["execution_fallback"], "forbidden")
        self.assertEqual(policy["acquisition"]["approved_https_artifacts"], [])
        self.assertEqual(policy["acquisition"]["allowed_archive_media_types"], [])
        self.assertEqual(policy["dependency_lock"]["status"], "unselected")
        self.assertEqual(policy["base_image"]["status"], "unselected")
        self.assertEqual(
            policy["policy_digest"],
            canonical_sha256_v1(policy, own_digest_field="policy_digest"),
        )
        self.assertTrue(all(row["matrix_status"] != "supported" for row in policy["backend_rows"]))

    def test_present_backend_executable_cannot_become_supported(self) -> None:
        observed = probe_candidate_isolation(
            platform_values={
                "os": "Darwin",
                "release": "26.6.1",
                "architecture": "arm64",
            },
            executable_probe=lambda executable: executable in {"podman", "/usr/bin/sandbox-exec"},
            collected_at=datetime(2026, 8, 28, 13, 8, 12, tzinfo=timezone.utc),
        )
        self.assertEqual(observed["decision"], "none_supported")
        self.assertEqual(observed["capability_status"], "unsupported")
        self.assertEqual(observed["enforcement_boundary"], "none")
        self.assertEqual(observed["reason_codes"], ["isolation_no_supported_backend"])
        self.assertFalse(observed["candidate_execution_attempted"])
        self.assertEqual(observed["collected_at"], "2026-08-28T13:08:12Z")

        by_backend = {
            item["backend_id"]: item for item in observed["backend_observations"]
        }
        self.assertEqual(
            by_backend["macos_podman_machine"]["executable_status"], "present"
        )
        self.assertEqual(
            by_backend["macos_podman_machine"]["matrix_status"], "not_supported"
        )
        self.assertEqual(
            by_backend["macos_sandbox_exec"]["matrix_status"], "rejected"
        )
        self.assertTrue(
            all(item["matrix_status"] != "supported" for item in by_backend.values())
        )

    def test_every_required_control_is_unique_and_fails_without_execution(self) -> None:
        policy = candidate_isolation_policy()
        observed = probe_candidate_isolation(
            platform_values={
                "os": "Linux",
                "release": "example",
                "architecture": "x86_64",
            },
            executable_probe=lambda _executable: False,
            collected_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )
        controls = policy["mandatory_controls"]
        results = observed["control_results"]
        self.assertEqual(len(controls), 28)
        self.assertEqual(len(results), 28)
        self.assertEqual(
            len({item["control_id"] for item in controls}),
            len(controls),
        )
        self.assertEqual(
            [(item["phase"], item["control_id"], item["probe_id"], item["failure_code"]) for item in controls],
            [(item["phase"], item["control_id"], item["probe_id"], item["reason_code"]) for item in results],
        )
        self.assertTrue(all(item["status"] == "not_run" for item in results))

    def test_probe_output_matches_packaged_schema_surface(self) -> None:
        docs_schema = self.repo_root / "docs/schemas/candidate_isolation_probe.schema.json"
        packaged_schema = self.repo_root / "yolozu/data/schemas/candidate_isolation_probe.schema.json"
        self.assertEqual(docs_schema.read_bytes(), packaged_schema.read_bytes())
        schema = json.loads(docs_schema.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["decision"]["const"], "none_supported")
        self.assertEqual(schema["properties"]["capability_status"]["const"], "unsupported")
        self.assertEqual(schema["properties"]["candidate_execution_attempted"]["const"], False)
        self.assertEqual(schema["properties"]["control_results"]["minItems"], 28)
        self.assertEqual(schema["properties"]["control_results"]["maxItems"], 28)

        result = subprocess.run(
            [sys.executable, "tools/probe_candidate_isolation.py"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(set(payload), set(schema["required"]))
        self.assertEqual(payload["decision"], "none_supported")
        self.assertEqual(payload["capability_status"], "unsupported")
        self.assertFalse(payload["candidate_execution_attempted"])

    def test_help_and_decision_document_cover_code_owned_matrix(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/probe_candidate_isolation.py", "--help"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("candidate-isolation", result.stdout)
        document = (
            self.repo_root / "docs/candidate_isolation_threat_model.md"
        ).read_text(encoding="utf-8")
        policy = candidate_isolation_policy()
        for row in policy["backend_rows"]:
            self.assertIn(row["backend_id"], document)
        for control in policy["mandatory_controls"]:
            self.assertIn(control["probe_id"], document)
            self.assertIn(control["failure_code"], document)


if __name__ == "__main__":
    unittest.main()
