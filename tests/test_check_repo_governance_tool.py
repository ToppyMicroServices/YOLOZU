import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCheckRepoGovernanceTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_repo_governance.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"check_repo_governance --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--repo-json", proc.stdout)
        self.assertIn("--branch-protection-json", proc.stdout)
        self.assertIn("--require-reviews", proc.stdout)
        self.assertIn("--allow-missing-evidence", proc.stdout)

    def test_snapshot_audit_passes_for_expected_policy(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_repo_governance.py"

        repo_payload = {
            "default_branch": "main",
            "security_and_analysis": {
                "dependabot_security_updates": {"status": "enabled"},
                "secret_scanning": {"status": "enabled"},
                "secret_scanning_push_protection": {"status": "enabled"},
            },
        }
        branch_payload = {
            "required_pull_request_reviews": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews": True,
                "require_last_push_approval": True,
            },
            "required_conversation_resolution": {"enabled": True},
            "required_linear_history": {"enabled": True},
            "enforce_admins": {"enabled": True},
            "allow_force_pushes": {"enabled": False},
            "allow_deletions": {"enabled": False},
        }

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            temp_dir = Path(td)
            repo_json = temp_dir / "repo.json"
            branch_json = temp_dir / "branch_protection.json"
            out = temp_dir / "governance_report.json"
            repo_json.write_text(json.dumps(repo_payload), encoding="utf-8")
            branch_json.write_text(json.dumps(branch_payload), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(repo_root),
                    "--repo-json",
                    str(repo_json),
                    "--branch-protection-json",
                    str(branch_json),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"check_repo_governance failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertEqual(payload.get("missing_evidence"), [])
            self.assertEqual(payload.get("failed_required_checks"), [])
            self.assertIn("CodeReviewID", {item["id"] for item in payload.get("manual_followups", [])})

    def test_local_only_mode_can_pass_when_missing_evidence_is_allowed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_repo_governance.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "governance_report.local.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(repo_root),
                    "--allow-missing-evidence",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"check_repo_governance local-only mode failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertEqual(set(payload.get("missing_evidence", [])), {"repo_json", "branch_protection_json"})


if __name__ == "__main__":
    unittest.main()
