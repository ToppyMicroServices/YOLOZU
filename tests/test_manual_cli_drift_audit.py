import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestManualCliDriftAudit(unittest.TestCase):
    def test_manual_cli_drift_audit_passes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_manual_cli_drift.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manual",
                "manual/chapters/04_cli_reference.tex",
                "--allowlist",
                "docs/manual_cli_drift_allowlist.json",
                "--json",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"audit_manual_cli_drift.py failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload.get("ok"))
        self.assertIn("benchmark", payload.get("documented_commands") or [])
        self.assertIn("train", payload.get("documented_commands") or [])
        self.assertIn("benchmark", payload.get("canonical_commands") or [])


if __name__ == "__main__":
    unittest.main()
