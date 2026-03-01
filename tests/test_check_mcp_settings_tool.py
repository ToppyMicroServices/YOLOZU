import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCheckMCPSettingsTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_mcp_settings.py"
        self.assertTrue(script.is_file(), "missing tools/check_mcp_settings.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"check_mcp_settings --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--manifest", proc.stdout)
        self.assertIn("--json-ref", proc.stdout)
        self.assertIn("--md-ref", proc.stdout)
        self.assertIn("--strict", proc.stdout)

    def test_check_mcp_settings_default_paths(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_mcp_settings.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "mcp_settings_check.json"
            proc = subprocess.run(
                [sys.executable, str(script), "--output", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"check_mcp_settings failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertEqual(str(payload.get("task")), "mcp_settings_check")
            checks = payload.get("checks") or {}
            self.assertTrue(bool(checks.get("manifest_contains_supported_tools")))
            self.assertTrue(bool(checks.get("generated_json_uptodate")))
            self.assertTrue(bool(checks.get("generated_md_uptodate")))


if __name__ == "__main__":
    unittest.main()
