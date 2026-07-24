import json
import re
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


def _package_version(repo_root: Path) -> str:
    text = (repo_root / "yolozu" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise RuntimeError("could not parse version")
    return str(m.group(1))


class TestReleaseTagTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "release_tag.py"
        self.assertTrue(script.is_file(), "missing tools/release_tag.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"release_tag --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--release-state", proc.stdout)
        self.assertIn("--push-tag", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)

    def test_dry_run_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "release_tag.py"
        version = _package_version(repo_root)
        prefix = f"codex-test-{uuid.uuid4().hex[:8]}-v"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "release_tag_report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--version",
                    version,
                    "--tag-prefix",
                    prefix,
                    "--allow-dirty",
                    "--allow-non-main",
                    "--dry-run",
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
                self.fail(f"release_tag dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(str(payload.get("tag")), f"{prefix}{version}")
            self.assertEqual(str(payload.get("release_state")), "none")
            self.assertTrue((payload.get("metadata_validation") or {}).get("ok"))
            steps = payload.get("steps") or []
            self.assertGreaterEqual(len(steps), 1)
            self.assertEqual(str((steps[0] or {}).get("status")), "dry_run")


if __name__ == "__main__":
    unittest.main()
