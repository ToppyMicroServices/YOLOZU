import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _package_version(repo_root: Path) -> str:
    text = (repo_root / "yolozu" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise RuntimeError("could not parse version")
    return str(m.group(1))


class TestReleaseTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "release.py"
        self.assertTrue(script.is_file(), "missing tools/release.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"release --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--skip-gh", proc.stdout)

    def test_dry_run_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "release.py"
        version = _package_version(repo_root)

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "release_report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dry-run",
                    "--allow-dirty",
                    "--allow-non-main",
                    "--skip-checks",
                    "--skip-gh",
                    "--skip-zenodo",
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
                self.fail(f"release dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(str(payload.get("current_version")), version)
            self.assertRegex(str(payload.get("next_version")), r"^\d+\.\d+\.\d+$")
            self.assertIn(str(payload.get("bump_scale")), {"small", "medium", "large"})
            release_actions = payload.get("release_actions") or {}
            self.assertFalse(bool(release_actions.get("github_release_publish")))
            self.assertFalse(bool(release_actions.get("pypi_update_via_publish_workflow")))
            self.assertFalse(bool(release_actions.get("zenodo_manual_doi_dispatch")))

    def test_tools_wrapper_release_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        wrapper = repo_root / "tools" / "yolozu.py"
        proc = subprocess.run(
            [sys.executable, str(wrapper), "release", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"tools/yolozu.py release --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("usage: yolozu release", proc.stdout)


if __name__ == "__main__":
    unittest.main()
