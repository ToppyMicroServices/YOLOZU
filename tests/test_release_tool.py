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
    def test_shell_wrapper_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        wrapper = repo_root / "release.sh"
        self.assertTrue(wrapper.is_file(), "missing release.sh")

        proc = subprocess.run(
            ["bash", str(wrapper), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"release.sh --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("Usage:", proc.stdout)
        self.assertIn("Delegated tool help:", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)

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
        self.assertIn("--versioning", proc.stdout)
        self.assertIn("--allow-major", proc.stdout)

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
            self.assertEqual(str(payload.get("versioning_scheme")), "semver")
            self.assertRegex(str(payload.get("next_version")), r"^\d+\.\d+\.\d+$")
            self.assertIn(str(payload.get("bump_scale")), {"small", "medium", "large"})
            release_actions = payload.get("release_actions") or {}
            self.assertFalse(bool(release_actions.get("github_release_publish")))
            self.assertFalse(bool(release_actions.get("pypi_update_via_publish_workflow")))
            self.assertFalse(bool(release_actions.get("zenodo_manual_doi_dispatch")))

    def test_shell_wrapper_dry_run_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        wrapper = repo_root / "release.sh"
        version = _package_version(repo_root)

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "release_report.from_shell.json"
            proc = subprocess.run(
                [
                    "bash",
                    str(wrapper),
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
                self.fail(f"release.sh dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(str(payload.get("current_version")), version)
            self.assertEqual(str(payload.get("versioning_scheme")), "semver")

    def test_calver_helpers(self) -> None:
        from tools import release as release_tool

        self.assertEqual(str(release_tool._detect_versioning_scheme("2026.03.20.0")), "calver")
        self.assertEqual(str(release_tool._bump_calver("2026.03.20.0", today=(2026, 3, 20))), "2026.03.20.1")
        self.assertEqual(str(release_tool._bump_calver("2026.03.19.4", today=(2026, 3, 20))), "2026.03.20.0")

    def test_semver_bump_prefers_minor_for_large_non_breaking_change(self) -> None:
        from tools import release as release_tool

        self.assertEqual(str(release_tool._recommended_semver_bump("small", breaking=False)), "patch")
        self.assertEqual(str(release_tool._recommended_semver_bump("medium", breaking=False)), "minor")
        self.assertEqual(str(release_tool._recommended_semver_bump("large", breaking=False)), "minor")
        self.assertEqual(str(release_tool._bump_semver("1.2.3", "minor")), "1.3.0")

    def test_semver_bump_uses_major_only_for_breaking_signal(self) -> None:
        from tools import release as release_tool

        self.assertEqual(str(release_tool._recommended_semver_bump("small", breaking=True)), "major")
        self.assertEqual(str(release_tool._recommended_semver_bump("large", breaking=True)), "major")
        self.assertEqual(str(release_tool._bump_semver("1.2.3", "major")), "2.0.0")

    def test_major_bump_requires_explicit_allow_major(self) -> None:
        from tools import release as release_tool

        self.assertTrue(
            bool(
                release_tool._major_bump_requires_confirmation(
                    versioning="semver",
                    semver_bump="major",
                    allow_major=False,
                )
            )
        )
        self.assertFalse(
            bool(
                release_tool._major_bump_requires_confirmation(
                    versioning="semver",
                    semver_bump="major",
                    allow_major=True,
                )
            )
        )
        self.assertFalse(
            bool(
                release_tool._major_bump_requires_confirmation(
                    versioning="semver",
                    semver_bump="minor",
                    allow_major=False,
                )
            )
        )

    def test_breaking_change_signal_helper(self) -> None:
        from tools import release as release_tool

        self.assertTrue(
            bool(
                release_tool._contains_breaking_change_signal(
                    "feat: add API\n\nBREAKING CHANGE: drops old behavior",
                    "feat: add API",
                )
            )
        )
        self.assertTrue(
            bool(
                release_tool._contains_breaking_change_signal(
                    "chore: cleanup\n\nBREAKING-CHANGE: incompatible config",
                    "chore: cleanup",
                )
            )
        )
        self.assertTrue(bool(release_tool._contains_breaking_change_signal("details", "feat!: overhaul public API")))
        self.assertFalse(
            bool(
                release_tool._contains_breaking_change_signal(
                    "feat: add option\n\nWarning!: documentation note only",
                    "feat: add option",
                )
            )
        )

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
