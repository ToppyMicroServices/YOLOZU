from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_HELPER = ROOT / "refresh_beads_sync.sh"
RUNPOD_HELPER = ROOT / "deploy" / "runpod" / "refresh_beads_sync.sh"


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


class BeadsHelperContractTests(unittest.TestCase):
    def test_shell_helpers_support_help(self) -> None:
        for helper in (ROOT_HELPER, RUNPOD_HELPER):
            with self.subTest(helper=helper):
                proc = run(["bash", str(helper), "--help"], cwd=ROOT)
                self.assertIn("Usage:", proc.stdout)
                self.assertIn("bd", proc.stdout)

    def test_active_workflow_uses_current_bd_commands(self) -> None:
        paths = (
            ROOT / "AGENTS.md",
            ROOT / ".beads" / "README.md",
            ROOT / ".beads" / "config.yaml",
            ROOT / "docs" / "beads_github_workflow.md",
            ROOT / "docs" / "roadmap.md",
            ROOT_HELPER,
            RUNPOD_HELPER,
            ROOT / "deploy" / "runpod" / "README.md",
            ROOT / "tools" / "link_beads_to_github.py",
        )
        unsupported = ("bd " + "sync", "bd " + "resolve-conflicts")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for command in unsupported:
                    self.assertNotIn(command, text)

    def test_runpod_refresh_imports_remote_snapshot_with_stub_bd(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            remote = base / "remote.git"
            publisher = base / "publisher"
            client = base / "client"

            run(["git", "init", "--bare", str(remote)], cwd=base)
            run(["git", "init", "-b", "main", str(publisher)], cwd=base)
            self._configure_git(publisher)

            shutil.copy2(ROOT_HELPER, publisher / "refresh_beads_sync.sh")
            (publisher / "deploy" / "runpod").mkdir(parents=True)
            shutil.copy2(
                RUNPOD_HELPER,
                publisher / "deploy" / "runpod" / "refresh_beads_sync.sh",
            )
            (publisher / "README.md").write_text(
                "temporary workflow fixture\n",
                encoding="utf-8",
            )
            run(["git", "add", "."], cwd=publisher)
            run(["git", "commit", "-m", "initial fixture"], cwd=publisher)
            run(["git", "remote", "add", "origin", str(remote)], cwd=publisher)
            run(["git", "push", "-u", "origin", "main"], cwd=publisher)

            expected_snapshot = '{"id":"T37-stub","title":"stub issue"}\n'
            run(["git", "switch", "-c", "beads-sync"], cwd=publisher)
            (publisher / ".beads").mkdir()
            (publisher / ".beads" / "issues.jsonl").write_text(
                expected_snapshot,
                encoding="utf-8",
            )
            run(["git", "add", ".beads/issues.jsonl"], cwd=publisher)
            run(["git", "commit", "-m", "add snapshot fixture"], cwd=publisher)
            run(["git", "push", "-u", "origin", "beads-sync"], cwd=publisher)

            run(
                [
                    "git",
                    "clone",
                    "--single-branch",
                    "--branch",
                    "main",
                    str(remote),
                    str(client),
                ],
                cwd=base,
            )
            self._configure_git(client)

            capture = base / "imported.jsonl"
            stub_bd = base / "bd-stub"
            stub_bd.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  import)
    if [[ "$#" -ne 3 || "$3" != "--json" ]]; then
      echo "unexpected bd import arguments: $*" >&2
      exit 64
    fi
    cp "$2" "${BD_IMPORT_CAPTURE}"
    printf '{"created":1}\\n'
    ;;
  list)
    [[ "$#" -eq 1 ]] || exit 64
    printf 'T37-stub stub issue\\n'
    ;;
  *)
    echo "unexpected bd arguments: $*" >&2
    exit 64
    ;;
esac
""",
                encoding="utf-8",
            )
            stub_bd.chmod(0o755)
            env = dict(os.environ)
            env["BD_BIN"] = str(stub_bd)
            env["BD_IMPORT_CAPTURE"] = str(capture)

            proc = run(
                [
                    "bash",
                    str(client / "deploy" / "runpod" / "refresh_beads_sync.sh"),
                ],
                cwd=base,
                env=env,
            )

            self.assertEqual(capture.read_text(encoding="utf-8"), expected_snapshot)
            self.assertIn('{"created":1}', proc.stdout)
            self.assertIn("refreshed local bd database", proc.stdout)
            self.assertFalse((client / ".beads" / "issues.jsonl").exists())
            branch = run(
                ["git", "branch", "--show-current"],
                cwd=client,
            ).stdout.strip()
            self.assertEqual(branch, "main")
            run(
                [
                    "git",
                    "show-ref",
                    "--verify",
                    "refs/remotes/origin/beads-sync",
                ],
                cwd=client,
            )

    @staticmethod
    def _configure_git(repo: Path) -> None:
        run(["git", "config", "user.email", "test@example.com"], cwd=repo)
        run(["git", "config", "user.name", "Workflow Test"], cwd=repo)


@unittest.skipUnless(shutil.which("bd"), "bd is required for the integration test")
class BeadsExportImportIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.remote = self.base / "remote.git"
        self.publisher = self.base / "publisher"
        self.client = self.base / "client"
        self.exporter = self.base / "exporter"

        run(["git", "init", "--bare", str(self.remote)], cwd=self.base)
        run(["git", "init", "-b", "main", str(self.publisher)], cwd=self.base)
        self._configure_git(self.publisher)

        shutil.copy2(ROOT_HELPER, self.publisher / "refresh_beads_sync.sh")
        (self.publisher / "deploy" / "runpod").mkdir(parents=True)
        shutil.copy2(
            RUNPOD_HELPER,
            self.publisher / "deploy" / "runpod" / "refresh_beads_sync.sh",
        )
        (self.publisher / "README.md").write_text("temporary workflow fixture\n", encoding="utf-8")
        run(["git", "add", "."], cwd=self.publisher)
        run(["git", "commit", "-m", "initial fixture"], cwd=self.publisher)
        run(["git", "remote", "add", "origin", str(self.remote)], cwd=self.publisher)
        run(["git", "push", "-u", "origin", "main"], cwd=self.publisher)

        run(["git", "init", str(self.exporter)], cwd=self.base)
        self._configure_git(self.exporter)
        run(
            [
                "bd",
                "init",
                "--non-interactive",
                "--skip-hooks",
                "--skip-agents",
                "--prefix",
                "T37",
            ],
            cwd=self.exporter,
        )
        run(
            [
                "bd",
                "create",
                "remote snapshot issue",
                "--id",
                "T37-remote",
                "--silent",
            ],
            cwd=self.exporter,
        )
        snapshot = self.base / "issues.jsonl"
        run(["bd", "export", "-o", str(snapshot)], cwd=self.exporter)

        run(["git", "switch", "-c", "beads-sync"], cwd=self.publisher)
        (self.publisher / ".beads").mkdir()
        shutil.copy2(snapshot, self.publisher / ".beads" / "issues.jsonl")
        run(["git", "add", ".beads/issues.jsonl"], cwd=self.publisher)
        run(["git", "commit", "-m", "add exported issue snapshot"], cwd=self.publisher)
        run(["git", "push", "-u", "origin", "beads-sync"], cwd=self.publisher)

        run(["git", "switch", "main"], cwd=self.publisher)
        run(["git", "switch", "-c", "invalid-beads-snapshot"], cwd=self.publisher)
        (self.publisher / ".beads").mkdir(exist_ok=True)
        (self.publisher / ".beads" / "issues.jsonl").write_text(
            snapshot.read_text(encoding="utf-8") + "not valid JSONL\n",
            encoding="utf-8",
        )
        run(["git", "add", ".beads/issues.jsonl"], cwd=self.publisher)
        run(["git", "commit", "-m", "add invalid snapshot fixture"], cwd=self.publisher)
        run(
            ["git", "push", "-u", "origin", "invalid-beads-snapshot"],
            cwd=self.publisher,
        )

        run(
            [
                "git",
                "clone",
                "--single-branch",
                "--branch",
                "main",
                str(self.remote),
                str(self.client),
            ],
            cwd=self.base,
        )
        self._configure_git(self.client)
        run(
            [
                "bd",
                "init",
                "--non-interactive",
                "--skip-hooks",
                "--skip-agents",
                "--prefix",
                "CLIENT",
            ],
            cwd=self.client,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _configure_git(repo: Path) -> None:
        run(["git", "config", "user.email", "test@example.com"], cwd=repo)
        run(["git", "config", "user.name", "Workflow Test"], cwd=repo)

    def test_single_branch_clone_imports_exported_snapshot_without_worktree(self) -> None:
        self.assertFalse(
            (self.client / ".git" / "beads-worktrees" / "beads-sync").exists()
        )
        working_snapshot = self.client / ".beads" / "issues.jsonl"
        before = (
            working_snapshot.read_bytes() if working_snapshot.exists() else None
        )

        proc = run(
            [
                "bash",
                str(self.client / "deploy" / "runpod" / "refresh_beads_sync.sh"),
            ],
            cwd=self.base,
        )
        self.assertIn('"created": 1', proc.stdout)
        self.assertIn("refreshed local bd database", proc.stdout)
        self.assertIn("T37-remote", proc.stdout)
        run(
            ["git", "show-ref", "--verify", "refs/remotes/origin/beads-sync"],
            cwd=self.client,
        )
        after = working_snapshot.read_bytes() if working_snapshot.exists() else None
        self.assertEqual(after, before)

        issue = json.loads(
            run(["bd", "show", "T37-remote", "--json"], cwd=self.client).stdout
        )
        self.assertEqual(issue[0]["title"], "remote snapshot issue")

        roundtrip = self.base / "roundtrip.jsonl"
        run(["bd", "export", "-o", str(roundtrip)], cwd=self.client)
        records = [
            json.loads(line)
            for line in roundtrip.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertIn("T37-remote", {record["id"] for record in records})

    def test_missing_snapshot_branch_fails_without_importing(self) -> None:
        env = dict(os.environ)
        env["SYNC_BRANCH"] = "missing-snapshot"
        proc = run(
            ["bash", "refresh_beads_sync.sh"],
            cwd=self.client,
            env=env,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        missing = run(
            ["bd", "show", "T37-remote", "--json"],
            cwd=self.client,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)

        env["SYNC_BRANCH"] = "invalid-beads-snapshot"
        invalid = run(
            ["bash", "refresh_beads_sync.sh"],
            cwd=self.client,
            env=env,
            check=False,
        )
        self.assertNotEqual(invalid.returncode, 0)
        still_missing = run(
            ["bd", "show", "T37-remote", "--json"],
            cwd=self.client,
            check=False,
        )
        self.assertNotEqual(still_missing.returncode, 0)


if __name__ == "__main__":
    unittest.main()
