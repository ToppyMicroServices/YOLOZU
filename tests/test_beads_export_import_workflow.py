from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_HELPER = ROOT / "refresh_beads_sync.sh"
EXPORT_HELPER = ROOT / "export_beads_snapshot.sh"
RUNPOD_HELPER = ROOT / "deploy" / "runpod" / "refresh_beads_sync.sh"
COMPAT_TOOL = ROOT / "tools" / "beads_snapshot_compat.py"


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


def copy_workflow_files(destination: Path) -> None:
    shutil.copy2(ROOT_HELPER, destination / "refresh_beads_sync.sh")
    shutil.copy2(EXPORT_HELPER, destination / "export_beads_snapshot.sh")
    (destination / "deploy" / "runpod").mkdir(parents=True)
    shutil.copy2(
        RUNPOD_HELPER,
        destination / "deploy" / "runpod" / "refresh_beads_sync.sh",
    )
    (destination / "tools").mkdir(parents=True)
    shutil.copy2(COMPAT_TOOL, destination / "tools" / COMPAT_TOOL.name)


class BeadsHelperContractTests(unittest.TestCase):
    def test_shell_helpers_support_help(self) -> None:
        for helper in (ROOT_HELPER, EXPORT_HELPER, RUNPOD_HELPER):
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
            EXPORT_HELPER,
            RUNPOD_HELPER,
            ROOT / "deploy" / "runpod" / "README.md",
            COMPAT_TOOL,
            ROOT / "tools" / "link_beads_to_github.py",
        )
        unsupported = ("bd " + "sync", "bd " + "resolve-conflicts")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for command in unsupported:
                    self.assertNotIn(command, text)

        publication_paths = (
            ROOT / "AGENTS.md",
            ROOT / ".beads" / "README.md",
            ROOT / "docs" / "beads_github_workflow.md",
            ROOT / "deploy" / "runpod" / "README.md",
            ROOT / "tools" / "link_beads_to_github.py",
        )
        for path in publication_paths:
            with self.subTest(publication_path=path):
                self.assertNotIn("bd export -o", path.read_text(encoding="utf-8"))

    def test_runpod_refresh_imports_remote_snapshot_with_stub_bd(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            expected_snapshot = '{"id":"T37-stub","title":"stub issue"}\n'
            client = self._make_single_branch_fixture(base, expected_snapshot)
            capture = base / "imported.jsonl"
            backup_path_capture = base / "backup-path.txt"
            stub_bd = base / "bd-stub"
            self._write_stub_bd(stub_bd)
            env = dict(os.environ)
            env["BD_BIN"] = str(stub_bd)
            env["BD_IMPORT_CAPTURE"] = str(capture)
            env["BD_BACKUP_PATH_CAPTURE"] = str(backup_path_capture)

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

    def test_double_failure_preserves_recovery_backup_with_stub_bd(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            client = self._make_single_branch_fixture(
                base,
                '{"id":"T37-stub","title":"stub issue"}\n',
            )
            stub_bd = base / "bd-stub"
            self._write_stub_bd(stub_bd)
            capture = base / "imported.jsonl"
            backup_path_capture = base / "backup-path.txt"
            env = dict(os.environ)
            env.update(
                {
                    "BD_BIN": str(stub_bd),
                    "BD_IMPORT_CAPTURE": str(capture),
                    "BD_BACKUP_PATH_CAPTURE": str(backup_path_capture),
                    "BD_FAIL_IMPORT": "1",
                    "BD_FAIL_RESTORE": "1",
                }
            )

            proc = run(
                ["bash", str(client / "refresh_beads_sync.sh")],
                cwd=client,
                env=env,
                check=False,
            )

            self.assertEqual(proc.returncode, 70)
            self.assertIn(
                "bd import and backup restore both failed",
                proc.stderr,
            )
            backup_path = Path(backup_path_capture.read_text(encoding="utf-8").strip())
            self.assertIn(str(backup_path), proc.stderr)
            self.assertTrue((backup_path / "recovery-sentinel").is_file())

    def _make_single_branch_fixture(
        self,
        base: Path,
        expected_snapshot: str,
    ) -> Path:
        remote = base / "remote.git"
        publisher = base / "publisher"
        client = base / "client"

        run(["git", "init", "--bare", str(remote)], cwd=base)
        run(["git", "init", "-b", "main", str(publisher)], cwd=base)
        self._configure_git(publisher)
        copy_workflow_files(publisher)
        (publisher / "README.md").write_text(
            "temporary workflow fixture\n",
            encoding="utf-8",
        )
        run(["git", "add", "."], cwd=publisher)
        run(["git", "commit", "-m", "initial fixture"], cwd=publisher)
        run(["git", "remote", "add", "origin", str(remote)], cwd=publisher)
        run(["git", "push", "-u", "origin", "main"], cwd=publisher)

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
        return client

    @staticmethod
    def _write_stub_bd(path: Path) -> None:
        path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  backup)
    case "${2:-}" in
      status)
        [[ "$#" -eq 3 && "$3" == "--json" ]] || exit 64
        printf '{"dolt":{"configured":false}}\\n'
        ;;
      init)
        [[ "$#" -eq 3 ]] || exit 64
        mkdir -p "$3"
        : > "$3/recovery-sentinel"
        printf '%s\\n' "$3" > "${BD_BACKUP_PATH_CAPTURE}"
        ;;
      sync|remove)
        [[ "$#" -eq 2 ]] || exit 64
        ;;
      restore)
        [[ "$#" -eq 4 && "$4" == "--force" ]] || exit 64
        [[ "${BD_FAIL_RESTORE:-0}" != "1" ]] || exit 43
        ;;
      *)
        echo "unexpected bd backup arguments: $*" >&2
        exit 64
        ;;
    esac
    ;;
  import)
    if [[ "$#" -ne 3 || "$3" != "--json" ]]; then
      echo "unexpected bd import arguments: $*" >&2
      exit 64
    fi
    cp "$2" "${BD_IMPORT_CAPTURE}"
    [[ "${BD_FAIL_IMPORT:-0}" != "1" ]] || exit 42
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
        path.chmod(0o755)

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

        copy_workflow_files(self.publisher)
        (self.publisher / "README.md").write_text(
            "temporary workflow fixture\n", encoding="utf-8"
        )
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
        exported_records = [
            json.loads(line)
            for line in snapshot.read_text(encoding="utf-8").splitlines()
            if line
        ]
        exported_records.extend(
            [
                {
                    "id": "T37-legacy",
                    "title": "historical deleted parent",
                    "status": "tombstone",
                    "priority": 2,
                    "issue_type": "epic",
                    "created_at": "2026-01-01T00:00:00.123456Z",
                    "updated_at": "2026-01-02T00:00:00.654321Z",
                    "close_reason": "historical cleanup",
                    "deleted_at": "2026-01-02T00:00:00.654321Z",
                    "deleted_by": "fixture",
                    "delete_reason": "historical cleanup",
                    "original_type": "epic",
                },
                {
                    "id": "T37-legacy.1",
                    "title": "retained closed descendant",
                    "status": "closed",
                    "priority": 2,
                    "issue_type": "task",
                    "created_at": "2026-01-01T01:00:00Z",
                    "updated_at": "2026-01-03T00:00:00Z",
                    "closed_at": "2026-01-03T00:00:00Z",
                    "close_reason": "completed before cleanup",
                    "dependencies": [
                        {
                            "issue_id": "T37-legacy.1",
                            "depends_on_id": "T37-legacy",
                            "type": "parent-child",
                            "created_at": "2026-01-01T01:00:00Z",
                        },
                        {
                            "issue_id": "T37-legacy.1",
                            "depends_on_id": "T37-legacy",
                            "type": "blocks",
                            "created_at": "2026-01-01T01:00:01Z",
                        },
                    ],
                },
            ]
        )
        snapshot_lines = [
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            for record in exported_records
        ]
        snapshot.write_text(
            "".join(f"{line}\n" for line in snapshot_lines),
            encoding="utf-8",
        )
        self.snapshot_text = snapshot.read_text(encoding="utf-8")
        self.snapshot_records = exported_records
        self.snapshot_lines_by_id = {
            record["id"]: line
            for record, line in zip(exported_records, snapshot_lines, strict=True)
        }

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

    def test_single_branch_clone_imports_exported_snapshot_without_worktree(
        self,
    ) -> None:
        self.assertFalse(
            (self.client / ".git" / "beads-worktrees" / "beads-sync").exists()
        )
        working_snapshot = self.client / ".beads" / "issues.jsonl"
        before = working_snapshot.read_bytes() if working_snapshot.exists() else None

        proc = run(
            [
                "bash",
                str(self.client / "deploy" / "runpod" / "refresh_beads_sync.sh"),
            ],
            cwd=self.base,
        )
        self.assertIn('"created": 3', proc.stdout)
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

        parent = json.loads(
            run(["bd", "show", "T37-legacy", "--json"], cwd=self.client).stdout
        )[0]
        self.assertEqual(parent["status"], "closed")
        self.assertIn("beads-sync-legacy-tombstone", parent["labels"])
        marker = parent["metadata"]["beads_sync_legacy_tombstone"]
        self.assertEqual(
            marker["original_json"],
            self.snapshot_lines_by_id["T37-legacy"],
        )
        child = json.loads(
            run(["bd", "show", "T37-legacy.1", "--json"], cwd=self.client).stdout
        )[0]
        self.assertEqual(child["status"], "closed")

        second_refresh = run(
            ["bash", str(self.client / "refresh_beads_sync.sh")],
            cwd=self.client,
        )
        self.assertIn("refreshed local bd database", second_refresh.stdout)

        raw_local = self.base / "raw-local.jsonl"
        run(["bd", "export", "-o", str(raw_local)], cwd=self.client)
        raw_records = [
            json.loads(line)
            for line in raw_local.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_by_id = {record["id"]: record for record in raw_records}
        self.assertEqual(
            set(raw_by_id), {record["id"] for record in self.snapshot_records}
        )
        self.assertEqual(raw_by_id["T37-legacy"]["status"], "closed")

        publish_dir = self.base / "publish" / ".beads"
        publish_dir.mkdir(parents=True)
        destination = publish_dir / "issues.jsonl"
        destination.write_text(self.snapshot_text, encoding="utf-8")
        interactions = publish_dir / "interactions.jsonl"
        interactions.write_bytes(b'{"id":"interaction-1","kind":"comment"}\n')
        interactions_before = interactions.read_bytes()
        interactions_sha256 = hashlib.sha256(interactions_before).hexdigest()

        export_proc = run(
            ["bash", str(self.client / "export_beads_snapshot.sh"), str(destination)],
            cwd=self.client,
        )
        self.assertIn("exported compatible Beads snapshot", export_proc.stdout)
        published_lines = [
            line
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line
        ]
        published = [json.loads(line) for line in published_lines]
        published_by_id = {record["id"]: record for record in published}
        published_lines_by_id = {
            record["id"]: line
            for record, line in zip(published, published_lines, strict=True)
        }
        self.assertEqual(
            set(published_by_id),
            {record["id"] for record in self.snapshot_records},
        )
        self.assertEqual(
            published_lines_by_id["T37-legacy"],
            self.snapshot_lines_by_id["T37-legacy"],
        )
        self.assertEqual(
            published_by_id["T37-legacy"]["status"],
            "tombstone",
        )
        self.assertEqual(
            published_by_id["T37-legacy.1"]["dependencies"],
            next(
                record["dependencies"]
                for record in self.snapshot_records
                if record["id"] == "T37-legacy.1"
            ),
        )
        self.assertEqual(interactions.read_bytes(), interactions_before)
        self.assertEqual(
            hashlib.sha256(interactions.read_bytes()).hexdigest(),
            interactions_sha256,
        )

    def test_failed_import_restores_exact_pre_import_database(self) -> None:
        run(
            [
                "bd",
                "create",
                "local anchor",
                "--id",
                "CLIENT-anchor",
                "--silent",
            ],
            cwd=self.client,
        )
        before = self.base / "before-failed-import.jsonl"
        after = self.base / "after-failed-import.jsonl"
        run(["bd", "export", "-o", str(before)], cwd=self.client)

        real_bd = shutil.which("bd")
        self.assertIsNotNone(real_bd)
        wrapper = self.base / "bd-partial-import-failure"
        wrapper.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "import" ]]; then
  "${REAL_BD}" create "forced partial mutation" \
    --id CLIENT-mutated --silent
  exit 42
fi
exec "${REAL_BD}" "$@"
""",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        env = dict(os.environ)
        env["BD_BIN"] = str(wrapper)
        env["REAL_BD"] = str(real_bd)

        proc = run(
            ["bash", "refresh_beads_sync.sh"],
            cwd=self.client,
            env=env,
            check=False,
        )

        self.assertEqual(proc.returncode, 42)
        self.assertIn("restored local bd database", proc.stderr)
        run(["bd", "export", "-o", str(after)], cwd=self.client)
        self.assertEqual(after.read_bytes(), before.read_bytes())
        self.assertEqual(
            hashlib.sha256(after.read_bytes()).hexdigest(),
            hashlib.sha256(before.read_bytes()).hexdigest(),
        )
        restored_anchor = run(
            ["bd", "show", "CLIENT-anchor", "--json"],
            cwd=self.client,
        )
        self.assertIn("local anchor", restored_anchor.stdout)
        missing_mutation = run(
            ["bd", "show", "CLIENT-mutated", "--json"],
            cwd=self.client,
            check=False,
        )
        self.assertNotEqual(missing_mutation.returncode, 0)

    def test_refresh_restores_existing_backup_configuration(self) -> None:
        prior_backup = self.base / "prior-backup"
        run(["bd", "backup", "init", str(prior_backup)], cwd=self.client)
        run(["bd", "backup", "sync"], cwd=self.client)
        before = json.loads(
            run(
                ["bd", "backup", "status", "--json"],
                cwd=self.client,
            ).stdout
        )
        try:
            proc = run(
                ["bash", "refresh_beads_sync.sh"],
                cwd=self.client,
            )
            self.assertIn("refreshed local bd database", proc.stdout)
            after = json.loads(
                run(
                    ["bd", "backup", "status", "--json"],
                    cwd=self.client,
                ).stdout
            )
            self.assertTrue(before["dolt"]["configured"])
            self.assertEqual(
                after["dolt"]["backup_url"],
                before["dolt"]["backup_url"],
            )
        finally:
            run(
                ["bd", "backup", "remove"],
                cwd=self.client,
                check=False,
            )

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
