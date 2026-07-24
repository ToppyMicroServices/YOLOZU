from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "link_beads_to_github.py"
SPEC = importlib.util.spec_from_file_location("link_beads_to_github", TOOL)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot load {TOOL}")
LINKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LINKER
SPEC.loader.exec_module(LINKER)


def _write_snapshot(path: Path, record: dict[str, object]) -> None:
    path.write_text(
        json.dumps(record, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class LinkBeadsToGitHubTests(unittest.TestCase):
    def test_snapshot_argument_is_required(self) -> None:
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                LINKER.parse_args([])
        self.assertEqual(raised.exception.code, 2)

    def test_dry_run_does_not_create_issue_or_update_beads(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            snapshot = Path(tempdir) / "pulled.jsonl"
            _write_snapshot(
                snapshot,
                {
                    "id": "T44-open",
                    "title": "new issue",
                    "status": "open",
                },
            )

            output = StringIO()
            with (
                mock.patch.object(
                    LINKER,
                    "find_exact_title_match",
                    return_value=None,
                ) as search,
                mock.patch.object(LINKER, "gh_create_issue") as create,
                mock.patch.object(LINKER, "bd_link_external_ref") as update,
                mock.patch.object(LINKER, "gh_close_issue") as close,
                redirect_stdout(output),
            ):
                result = LINKER.main(
                    [
                        "--repo",
                        "example/repo",
                        "--snapshot",
                        str(snapshot),
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            search.assert_called_once_with("example/repo", "new issue")
            create.assert_not_called()
            update.assert_not_called()
            close.assert_not_called()
            self.assertIn(
                "DRY: CREATE+LINK T44-open -> new GitHub issue",
                output.getvalue(),
            )

    def test_dry_run_exact_match_may_read_state_but_never_mutates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            snapshot = Path(tempdir) / "pulled.jsonl"
            _write_snapshot(
                snapshot,
                {
                    "id": "T44-closed",
                    "title": "existing issue",
                    "status": "closed",
                },
            )

            output = StringIO()
            with (
                mock.patch.object(
                    LINKER,
                    "find_exact_title_match",
                    return_value=17,
                ),
                mock.patch.object(
                    LINKER,
                    "gh_get_state",
                    return_value="open",
                ) as get_state,
                mock.patch.object(LINKER, "gh_create_issue") as create,
                mock.patch.object(LINKER, "bd_link_external_ref") as update,
                mock.patch.object(LINKER, "gh_close_issue") as close,
                redirect_stdout(output),
            ):
                result = LINKER.main(
                    [
                        "--repo",
                        "example/repo",
                        "--snapshot",
                        str(snapshot),
                        "--sync-close",
                        "--dry-run",
                    ]
                )

            self.assertEqual(result, 0)
            get_state.assert_called_once_with("example/repo", 17)
            create.assert_not_called()
            update.assert_not_called()
            close.assert_not_called()
            self.assertIn("DRY: LINK T44-closed -> #17", output.getvalue())
            self.assertIn("DRY: CLOSE T44-closed -> #17", output.getvalue())

    def test_non_dry_run_keeps_create_and_link_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            snapshot = Path(tempdir) / "pulled.jsonl"
            _write_snapshot(
                snapshot,
                {
                    "id": "T44-open",
                    "title": "new issue",
                    "status": "open",
                },
            )

            with (
                mock.patch.object(
                    LINKER,
                    "find_exact_title_match",
                    return_value=None,
                ),
                mock.patch.object(
                    LINKER,
                    "gh_create_issue",
                    return_value=23,
                ) as create,
                mock.patch.object(LINKER, "bd_link_external_ref") as update,
            ):
                result = LINKER.main(
                    [
                        "--repo",
                        "example/repo",
                        "--snapshot",
                        str(snapshot),
                    ]
                )

            self.assertEqual(result, 0)
            create.assert_called_once()
            update.assert_called_once_with("T44-open", 23)

    def test_missing_snapshot_fails_before_repository_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            missing = Path(tempdir) / "missing.jsonl"
            with mock.patch.object(LINKER, "detect_repo") as detect:
                with self.assertRaisesRegex(RuntimeError, "missing.jsonl"):
                    LINKER.main(["--snapshot", str(missing), "--dry-run"])
            detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
