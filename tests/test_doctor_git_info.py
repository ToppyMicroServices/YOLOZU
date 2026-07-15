from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from yolozu.core.doctor import _gather_git_info


class DoctorGitInfoTests(unittest.TestCase):
    def test_non_git_directory_is_not_reported_as_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            info = _gather_git_info(cwd=Path(td))

        self.assertEqual(info, {"head": None, "dirty": None})

    def test_clean_and_dirty_worktrees_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "YOLOZU test"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "test fixture"], cwd=root, check=True)

            clean = _gather_git_info(cwd=root)
            self.assertIsInstance(clean["head"], str)
            self.assertFalse(clean["dirty"])

            tracked.write_text("dirty\n", encoding="utf-8")
            dirty = _gather_git_info(cwd=root)
            self.assertEqual(dirty["head"], clean["head"])
            self.assertTrue(dirty["dirty"])


if __name__ == "__main__":
    unittest.main()
