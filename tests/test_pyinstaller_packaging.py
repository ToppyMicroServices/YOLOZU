import runpy
import unittest
from pathlib import Path

from yolozu import _LEGACY_SUBMODULE_ALIASES


class TestPyInstallerPackaging(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_hook_tracks_every_legacy_alias_target(self) -> None:
        hook = runpy.run_path(str(self.repo_root / "deploy" / "pyinstaller" / "hook-yolozu.py"))

        self.assertEqual(hook["hiddenimports"], sorted(set(_LEGACY_SUBMODULE_ALIASES.values())))

    def test_supported_builds_enable_the_hook_directory(self) -> None:
        flag = "--additional-hooks-dir deploy/pyinstaller"
        workflow = (self.repo_root / ".github" / "workflows" / "build_and_test.yml").read_text(encoding="utf-8")
        readme = (self.repo_root / "deploy" / "pyinstaller" / "README.md").read_text(encoding="utf-8")

        self.assertEqual(workflow.count(flag), 2)
        self.assertGreaterEqual(readme.count(flag), 4)


if __name__ == "__main__":
    unittest.main()
