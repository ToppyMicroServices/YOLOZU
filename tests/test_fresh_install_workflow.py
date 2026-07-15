from __future__ import annotations

import unittest
from pathlib import Path


class FreshInstallWorkflowTests(unittest.TestCase):
    def test_public_pypi_matrix_covers_current_supported_interpreters_and_oses(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github" / "workflows" / "fresh_install_journey.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("ubuntu-latest", workflow)
        self.assertIn("macos-14", workflow)
        for version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(f'- "{version}"', workflow)
        self.assertIn("scripts/fresh_install_journey.sh", workflow)
        self.assertIn("YOLOZU_PACKAGE_SPEC", workflow)
        self.assertIn("actions/upload-artifact@043fb46", workflow)


if __name__ == "__main__":
    unittest.main()
