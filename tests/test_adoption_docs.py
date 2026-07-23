from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AdoptionDocsTests(unittest.TestCase):
    def test_baseline_preserves_privacy_and_automation_boundaries(self) -> None:
        guide = (ROOT / "docs/adoption/README.md").read_text(encoding="utf-8")
        baseline = (ROOT / "docs/adoption/2026-07-23-baseline.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("does not include product telemetry", guide)
        self.assertIn("automation-sensitive", guide)
        self.assertIn("Human adoption from", baseline)
        self.assertIn("`unknown`, not 72", baseline)
        self.assertIn("## 30-day targets", baseline)
        self.assertIn("## 90-day targets", baseline)

    def test_docs_index_links_the_measurement_guide(self) -> None:
        docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
        self.assertIn("[`adoption/README.md`](adoption/README.md)", docs_index)


if __name__ == "__main__":
    unittest.main()
