import unittest
from pathlib import Path


class TestResearchLaneDocs(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.research_doc = self.repo_root / "docs" / "research_lanes.md"

    def test_research_landing_page_separates_stable_and_opt_in_lanes(self):
        text = self.research_doc.read_text(encoding="utf-8")

        self.assertIn("stable product lane is the evaluation layer", text)
        self.assertIn("already evaluated artifact", text)
        self.assertIn("opt-in workflows", text)
        self.assertIn("predictions interface contract", text)
        for term in (
            "Continual learning",
            "TTT / CTTA",
            "Offline distillation",
            "Hessian refinement",
            "SynthGen research handoff",
        ):
            self.assertIn(term, text)

    def test_entry_docs_point_to_research_landing_page(self):
        for rel in ("README.md", "README_jp.md", "Readme_zh.md", "docs/README.md", "docs/production_readiness.md"):
            text = (self.repo_root / rel).read_text(encoding="utf-8")
            self.assertIn("research_lanes.md", text, rel)


if __name__ == "__main__":
    unittest.main()
