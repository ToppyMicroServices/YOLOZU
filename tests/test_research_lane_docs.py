import json
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
        for rel in ("README.md", "Readme_jp.md", "Readme_zh.md", "docs/README.md", "docs/production_readiness.md"):
            text = (self.repo_root / rel).read_text(encoding="utf-8")
            self.assertIn("research_lanes.md", text, rel)

    def test_research_workflow_docs_mark_methods_as_opt_in(self):
        expected = {
            "docs/ttt_protocol.md": ("opt-in research lane", "OFF by default", "Default validation"),
            "docs/continual_learning.md": ("research-oriented lane", "opt-in", "promotion decision report"),
            "docs/hessian_solver.md": ("opt-in research lane", "offline analysis or controlled studies", "--enable"),
            "docs/learning_features.md": ("opt-in research workflows", "default export/eval commands do not enable `--ttt`", "default evaluation does not run Hessian refinement"),
            "docs/tools_index.md": ("Research-only opt-in TTT extensions", "stable export/eval defaults do not enable these flags"),
        }

        for rel, phrases in expected.items():
            text = (self.repo_root / rel).read_text(encoding="utf-8")
            for phrase in phrases:
                self.assertIn(phrase, text, rel)

    def test_stable_entry_examples_do_not_enable_research_flags(self):
        protected_docs = ("README.md", "Readme_jp.md", "docs/README.md", "docs/cpu_only_dod.md")
        forbidden = (
            "--ttt",
            "refine_predictions_hessian.py",
            "train_continual.py",
            "yolozu demo continual",
        )

        hits = []
        for rel in protected_docs:
            text = (self.repo_root / rel).read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    hits.append(f"{rel}: {token}")

        self.assertEqual(hits, [], "stable entry examples should not enable research workflows")

    def test_research_lane_report_schema_is_documented(self):
        schema_path = self.repo_root / "docs" / "schemas" / "research_lane_report.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required = set(schema.get("required") or [])

        for field in (
            "kind",
            "lane",
            "stable_baseline_artifact",
            "research_output_artifact",
            "latency_overhead",
            "rollback",
            "promotion_gate",
        ):
            self.assertIn(field, required)

        research_doc = self.research_doc.read_text(encoding="utf-8")
        self.assertIn("schemas/research_lane_report.schema.json", research_doc)
        self.assertIn("stable_baseline_artifact", research_doc)


if __name__ == "__main__":
    unittest.main()
