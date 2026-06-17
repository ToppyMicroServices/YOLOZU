import unittest
from pathlib import Path


class TestManualWorkflowOpenings(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_workflow_opening_macro_defines_operator_template(self):
        main = (self.repo_root / "manual" / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\newcommand{\WorkflowOpening}", main)
        for label in (
            "Goal",
            "When to use",
            "Inputs",
            "Command",
            "Outputs",
            "How to verify",
            "Common failure modes",
        ):
            with self.subTest(label=label):
                self.assertIn(f"\\item[{label}]", main)

    def test_major_workflow_chapters_start_with_operator_opening(self):
        chapters = [
            "05_workflows_eval_export.tex",
            "07_training_run_contract.tex",
            "09_parity_bench_protocols.tex",
            "10_ttt_hessian.tex",
            "13_research_workflows.tex",
            "14_continual_learning.tex",
            "15_ttt_tent_mim.tex",
            "17_realtime_batch_inference.tex",
            "21_synthgen_repo_integration.tex",
        ]
        for chapter in chapters:
            with self.subTest(chapter=chapter):
                text = (self.repo_root / "manual" / "chapters" / chapter).read_text(encoding="utf-8")
                opening = text.find(r"\WorkflowOpening")
                first_section = text.find(r"\section")
                self.assertNotEqual(opening, -1, f"{chapter} is missing \\WorkflowOpening")
                self.assertNotEqual(first_section, -1, f"{chapter} has no section marker")
                self.assertLess(opening, first_section, f"{chapter} should put operator guidance before deep material")

    def test_overview_surfaces_learning_mode_comparison_early(self):
        overview = (self.repo_root / "manual" / "chapters" / "01_overview.tex").read_text(encoding="utf-8")
        marker = "Continual, online, and TTT in one line"
        self.assertIn(marker, overview)
        self.assertIn("Continual learning", overview)
        self.assertIn("Online learning", overview)
        self.assertIn("TTT", overview)
        self.assertNotIn("AI-first development", overview)


if __name__ == "__main__":
    unittest.main()
