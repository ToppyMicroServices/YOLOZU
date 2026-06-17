import re
import unittest
from pathlib import Path


REPRESENTATIVE_MANUAL_PROSE = [
    "manual/chapters/01_overview.tex",
    "manual/chapters/07_training_run_contract.tex",
    "manual/chapters/09_parity_bench_protocols.tex",
    "manual/chapters/13_research_workflows.tex",
]


class TestManualReadability(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_overview_uses_automation_framing(self):
        text = self._read("manual/chapters/01_overview.tex")
        self.assertIn(r"\section{Automation-Oriented Design Intent}", text)
        self.assertNotRegex(text, re.compile(r"AI[- ]first", re.IGNORECASE))
        self.assertNotIn("AI agents", text)

    def test_representative_manual_sections_avoid_formal_fillers(self):
        forbidden_terms = [
            "elucidates",
            "utilization",
            "necessitate",
            "paramount",
            "comprehensive",
            "subsequently",
            "disparate",
            "encompasses",
        ]
        hits = []
        for rel_path in REPRESENTATIVE_MANUAL_PROSE:
            text = self._read(rel_path)
            for term in forbidden_terms:
                if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                    hits.append(f"{rel_path}: {term}")
        self.assertEqual(hits, [], "formal manual phrasing found:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
