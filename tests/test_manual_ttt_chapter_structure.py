import unittest
from pathlib import Path


class TestManualTttChapterStructure(unittest.TestCase):
    def setUp(self):
        self.chapter = Path(__file__).resolve().parents[1] / "manual" / "chapters" / "15_ttt_tent_mim.tex"
        self.assertTrue(self.chapter.is_file(), f"missing TTT manual chapter: {self.chapter}")
        self.text = self.chapter.read_text(encoding="utf-8")

    def test_operator_quickstart_appears_before_deep_material(self):
        quickstart = self.text.find(r"\section{TTT Quickstart (operator first)}")
        scope = self.text.find(r"\section{Scope and support}")
        catalog = self.text.find(r"\section{Method Catalog with concrete examples}")
        self.assertNotEqual(quickstart, -1, "TTT quickstart section is missing")
        self.assertNotEqual(scope, -1, "Scope and support section is missing")
        self.assertNotEqual(catalog, -1, "Method Catalog section is missing")
        self.assertLess(quickstart, scope)
        self.assertLess(quickstart, catalog)

    def test_quickstart_has_operator_acceptance_fields(self):
        quickstart = self._section_text(r"\section{TTT Quickstart (operator first)}", r"\section{Scope and support}")
        for phrase in (
            "Expected outputs",
            "How to verify the run",
            "Common failures",
            "tent_before_after_compare.md",
            "tent_ttt_log.json",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, quickstart)

    def test_audience_sections_are_named(self):
        ordered_sections = [
            r"\section{TTT Quickstart (operator first)}",
            r"\section{Method Catalog with concrete examples}",
            r"\section{Implementation Notes: YOLOZU rollout priority",
            r"\section{Evaluation and Failure Modes}",
        ]
        positions = []
        for marker in ordered_sections:
            pos = self.text.find(marker)
            self.assertNotEqual(pos, -1, f"missing section marker: {marker}")
            positions.append(pos)
        self.assertEqual(positions, sorted(positions), "TTT chapter audience sections should stay in reading order")

    def _section_text(self, start_marker: str, end_marker: str) -> str:
        start = self.text.find(start_marker)
        end = self.text.find(end_marker)
        self.assertNotEqual(start, -1, f"missing start marker: {start_marker}")
        self.assertNotEqual(end, -1, f"missing end marker: {end_marker}")
        self.assertLess(start, end)
        return self.text[start:end]


if __name__ == "__main__":
    unittest.main()
