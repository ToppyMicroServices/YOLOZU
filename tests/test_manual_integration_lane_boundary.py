import unittest
from pathlib import Path


class TestManualIntegrationLaneBoundary(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.main_tex = root / "manual" / "main.tex"
        self.overview = root / "manual" / "chapters" / "01_overview.tex"

    def test_integration_chapters_are_in_appendix_part(self):
        text = self.main_tex.read_text(encoding="utf-8")
        markers = {
            "quickstart": "\\part{User Quickstart}",
            "production": "\\part{Production Evaluation Manual}",
            "research": "\\part{Training and Research Workflows}",
            "appendices": "\\part{Maintainer, Automation, and Integration Appendices}",
            "mcp": "\\include{chapters/20_llm_mcp_integrations}",
            "synthgen": "\\include{chapters/21_synthgen_repo_integration}",
        }
        for marker in markers.values():
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

        positions = {name: text.index(marker) for name, marker in markers.items()}
        self.assertLess(positions["quickstart"], positions["production"])
        self.assertLess(positions["production"], positions["research"])
        self.assertLess(positions["research"], positions["appendices"])
        self.assertGreater(positions["mcp"], positions["appendices"])
        self.assertGreater(positions["synthgen"], positions["appendices"])

    def test_overview_marks_mcp_and_synthgen_as_optional_integrations(self):
        text = self.overview.read_text(encoding="utf-8")
        required_phrases = [
            "Optional integration appendices: MCP automation and SynthGen handoff.",
            "not part of the primary production evaluation lane",
            "client automation or synthetic-data intake",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
