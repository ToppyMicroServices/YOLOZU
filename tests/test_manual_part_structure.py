import unittest
from pathlib import Path


class TestManualPartStructure(unittest.TestCase):
    def setUp(self):
        self.main_tex = Path(__file__).resolve().parents[1] / "manual" / "main.tex"
        self.assertTrue(self.main_tex.exists())

    def test_manual_declares_reader_parts_in_order(self):
        text = self.main_tex.read_text(encoding="utf-8")
        parts = [
            "\\part{User Quickstart}",
            "\\part{Production Evaluation Manual}",
            "\\part{Training and Research Workflows}",
            "\\part{Maintainer, Automation, and Appendices}",
        ]
        positions = [text.index(part) for part in parts]
        self.assertEqual(positions, sorted(positions))

    def test_troubleshooting_is_in_quickstart_part(self):
        text = self.main_tex.read_text(encoding="utf-8")
        quickstart = text.index("\\part{User Quickstart}")
        production = text.index("\\part{Production Evaluation Manual}")
        troubleshooting = text.index("\\include{chapters/11_troubleshooting}")
        self.assertGreater(troubleshooting, quickstart)
        self.assertLess(troubleshooting, production)


if __name__ == "__main__":
    unittest.main()
