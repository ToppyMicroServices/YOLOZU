import unittest
from pathlib import Path


class TestTttAndLoraDocs(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.ttt_doc = (self.repo_root / "docs" / "ttt_protocol.md").read_text(encoding="utf-8")
        self.ttt_manual = (self.repo_root / "manual" / "chapters" / "15_ttt_tent_mim.tex").read_text(
            encoding="utf-8"
        )
        self.continual_doc = (self.repo_root / "docs" / "continual_learning.md").read_text(encoding="utf-8")
        self.continual_manual = (
            self.repo_root / "manual" / "chapters" / "14_continual_learning.tex"
        ).read_text(encoding="utf-8")

    def test_cotta_eata_sar_have_matching_docs_and_manual_depth(self):
        expectations = {
            "CoTTA": ("Continual Test-Time Adaptation", "EMA", "restoration"),
            "EATA": ("Efficient Test-Time Adaptation", "selected", "regularization"),
            "SAR": ("Sharpness-Aware", "perturb", "stable"),
        }
        for method, phrases in expectations.items():
            with self.subTest(method=method, surface="docs"):
                section = self._markdown_section(self.ttt_doc, f"### {method}")
                for phrase in phrases:
                    self.assertIn(phrase, section)
                self.assertIn("When to use", section)
                self.assertIn("Concrete repo result", section)
            with self.subTest(method=method, surface="manual"):
                section = self._latex_subsection(self.ttt_manual, method)
                for phrase in phrases:
                    self.assertIn(phrase, section)
                self.assertIn("When to use it", section)
                self.assertIn("Concrete repo result", section)

    def test_lora_update_scope_is_explained_in_docs_and_manual(self):
        for name, text in (("docs", self.continual_doc), ("manual", self.continual_manual)):
            with self.subTest(surface=name):
                self.assertIn("LoRA", text)
                self.assertIn("QLoRA", text)
                self.assertIn("frozen", text)
                self.assertIn("adapter", text)
                self.assertIn("replay", text)
                self.assertIn("self-distillation", text)
                self.assertIn("anti-forgetting", text)

    def _markdown_section(self, text: str, marker: str) -> str:
        start = text.find(marker)
        self.assertNotEqual(start, -1, f"missing markdown section: {marker}")
        next_start = text.find("\n### ", start + len(marker))
        if next_start == -1:
            next_start = text.find("\n## ", start + len(marker))
        return text[start : next_start if next_start != -1 else len(text)]

    def _latex_subsection(self, text: str, title: str) -> str:
        marker = rf"\subsection{{{title}}}"
        start = text.find(marker)
        self.assertNotEqual(start, -1, f"missing manual subsection: {marker}")
        next_start = text.find(r"\subsection{", start + len(marker))
        if next_start == -1:
            next_start = text.find(r"\section{", start + len(marker))
        return text[start : next_start if next_start != -1 else len(text)]


if __name__ == "__main__":
    unittest.main()
