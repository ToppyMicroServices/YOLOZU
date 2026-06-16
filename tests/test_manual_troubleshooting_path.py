import unittest
from pathlib import Path


class TestManualTroubleshootingPath(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.install_chapter = self.repo_root / "manual" / "chapters" / "02_installation.tex"
        self.troubleshooting_chapter = self.repo_root / "manual" / "chapters" / "11_troubleshooting.tex"

    def test_install_chapter_includes_first_failure_checklist(self):
        text = self.install_chapter.read_text(encoding="utf-8")
        for phrase in (
            "First-failure checklist",
            "Symptom",
            "Likely cause",
            "Verification command",
            "pip install yolozu",
            "missing import after install",
            "category_id",
            "predictions schema error",
            "ONNX Runtime model-load error",
            "TensorRT command fails locally",
            "metrics changed after protocol edit",
        ):
            self.assertIn(phrase, text)

    def test_troubleshooting_chapter_expands_common_failures(self):
        text = self.troubleshooting_chapter.read_text(encoding="utf-8")
        for phrase in (
            "Pip install failures",
            "Missing optional extras",
            "COCO \\cmd{category_id} mismatches",
            "Invalid predictions schema",
            "ONNX Runtime model-load failures",
            "TensorRT environment constraints",
            "Protocol-driven metric changes",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
