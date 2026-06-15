import unittest
from pathlib import Path


class TestManualPredictionsExamples(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.chapter = self.repo_root / "manual" / "chapters" / "03_concepts_contracts.tex"

    def test_chapter_includes_minimal_task_examples(self):
        text = self.chapter.read_text(encoding="utf-8")
        for phrase in (
            "Minimal task examples",
            "Detection",
            "Instance segmentation",
            "Keypoints",
            "Depth artifact",
            "6DoF pose",
            "benchmark_depth_predictions_artifact",
            "rot6d",
            "keypoints_format",
        ):
            self.assertIn(phrase, text)

    def test_chapter_includes_metric_glossary(self):
        text = self.chapter.read_text(encoding="utf-8")
        for term in (
            "mAP",
            "IoU",
            "NMS",
            "OKS",
            "PCK",
            "ADD",
            "ADDS",
            "abs_rel",
            "RMSE",
            "delta1",
            "calibration",
            "parity",
            "protocol pinning",
        ):
            self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
