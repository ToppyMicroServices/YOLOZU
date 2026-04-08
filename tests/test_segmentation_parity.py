import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

from yolozu.eval.segmentation_parity import compare_segmentation_predictions


class TestSegmentationParity(TestCase):
    def test_compare_segmentation_predictions_reports_drift(self):
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"segmentation parity deps unavailable: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            ref_mask = root / "ref.png"
            cand_mask = root / "cand.png"
            Image.fromarray(np.array([[0, 1], [1, 1]], dtype=np.uint8)).save(ref_mask)
            Image.fromarray(np.array([[0, 1], [0, 1]], dtype=np.uint8)).save(cand_mask)

            ref_pred = root / "ref.json"
            cand_pred = root / "cand.json"
            ref_pred.write_text(json.dumps({"sample0": ref_mask.name}), encoding="utf-8")
            cand_pred.write_text(json.dumps({"sample0": cand_mask.name}), encoding="utf-8")

            report = compare_segmentation_predictions(reference=ref_pred, candidate=cand_pred, mismatch_atol=0.0)
            self.assertFalse(report["ok"])
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["results"][0]["pixels_mismatched"], 1)
            self.assertAlmostEqual(report["results"][0]["mismatch_rate"], 0.25)


if __name__ == "__main__":
    main()
