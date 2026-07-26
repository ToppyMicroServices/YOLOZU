import unittest

from yolozu.inference import export_predictions_cli as export_predictions


class TestExportPredictionsTTALog(unittest.TestCase):
    def test_summarize_tta(self):
        preds = [
            {"image": "x.jpg", "detections": [], "tta_mask": [True, False, True]},
            {"image": "y.jpg", "detections": [], "tta_mask": [False]},
        ]
        summary = export_predictions._summarize_tta(preds, warnings=["w1"])
        self.assertEqual(summary["detections"], 4)
        self.assertEqual(summary["applied"], 2)
        self.assertAlmostEqual(summary["applied_ratio"], 0.5)
        self.assertEqual(summary["warnings"], ["w1"])

    def test_merge_model_tta_branches(self):
        base = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.6, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}
                ],
            }
        ]
        aug = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.8, "bbox": {"cx": 0.52, "cy": 0.5, "w": 0.2, "h": 0.2}}
                ],
            }
        ]
        merged, warnings = export_predictions._merge_model_tta_branches(
            base,
            aug,
            iou_threshold=0.5,
            max_detections=50,
        )
        self.assertFalse(warnings)
        self.assertEqual(len(merged), 1)
        dets = merged[0]["detections"]
        self.assertEqual(len(dets), 1)
        self.assertGreater(float(dets[0]["score"]), 0.6)


if __name__ == "__main__":
    unittest.main()
