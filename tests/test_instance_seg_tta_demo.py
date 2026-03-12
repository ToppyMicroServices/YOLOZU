import unittest

import numpy as np

from yolozu.demos.instance_seg_tta import _score_predictions, _select_best_case, _visual_metrics


class TestInstanceSegTTADemo(unittest.TestCase):
    def test_score_predictions_counts_tp_fp_fn(self):
        gt = [
            {"class_id": 1, "mask": np.array([[1, 1], [0, 0]], dtype=bool)},
            {"class_id": 2, "mask": np.array([[0, 0], [1, 1]], dtype=bool)},
        ]
        preds = [
            {"class_id": 1, "score": 0.9, "mask": np.array([[1, 1], [0, 0]], dtype=bool)},
            {"class_id": 9, "score": 0.2, "mask": np.array([[1, 0], [0, 0]], dtype=bool)},
        ]

        res = _score_predictions(preds, gt)

        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertAlmostEqual(float(res["mean_iou"]), 1.0, places=6)

    def test_select_best_case_prefers_tp_gain_then_fewer_fp(self):
        cases = [
            {
                "image_id": 1,
                "delta": {"tp": 0, "fn": -1, "fp": -1, "mean_iou": 0.1},
                "visual_delta": {"canvas_iou": 0.0, "f1": 0.0, "fp_area_ratio": 0.0, "fn_area_ratio": -0.1},
            },
            {
                "image_id": 2,
                "delta": {"tp": 1, "fn": 0, "fp": 1, "mean_iou": 0.01},
                "visual_delta": {"canvas_iou": 0.2, "f1": 0.2, "fp_area_ratio": 0.1, "fn_area_ratio": -0.1},
            },
            {
                "image_id": 3,
                "delta": {"tp": 1, "fn": 0, "fp": -1, "mean_iou": 0.0},
                "visual_delta": {"canvas_iou": 0.1, "f1": 0.1, "fp_area_ratio": -0.1, "fn_area_ratio": -0.1},
            },
        ]

        best = _select_best_case(cases)

        self.assertEqual(best["image_id"], 3)

    def test_visual_metrics_penalize_overpaint(self):
        gt = [{"class_id": 1, "mask": np.array([[1, 1], [0, 0]], dtype=bool)}]
        raw = [{"class_id": 1, "score": 0.9, "mask": np.array([[1, 1], [0, 0]], dtype=bool)}]
        tta = [{"class_id": 1, "score": 0.9, "mask": np.array([[1, 1], [1, 1]], dtype=bool)}]

        raw_vis = _visual_metrics(raw, gt)
        tta_vis = _visual_metrics(tta, gt)

        self.assertGreater(raw_vis["canvas_iou"], tta_vis["canvas_iou"])
        self.assertGreater(raw_vis["precision"], tta_vis["precision"])
        self.assertLess(raw_vis["fp_area_ratio"], tta_vis["fp_area_ratio"])


if __name__ == "__main__":
    unittest.main()
