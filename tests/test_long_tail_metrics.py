import unittest

from yolozu.long_tail_metrics import (
    build_fracal_stats,
    evaluate_long_tail_detection,
    fracal_calibrate_instance_segmentation,
    fracal_calibrate_predictions,
    la_calibrate_predictions,
    norcal_calibrate_predictions,
    temperature_calibrate_predictions,
)

try:
    import numpy as _np
    from PIL import Image as _PILImage

    _HAS_SEG_DEPS = True
except Exception:
    _HAS_SEG_DEPS = False
    _np = None
    _PILImage = None


def _mk_label(class_id: int, cx: float) -> dict:
    return {"class_id": int(class_id), "cx": float(cx), "cy": 0.5, "w": 0.2, "h": 0.2}


def _mk_det(class_id: int, score: float, cx: float) -> dict:
    return {
        "class_id": int(class_id),
        "score": float(score),
        "bbox": {"cx": float(cx), "cy": 0.5, "w": 0.2, "h": 0.2},
    }


class TestLongTailMetrics(unittest.TestCase):
    def _fixture(self):
        records = [
            {"image": "img1.jpg", "labels": [_mk_label(0, 0.30), _mk_label(1, 0.70)]},
            {"image": "img2.jpg", "labels": [_mk_label(0, 0.30)]},
            {"image": "img3.jpg", "labels": [_mk_label(0, 0.30), _mk_label(2, 0.70)]},
        ]
        preds = [
            {"image": "img1.jpg", "detections": [_mk_det(0, 0.90, 0.30), _mk_det(1, 0.40, 0.70)]},
            {"image": "img2.jpg", "detections": [_mk_det(0, 0.85, 0.30)]},
            {"image": "img3.jpg", "detections": [_mk_det(0, 0.88, 0.30), _mk_det(2, 0.40, 0.70)]},
        ]
        return records, preds

    def test_fracal_boosts_tail_vs_head(self):
        records, preds = self._fixture()
        calibrated, report = fracal_calibrate_predictions(records, preds, alpha=0.8, strength=1.0)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertEqual(report.get("method"), "fracal")

    def test_long_tail_report_contains_required_sections(self):
        records, preds = self._fixture()
        report = evaluate_long_tail_detection(records, preds)

        self.assertIn("per_class", report)
        self.assertIn("frequency_bins", report)
        self.assertIn("metrics", report)
        self.assertIn("macro", report["metrics"])
        self.assertIn("calibration", report["metrics"])

        self.assertTrue(report["per_class"])
        self.assertIn("head", report["frequency_bins"])
        self.assertIn("medium", report["frequency_bins"])
        self.assertIn("tail", report["frequency_bins"])

        macro = report["metrics"]["macro"]
        self.assertIn("ap50", macro)
        self.assertIn("ar50", macro)

        calib = report["metrics"]["calibration"]
        self.assertIn("ece", calib)
        self.assertIn("confidence_bias", calib)

    def test_fracal_uses_precomputed_class_counts_for_bbox(self):
        records, preds = self._fixture()
        counts = {"0": 100, "1": 1, "2": 1}
        calibrated, report = fracal_calibrate_predictions(records, preds, alpha=1.0, strength=1.0, class_counts=counts)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertEqual(report["class_counts"], {"0": 100, "1": 1, "2": 1})

    def test_la_boosts_tail_vs_head(self):
        records, preds = self._fixture()
        calibrated, report = la_calibrate_predictions(records, preds, tau=0.7)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertGreater(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertGreater((tail_after - tail_before), (head_after - head_before))
        self.assertEqual(report.get("method"), "la")

    def test_norcal_boosts_tail_vs_head(self):
        records, preds = self._fixture()
        calibrated, report = norcal_calibrate_predictions(records, preds, gamma=0.8)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertAlmostEqual(tail_after, tail_before, places=12)
        self.assertEqual(report.get("method"), "norcal")

    def test_temperature_scaling_softens_scores_when_t_gt_1(self):
        records, preds = self._fixture()
        calibrated, report = temperature_calibrate_predictions(records, preds, temperature=2.0)

        head_before = preds[0]["detections"][0]["score"]
        head_after = calibrated[0]["detections"][0]["score"]
        tail_before = preds[0]["detections"][1]["score"]
        tail_after = calibrated[0]["detections"][1]["score"]

        self.assertLess(head_after, head_before)
        self.assertGreater(tail_after, tail_before)
        self.assertEqual(report.get("method"), "temperature")
        self.assertAlmostEqual(float(report.get("config", {}).get("temperature", 0.0)), 2.0, places=8)

    def test_temperature_fit_path_returns_selected_temperature(self):
        records, preds = self._fixture()
        calibrated, report = temperature_calibrate_predictions(
            records,
            preds,
            temperature=1.0,
            fit_temperature=True,
            temperature_grid=[0.5, 1.0, 2.0],
        )

        self.assertTrue(calibrated)
        fit = report.get("fit") or {}
        self.assertTrue(bool(fit.get("fitted")))
        self.assertIn(float(fit.get("selected_temperature")), [0.5, 1.0, 2.0])

    def test_fracal_pose_stats_and_keypoints_are_preserved(self):
        records = [
            {"image": "img1.jpg", "labels": [_mk_label(0, 0.30), _mk_label(1, 0.70)]},
            {"image": "img2.jpg", "labels": [_mk_label(0, 0.30)]},
        ]
        preds = [
            {
                "image": "img1.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.90,
                        "bbox": {"cx": 0.3, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "keypoints": [[0.1, 0.2, 0.9], [0.3, 0.4, 0.8]],
                    },
                    {
                        "class_id": 1,
                        "score": 0.40,
                        "bbox": {"cx": 0.7, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "keypoints": [[0.6, 0.2, 0.7], [0.8, 0.4, 0.6]],
                    },
                ],
            }
        ]

        stats = build_fracal_stats(records, task="pose")
        self.assertEqual(stats.get("task"), "pose")
        self.assertEqual(stats.get("class_counts"), {"0": 2, "1": 1})

        calibrated, _report = fracal_calibrate_predictions(records, preds, alpha=1.0, strength=1.0)
        self.assertEqual(
            calibrated[0]["detections"][0]["keypoints"],
            preds[0]["detections"][0]["keypoints"],
        )
        self.assertEqual(
            calibrated[0]["detections"][1]["keypoints"],
            preds[0]["detections"][1]["keypoints"],
        )
        self.assertLess(calibrated[0]["detections"][0]["score"], preds[0]["detections"][0]["score"])
        self.assertGreater(calibrated[0]["detections"][1]["score"], preds[0]["detections"][1]["score"])

    @unittest.skipUnless(_HAS_SEG_DEPS, "instance-seg calibration requires numpy and Pillow")
    def test_build_fracal_stats_for_seg_and_calibrate_instances(self):
        records = [
            {
                "image": "img1.png",
                "mask": [
                    [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                ],
                "mask_classes": [0],
            },
            {
                "image": "img2.png",
                "mask": [
                    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                ],
                "mask_classes": [1],
            },
        ]
        preds = [
            {
                "image": "img1.png",
                "instances": [
                    {"class_id": 0, "score": 0.9, "mask": [[1, 1, 0], [1, 0, 0], [0, 0, 0]]},
                    {"class_id": 1, "score": 0.4, "mask": [[1, 1, 1], [0, 0, 0], [0, 0, 0]]},
                ],
            }
        ]

        stats = build_fracal_stats(records, task="seg")
        self.assertEqual(stats.get("task"), "seg")
        self.assertEqual(stats.get("class_counts"), {"0": 1, "1": 1})

        calibrated, report = fracal_calibrate_instance_segmentation(
            records,
            preds,
            alpha=0.6,
            strength=1.0,
            class_counts={"0": 10, "1": 1},
        )
        self.assertEqual(report.get("method"), "fracal")
        self.assertEqual(report.get("class_counts"), {"0": 10, "1": 1})
        self.assertIn("instances", calibrated[0])
        self.assertEqual(len(calibrated[0]["instances"]), 2)
        self.assertLess(calibrated[0]["instances"][0]["score"], preds[0]["instances"][0]["score"])
        self.assertGreater(calibrated[0]["instances"][1]["score"], preds[0]["instances"][1]["score"])


if __name__ == "__main__":
    unittest.main()
