import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions_transform import apply_ttt_lite, summarize_task_coverage


class TestPredictionsTransformExtensions(unittest.TestCase):
    def test_apply_ttt_lite_adjusts_scores(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.2, "entropy": 0.0},
                    {"class_id": 1, "score": 0.8, "entropy": 1.0},
                ],
            }
        ]
        out = apply_ttt_lite(
            entries,
            enabled=True,
            temperature=1.0,
            entropy_weight=0.5,
            minmax_norm=True,
        )
        det0 = out.entries[0]["detections"][0]
        det1 = out.entries[0]["detections"][1]
        self.assertAlmostEqual(float(det0["score"]), 0.0, places=6)
        self.assertAlmostEqual(float(det1["score"]), 0.5, places=6)
        self.assertAlmostEqual(float(det0["score_raw"]), 0.2, places=6)
        self.assertAlmostEqual(float(det1["score_raw"]), 0.8, places=6)

    def test_summarize_task_coverage(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "mask_path": "masks/x.png",
                        "keypoints": [{"x": 0.4, "y": 0.4, "v": 2}],
                        "log_z": 0.1,
                        "rot6d": [0.0] * 6,
                    }
                ],
            }
        ]
        coverage = summarize_task_coverage(entries)
        supported = coverage.get("supported") or {}
        self.assertTrue(bool(supported.get("bbox")))
        self.assertTrue(bool(supported.get("segmentation")))
        self.assertTrue(bool(supported.get("keypoints")))
        self.assertTrue(bool(supported.get("depth")))
        self.assertTrue(bool(supported.get("pose6d")))

    def test_apply_ttt_lite_non_finite_score_is_guarded(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 0, "score": float("nan")},
                    {"class_id": 1, "score": float("inf")},
                ],
            }
        ]
        out = apply_ttt_lite(entries, enabled=True, temperature=1.0, entropy_weight=0.0, minmax_norm=True)
        scores = [float(d.get("score", -1.0)) for d in out.entries[0]["detections"]]
        self.assertEqual(scores, [0.0, 0.0])
        self.assertTrue(any("non-finite" in str(w) for w in out.warnings))

    def test_apply_ttt_lite_torch_backend_matches_python(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not installed")

        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.1, "entropy": 0.2},
                    {"class_id": 1, "score": 0.8, "entropy": 0.5},
                    {"class_id": 2, "score": 0.4, "entropy": 0.1},
                ],
            }
        ]
        py = apply_ttt_lite(
            entries,
            enabled=True,
            temperature=0.9,
            entropy_weight=0.2,
            minmax_norm=True,
            prefer_torch=False,
        )
        th = apply_ttt_lite(
            entries,
            enabled=True,
            temperature=0.9,
            entropy_weight=0.2,
            minmax_norm=True,
            prefer_torch=True,
        )
        py_scores = [float(d.get("score", 0.0)) for d in py.entries[0]["detections"]]
        th_scores = [float(d.get("score", 0.0)) for d in th.entries[0]["detections"]]
        self.assertEqual(len(py_scores), len(th_scores))
        for a, b in zip(py_scores, th_scores):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
