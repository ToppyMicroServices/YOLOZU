import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions_transform import fuse_detection_scores


class TestPredictionsTransformGates(unittest.TestCase):
    def test_fuse_score_and_preserve(self):
        entries = [
            {
                "image": "img.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.8, "score_tmp_sym": 0.5, "sigma_z": 0.1, "sigma_rot": 0.2}
                ],
            }
        ]
        out = fuse_detection_scores(entries, weights={"det": 1.0, "tmp": 0.5, "unc": 1.0})
        det = out.entries[0]["detections"][0]
        self.assertAlmostEqual(det["score_det"], 0.8, places=7)
        self.assertAlmostEqual(det["score"], 0.8 + 0.25 - 0.3, places=7)

    def test_template_gate_filters(self):
        entries = [
            {
                "image": "img.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.8, "score_tmp_sym": 0.4, "sigma_z": 0.0, "sigma_rot": 0.0}
                ],
            }
        ]
        out = fuse_detection_scores(entries, template_gate_enabled=True, template_gate_tau=0.5)
        self.assertEqual(len(out.entries[0]["detections"]), 0)

    def test_min_score_and_topk(self):
        entries = [
            {
                "image": "img.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.9, "score_tmp_sym": 0.0},
                    {"class_id": 1, "score": 0.2, "score_tmp_sym": 0.0},
                    {"class_id": 2, "score": 0.7, "score_tmp_sym": 0.0},
                ],
            }
        ]
        out = fuse_detection_scores(entries, weights={"det": 1.0, "tmp": 0.0, "unc": 0.0}, min_score=0.5, topk_per_image=1)
        dets = out.entries[0]["detections"]
        self.assertEqual(len(dets), 1)
        self.assertEqual(int(dets[0]["class_id"]), 0)

    def test_fuse_preserves_entry_metadata(self):
        entries = [
            {
                "image": "img.jpg",
                "image_size": {"width": 320, "height": 240},
                "preprocess": {"method": "resize"},
                "detections": [
                    {"class_id": 0, "score": 0.9, "score_tmp_sym": 0.0},
                ],
            }
        ]
        out = fuse_detection_scores(entries, weights={"det": 1.0, "tmp": 0.0, "unc": 0.0})
        entry = out.entries[0]
        self.assertIn("image_size", entry)
        self.assertIn("preprocess", entry)
        self.assertEqual(entry["image_size"]["width"], 320)

    def test_topk_tie_break_is_deterministic(self):
        entries = [
            {
                "image": "img.jpg",
                "detections": [
                    {"class_id": 2, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                    {"class_id": 1, "score": 0.8, "bbox": {"cx": 0.6, "cy": 0.5, "w": 0.2, "h": 0.2}},
                    {"class_id": 1, "score": 0.8, "bbox": {"cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            }
        ]
        out = fuse_detection_scores(
            entries,
            weights={"det": 1.0, "tmp": 0.0, "unc": 0.0},
            topk_per_image=2,
        )
        kept = out.entries[0]["detections"]
        self.assertEqual(len(kept), 2)
        self.assertEqual(int(kept[0]["class_id"]), 1)
        self.assertAlmostEqual(float(kept[0]["bbox"]["cx"]), 0.4, places=6)
        self.assertEqual(int(kept[1]["class_id"]), 1)
        self.assertAlmostEqual(float(kept[1]["bbox"]["cx"]), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
