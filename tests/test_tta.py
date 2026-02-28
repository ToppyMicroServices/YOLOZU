import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions_transform import apply_tta


class TestTTA(unittest.TestCase):
    def test_tta_off_keeps_schema(self):
        entries = [
            {
                "image": "x.jpg",
                "meta": {"foo": 1},
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.9,
                        "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1},
                        "extra": "keep",
                    }
                ],
            }
        ]

        out = apply_tta(entries, enabled=False, seed=123, flip_prob=1.0)
        self.assertEqual(set(out.entries[0].keys()), set(entries[0].keys()))
        self.assertNotIn("tta_mask", out.entries[0])
        self.assertEqual(out.entries[0]["detections"][0]["extra"], "keep")
        self.assertEqual(out.entries[0]["detections"][0]["bbox"]["cx"], 0.2)

    def test_mask_determinism_with_seed(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": 1, "score": 0.9, "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1}},
                    {"class_id": 2, "score": 0.8, "bbox": {"cx": 0.6, "cy": 0.4, "w": 0.2, "h": 0.2}},
                ],
            }
        ]

        a = apply_tta(entries, enabled=True, seed=42, flip_prob=0.5)
        b = apply_tta(entries, enabled=True, seed=42, flip_prob=0.5)
        self.assertEqual(a.entries[0]["tta_mask"], b.entries[0]["tta_mask"])
        self.assertEqual(len(a.entries[0]["tta_mask"]), len(entries[0]["detections"]))

    def test_norm_only_updates_only_norm(self):
        entries = [
            {
                "image": "x.jpg",
                "image_size": {"width": 100, "height": 50},
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.9,
                        "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1},
                        "bbox_abs": {"cx": 20.0, "cy": 15.0, "w": 10.0, "h": 5.0},
                    }
                ],
            }
        ]

        norm_only = apply_tta(entries, enabled=True, seed=7, flip_prob=1.0, norm_only=True)
        det = norm_only.entries[0]["detections"][0]
        self.assertAlmostEqual(det["bbox"]["cx"], 0.8)
        self.assertAlmostEqual(det["bbox_abs"]["cx"], 20.0)

        full = apply_tta(entries, enabled=True, seed=7, flip_prob=1.0, norm_only=False)
        det_full = full.entries[0]["detections"][0]
        self.assertAlmostEqual(det_full["bbox"]["cx"], 0.8)
        self.assertAlmostEqual(det_full["bbox_abs"]["cx"], 80.0)

    def test_tta_flips_keypoints_norm_and_abs(self):
        entries = [
            {
                "image": "x.jpg",
                "image_size": {"width": 100, "height": 50},
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.9,
                        "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1},
                        "keypoints": [
                            {"x": 0.25, "y": 0.2, "v": 2},
                            [80.0, 20.0, 2],
                        ],
                    }
                ],
            }
        ]

        out = apply_tta(
            entries,
            enabled=True,
            seed=7,
            flip_prob=1.0,
            norm_only=False,
            flip_keypoints=True,
        )
        det = out.entries[0]["detections"][0]
        self.assertAlmostEqual(float(det["keypoints"][0]["x"]), 0.75)
        self.assertAlmostEqual(float(det["keypoints"][1][0]), 20.0)

    def test_tta_flips_pose_offsets(self):
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {
                        "class_id": 1,
                        "score": 0.9,
                        "bbox": {"cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1},
                        "offsets": [0.15, -0.2],
                    }
                ],
            }
        ]

        out = apply_tta(
            entries,
            enabled=True,
            seed=7,
            flip_prob=1.0,
            flip_pose_offsets=True,
        )
        det = out.entries[0]["detections"][0]
        self.assertAlmostEqual(float(det["offsets"][0]), -0.15)
        self.assertAlmostEqual(float(det["offsets"][1]), -0.2)


if __name__ == "__main__":
    unittest.main()
