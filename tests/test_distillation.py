import unittest

from yolozu.distillation import distill_predictions
from yolozu.simple_map import evaluate_map


class TestDistillation(unittest.TestCase):
    def test_distill_adds_teacher(self):
        records = [
            {
                "image": "img1.jpg",
                "labels": [
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                ],
            },
            {
                "image": "img2.jpg",
                "labels": [
                    {"class_id": 0, "cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2},
                ],
            },
        ]

        student = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.4, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            },
            {"image": "img2.jpg", "detections": []},
        ]

        teacher = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            },
            {
                "image": "img2.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2}},
                ],
            },
        ]

        base_map = evaluate_map(records, student, iou_thresholds=[0.5]).map50
        distilled, stats = distill_predictions(student, teacher, add_missing=True)
        distilled_map = evaluate_map(records, distilled, iou_thresholds=[0.5]).map50

        self.assertGreaterEqual(distilled_map, base_map)
        self.assertGreater(stats.added, 0)

    def test_distill_add_missing_guards(self):
        student = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.2, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                ],
            }
        ]
        teacher = [
            {
                "image": "img1.jpg",
                "detections": [
                    {"class_id": 0, "score": 0.95, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                    {"class_id": 0, "score": 0.90, "bbox": {"cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}},
                    {"class_id": 0, "score": 0.85, "bbox": {"cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1}},
                    {"class_id": 0, "score": 0.10, "bbox": {"cx": 0.3, "cy": 0.8, "w": 0.1, "h": 0.1}},
                ],
            }
        ]

        distilled, stats = distill_predictions(
            student,
            teacher,
            add_missing=True,
            teacher_min_score=0.8,
            max_added_per_image=1,
            add_duplicate_iou_threshold=0.8,
        )

        self.assertEqual(stats.added, 1)
        self.assertEqual(len(distilled[0]["detections"]), 2)

    def test_distill_param_validation(self):
        with self.assertRaises(ValueError):
            distill_predictions([], [], iou_threshold=1.1)
        with self.assertRaises(ValueError):
            distill_predictions([], [], teacher_min_score=-0.1)
        with self.assertRaises(ValueError):
            distill_predictions([], [], max_added_per_image=-1)
        with self.assertRaises(ValueError):
            distill_predictions([], [], add_duplicate_iou_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
