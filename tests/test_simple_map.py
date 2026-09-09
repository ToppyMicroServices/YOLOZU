import unittest
from unittest.mock import patch

from yolozu.eval import simple_map
from yolozu.simple_map import evaluate_map


class TestSimpleMap(unittest.TestCase):
    def test_thresholds_keep_independent_matches_and_aliases(self):
        bbox = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        records = [
            {"image": "/dataset/images/img1.jpg", "labels": [{"class_id": 0, **bbox}]},
            {"image": "/dataset/images/img2.jpg", "labels": [{"class_id": 1, **bbox}]},
        ]
        predictions = [
            {"image": "img1.jpg", "detections": [
                {"class_id": 0, "score": 0.8, "bbox": bbox},
                {"class_id": 0, "score": 0.9, "bbox": {**bbox, "cx": 0.55}},
                {"class_id": 2, "score": 0.6, "bbox": bbox},
            ]},
            {"image": "unknown.jpg", "detections": [{"class_id": 0, "score": 0.7, "bbox": bbox}]},
        ]
        result = evaluate_map(records, iter(predictions), iou_thresholds=[0.75, 0.5])
        self.assertEqual(result.per_class[0], {"ap@0.75": 0.5, "ap@0.50": 1.0})
        self.assertEqual(result.per_class[1], {"ap@0.75": 0.0, "ap@0.50": 0.0})
        self.assertEqual(result.per_class[2], {"ap@0.75": 0.0, "ap@0.50": 0.0})
        self.assertAlmostEqual(result.map50, 1.0 / 3.0)
        self.assertAlmostEqual(result.map50_95, 0.25)

    def test_iou_work_does_not_grow_with_threshold_count(self):
        bbox = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        records = [{"image": "img.jpg", "labels": [{"class_id": 0, **bbox}]}]
        predictions = [{"image": "img.jpg", "detections": [
            {"class_id": 0, "score": 0.9, "bbox": bbox},
            {"class_id": 0, "score": 0.8, "bbox": bbox},
        ]}]
        with patch.object(simple_map, "_bbox_iou_cxcywh_norm", wraps=simple_map._bbox_iou_cxcywh_norm) as iou:
            result = evaluate_map(records, predictions, iou_thresholds=[0.5 + 0.05 * idx for idx in range(10)])
        self.assertEqual(iou.call_count, 2)
        self.assertEqual(result.map50_95, 1.0)

    def test_distillation_improves_map(self):
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

        student_map = evaluate_map(records, student, iou_thresholds=[0.5]).map50
        teacher_map = evaluate_map(records, teacher, iou_thresholds=[0.5]).map50
        self.assertGreaterEqual(teacher_map, student_map)

    def test_map_matches_windows_style_prediction_image(self):
        records = [
            {
                "image": "/dataset/images/img1.jpg",
                "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            }
        ]
        preds = [
            {
                "image": r"C:\dataset\images\img1.jpg",
                "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
            }
        ]
        result = evaluate_map(records, preds, iou_thresholds=[0.5])
        self.assertAlmostEqual(float(result.map50), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
