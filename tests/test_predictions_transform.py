import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions_transform import normalize_class_ids


class TestPredictionsTransform(unittest.TestCase):
    def test_category_id_to_class_id(self):
        classes_json = {
            "category_id_to_class_id": {"1": 0, "3": 1},
            "class_id_to_category_id": {"0": 1, "1": 3},
            "class_names": ["a", "b"],
        }
        entries = [
            {"image": "x.jpg", "detections": [{"category_id": 3, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}]},
        ]
        out = normalize_class_ids(entries, classes_json=classes_json)
        self.assertEqual(out.entries[0]["detections"][0]["class_id"], 1)

    def test_assume_class_id_is_category_id(self):
        classes_json = {"category_id_to_class_id": {"90": 79}, "class_id_to_category_id": {}, "class_names": []}
        entries = [{"image": "x.jpg", "detections": [{"class_id": 90, "score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}]}]
        out = normalize_class_ids(entries, classes_json=classes_json, assume_class_id_is_category_id=True)
        self.assertEqual(out.entries[0]["detections"][0]["class_id"], 79)

    def test_invalid_class_id_and_category_id_warn_instead_of_crashing(self):
        classes_json = {
            "category_id_to_class_id": {"1": 0, "3": 1},
            "class_id_to_category_id": {"0": 1, "1": 3},
            "class_names": ["a", "b"],
        }
        entries = [
            {
                "image": "x.jpg",
                "detections": [
                    {"class_id": {"bad": True}, "score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}},
                    {"category_id": {"bad": True}, "score": 0.2, "bbox": {"cx": 0.4, "cy": 0.5, "w": 0.1, "h": 0.1}},
                ],
            }
        ]
        out = normalize_class_ids(entries, classes_json=classes_json, assume_class_id_is_category_id=True)
        self.assertEqual(len(out.entries[0]["detections"]), 2)
        self.assertIn("invalid class_id", " ".join(out.warnings))
        self.assertIn("invalid category_id", " ".join(out.warnings))


if __name__ == "__main__":
    unittest.main()
