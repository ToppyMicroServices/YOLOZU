import unittest

from yolozu.instance_segmentation_predictions import validate_instance_segmentation_predictions_payload
from yolozu.predictions import validate_predictions_payload
from yolozu.segmentation_predictions import validate_segmentation_predictions_payload


class TestSchemaGovernance(unittest.TestCase):
    def test_predictions_wrapped_without_schema_version_is_legacy_warning(self):
        payload = {
            "predictions": [
                {
                    "image": "a.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.8,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ]
        }
        res = validate_predictions_payload(payload, strict=False)
        self.assertTrue(any("schema_version missing" in w for w in res.warnings))

    def test_predictions_future_schema_version_rejected(self):
        payload = {
            "schema_version": 2,
            "predictions": [{"image": "a.jpg", "detections": [{"class_id": 0, "score": 0.8, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}],
        }
        with self.assertRaises(ValueError):
            validate_predictions_payload(payload, strict=False)

    def test_segmentation_future_schema_version_rejected(self):
        payload = {"schema_version": 2, "predictions": [{"id": "a", "mask": "a.png"}]}
        with self.assertRaises(ValueError):
            validate_segmentation_predictions_payload(payload)

    def test_instance_seg_future_schema_version_rejected(self):
        payload = {
            "schema_version": 2,
            "predictions": [{"image": "a.jpg", "instances": [{"class_id": 0, "score": 0.9, "mask": "a.png"}]}],
        }
        with self.assertRaises(ValueError):
            validate_instance_segmentation_predictions_payload(payload)

    def test_instance_seg_invalid_numeric_fields_are_rejected(self):
        payload = {
            "predictions": [
                {
                    "image": "a.jpg",
                    "instances": [
                        {"class_id": {"bad": True}, "score": 0.9, "mask": "a.png"},
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError):
            validate_instance_segmentation_predictions_payload(payload)

        payload2 = {
            "predictions": [
                {
                    "image": "a.jpg",
                    "instances": [
                        {"class_id": 0, "score": {"bad": True}, "mask": "a.png"},
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError):
            validate_instance_segmentation_predictions_payload(payload2)


if __name__ == "__main__":
    unittest.main()
