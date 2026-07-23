import json
import unittest
from pathlib import Path

from yolozu.instance_segmentation_predictions import validate_instance_segmentation_predictions_payload
from yolozu.predictions import validate_predictions_payload
from yolozu.predictions.predictions import CURRENT_ENTRY_SCHEMA_VERSION
from yolozu.predictions.schema_governance import CURRENT_SCHEMA_VERSION
from yolozu.segmentation_predictions import validate_segmentation_predictions_payload


class TestSchemaGovernance(unittest.TestCase):
    def test_predictions_schema_copies_declare_current_versions(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_paths = [
            repo_root / "docs" / "schemas" / "predictions.schema.json",
            repo_root / "schemas" / "predictions.schema.json",
            repo_root / "yolozu" / "data" / "schemas" / "predictions.schema.json",
        ]
        schemas = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in schema_paths
        ]
        self.assertEqual(schemas[1:], [schemas[0], schemas[0]])

        definitions = schemas[0]["$defs"]
        wrapper_version = definitions["predictions_wrapper"]["properties"]["schema_version"]
        entry_version = definitions["prediction_entry"]["properties"]["schema_version"]
        self.assertEqual(wrapper_version["maximum"], CURRENT_SCHEMA_VERSION)
        self.assertEqual(entry_version["maximum"], CURRENT_ENTRY_SCHEMA_VERSION)

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


if __name__ == "__main__":
    unittest.main()
