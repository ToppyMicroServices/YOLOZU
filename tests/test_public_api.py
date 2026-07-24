import copy
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from yolozu.api import (
    InputError,
    OptionalDependencyError,
    PredictionsInput,
    PredictionsValidationError,
    evaluate_coco,
    validate_predictions,
)


class TestPublicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.dataset = cls.repo_root / "data" / "smoke"
        cls.predictions_path = cls.dataset / "predictions" / "predictions_dummy.json"
        cls.predictions_payload = json.loads(cls.predictions_path.read_text(encoding="utf-8"))

    def test_validate_predictions_payload_is_strict_by_default(self):
        payload = {
            "schema_version": 1,
            "predictions": [
                {
                    "schema_version": 2,
                    "image": "000000000009.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 1.5,
                            "bbox": {"cx": 1.2, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ],
        }

        with self.assertRaises(PredictionsValidationError) as ctx:
            validate_predictions(PredictionsInput.from_payload(payload))

        self.assertEqual(ctx.exception.code, "E_PREDICTIONS_INVALID")
        self.assertIn("score: must be in", ctx.exception.message)
        self.assertEqual(ctx.exception.to_dict()["category"], "PredictionsValidationError")

    def test_validate_predictions_explicit_repair_records_every_clamp(self):
        payload = {
            "schema_version": 1,
            "predictions": [
                {
                    "schema_version": 2,
                    "image": "000000000009.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 1.5,
                            "bbox": {"cx": 1.2, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ],
        }

        result = validate_predictions(payload, repair=True)

        self.assertEqual(result.mode, "repair")
        self.assertTrue(result.repair_enabled)
        self.assertEqual(result.entries[0]["detections"][0]["score"], 1.0)
        self.assertEqual(result.entries[0]["detections"][0]["bbox"]["cx"], 1.0)
        self.assertTrue(any("score: out of range" in warning for warning in result.warnings))
        self.assertTrue(any("bbox.cx: out of range" in warning for warning in result.warnings))
        json.dumps(result.to_dict())

    def test_relative_path_requires_explicit_base_dir(self):
        with self.assertRaises(InputError) as ctx:
            validate_predictions("predictions.json")
        self.assertEqual(ctx.exception.code, "E_RELATIVE_PATH")

    def test_evaluate_coco_accepts_path_and_payload_forms(self):
        path_result = evaluate_coco(
            self.dataset,
            self.predictions_path,
            split="val",
            max_images=2,
            dry_run=True,
        )
        payload_result = evaluate_coco(
            self.dataset,
            PredictionsInput.from_payload(self.predictions_payload, label="smoke payload"),
            split="val",
            max_images=2,
            dry_run=True,
        )

        for result in (path_result, payload_result):
            self.assertEqual(result.counts.dataset_images_total, 10)
            self.assertEqual(result.counts.images, 2)
            self.assertEqual(result.counts.prediction_images_total, 10)
            self.assertEqual(result.counts.prediction_images_evaluated, 2)
            self.assertEqual(result.counts.prediction_images_excluded, 8)
            self.assertEqual(result.counts.detections, 10)
            self.assertEqual(result.counts.detections_excluded, 40)
            self.assertTrue(any("known but unselected" in warning for warning in result.warnings))
            json.dumps(result.to_dict())

    def test_evaluate_coco_rejects_image_unknown_to_full_dataset(self):
        payload = copy.deepcopy(self.predictions_payload)
        payload["predictions"][0]["image"] = "not-in-dataset.jpg"

        with self.assertRaises(PredictionsValidationError) as ctx:
            evaluate_coco(
                self.dataset,
                payload,
                split="val",
                max_images=2,
                dry_run=True,
            )

        self.assertEqual(ctx.exception.code, "E_PREDICTION_UNKNOWN_IMAGE")

    def test_class_normalization_does_not_hide_malformed_detections(self):
        payload = {
            "predictions": [
                {
                    "image": "000000000009.jpg",
                    "detections": {"class_id": 1, "score": 0.9},
                }
            ]
        }

        with self.assertRaises(PredictionsValidationError) as ctx:
            evaluate_coco(
                self.dataset,
                payload,
                split="val",
                dry_run=True,
                classes=self.dataset / "labels" / "val" / "classes.json",
            )

        self.assertIn("detections: must be a list", ctx.exception.message)

    def test_evaluate_coco_surfaces_optional_dependency_error(self):
        with patch(
            "yolozu.api.evaluate_coco_map",
            side_effect=RuntimeError("pycocotools is required for COCO mAP evaluation"),
        ):
            with self.assertRaises(OptionalDependencyError) as ctx:
                evaluate_coco(
                    self.dataset,
                    self.predictions_path,
                    split="val",
                    max_images=2,
                )

        self.assertEqual(ctx.exception.code, "E_OPTIONAL_DEPENDENCY")
        self.assertEqual(ctx.exception.details["extra"], "coco")

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is not installed")
    def test_real_cocoeval_path_returns_serializable_metrics(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = evaluate_coco(
                self.dataset,
                self.predictions_path,
                split="val",
                max_images=2,
                dry_run=False,
            )

        self.assertEqual(captured.getvalue(), "")
        self.assertFalse(result.dry_run)
        self.assertIsInstance(result.metrics.map50_95, float)
        self.assertEqual(len(result.stats), 12)
        json.dumps(result.to_dict())

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is not installed")
    def test_real_cocoeval_repair_is_explicit_and_records_warning(self):
        payload = copy.deepcopy(self.predictions_payload)
        payload["predictions"][0]["detections"][0]["score"] = 1.5

        with self.assertRaises(PredictionsValidationError):
            evaluate_coco(
                self.dataset,
                payload,
                split="val",
                max_images=2,
                dry_run=False,
            )

        captured = io.StringIO()
        with redirect_stdout(captured):
            result = evaluate_coco(
                self.dataset,
                payload,
                split="val",
                max_images=2,
                dry_run=False,
                repair=True,
            )

        self.assertEqual(captured.getvalue(), "")
        self.assertTrue(result.repair)
        self.assertEqual(result.to_dict()["validation"]["mode"], "repair")
        self.assertTrue(any("score: out of range" in warning for warning in result.warnings))
        self.assertIsInstance(result.metrics.map50_95, float)
        self.assertEqual(len(result.stats), 12)


if __name__ == "__main__":
    unittest.main()
