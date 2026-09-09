import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDatasetValidator(unittest.TestCase):
    def test_nonfinite_coordinates_are_rejected(self):
        from yolozu.dataset_validator import validate_dataset_records

        for key in ("cx", "cy", "w", "h"):
            for value in (float("nan"), float("inf"), -float("inf")):
                for mode in ("fail", "warn"):
                    with self.subTest(key=key, value=value, mode=mode):
                        label = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, key: value}
                        result = validate_dataset_records(
                            [{"image": "unused.jpg", "labels": [label]}], mode=mode, check_images=False
                        )
                        messages = result.errors if mode == "fail" else result.warnings
                        self.assertTrue(any(f".{key}: must be a finite number" in msg for msg in messages))
                        self.assertEqual(result.ok(), mode == "warn")

    def test_nonfinite_image_dimensions_are_rejected(self):
        from yolozu.dataset_validator import validate_dataset_records

        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                result = validate_dataset_records(
                    [{"image": "unused.jpg", "labels": [], "image_hw": [value, 10]}], check_images=False
                )
                self.assertFalse(result.ok())
                self.assertTrue(any("image_hw" in msg for msg in result.errors))

    def test_falsey_nonlist_labels_are_rejected(self):
        from yolozu.dataset_validator import validate_dataset_records

        for labels in ({}, "", 0, False):
            with self.subTest(labels=labels):
                result = validate_dataset_records(
                    [{"image": "unused.jpg", "labels": labels}], check_images=False
                )
                self.assertFalse(result.ok())
                self.assertIn("records[0]: labels must be a list", result.errors)

    def test_generator_is_validated_incrementally(self):
        from yolozu.dataset_validator import validate_dataset_records

        with patch("yolozu.datasets.dataset_validator.get_image_size", return_value=(10, 10)) as size_reader:
            with patch("yolozu.datasets.dataset_validator.Path.exists", return_value=True):
                def records():
                    for idx in range(4):
                        self.assertEqual(size_reader.call_count, idx)
                        yield {"image": f"{idx}.jpg", "labels": []}

                result = validate_dataset_records(records())
        self.assertTrue(result.ok(), result.errors)
        self.assertEqual(size_reader.call_count, 4)

    def test_empty_dataset_is_always_an_error(self):
        from yolozu.dataset_validator import validate_dataset_records

        for mode in ("fail", "warn"):
            with self.subTest(mode=mode):
                res = validate_dataset_records([], strict=True, mode=mode)
                self.assertFalse(res.ok())
                self.assertTrue(any("no records" in error for error in res.errors))
                generator_result = validate_dataset_records(iter(()), strict=True, mode=mode)
                self.assertFalse(generator_result.ok())
                self.assertEqual(res.errors, generator_result.errors)

    def test_strict_allows_only_float_rounding_at_image_edge(self):
        from yolozu.dataset_validator import validate_dataset_records

        base = {
            "image": "unused.jpg",
            "labels": [{"class_id": 0, "cx": 0.4999995, "cy": 0.5, "w": 1.0, "h": 1.0}],
        }
        within_tolerance = validate_dataset_records(
            [base],
            strict=True,
            mode="fail",
            check_images=False,
        )
        self.assertTrue(within_tolerance.ok(), within_tolerance.errors)

        outside_tolerance = {
            **base,
            "labels": [{"class_id": 0, "cx": 0.4999985, "cy": 0.5, "w": 1.0, "h": 1.0}],
        }
        result = validate_dataset_records(
            [outside_tolerance],
            strict=True,
            mode="fail",
            check_images=False,
        )
        self.assertFalse(result.ok())
        self.assertTrue(any("extends outside" in error for error in result.errors))

    def test_strict_rejects_out_of_range_bbox(self):
        from PIL import Image

        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img = root / "images" / "train2017" / "0001.jpg"
            Image.new("RGB", (10, 10)).save(img)

            (root / "labels" / "train2017" / "0001.txt").write_text("0 1.2 0.5 0.2 0.2\n")

            manifest = build_manifest(root, split="train2017")
            res = validate_dataset_records(manifest["images"], strict=True, mode="fail")
            self.assertFalse(res.ok())
            self.assertTrue(any("out of range" in e for e in res.errors))

    def test_warn_mode_downgrades_errors(self):
        from PIL import Image

        from yolozu.dataset import build_manifest
        from yolozu.dataset_validator import validate_dataset_records

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True)
            (root / "labels" / "train2017").mkdir(parents=True)

            img = root / "images" / "train2017" / "0001.jpg"
            Image.new("RGB", (10, 10)).save(img)

            (root / "labels" / "train2017" / "0001.txt").write_text("0 1.2 0.5 0.2 0.2\n")

            manifest = build_manifest(root, split="train2017")
            res = validate_dataset_records(manifest["images"], strict=True, mode="warn")
            self.assertTrue(res.ok())
            self.assertEqual(res.errors, [])
            self.assertTrue(any("out of range" in w for w in res.warnings))


if __name__ == "__main__":
    unittest.main()
