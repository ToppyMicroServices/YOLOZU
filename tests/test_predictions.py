import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.predictions import (
    canonicalize_predictions,
    load_predictions_entries,
    load_predictions_index,
    load_predictions_payload,
    validate_predictions_entries,
    validate_predictions_payload,
)


class TestPredictionsIO(unittest.TestCase):
    def test_load_entries_list_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = [{"image": "a.jpg", "detections": [{"score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
            p.write_text(json.dumps(payload))
            entries = load_predictions_entries(p)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["image"], "a.jpg")

    def test_load_entries_wrapped_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = {"predictions": [{"image": "a.jpg", "detections": []}], "meta": {"x": 1}}
            p.write_text(json.dumps(payload))
            entries = load_predictions_entries(p)
            self.assertEqual(entries[0]["image"], "a.jpg")

    def test_load_payload_preserves_meta(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = {"predictions": [{"image": "a.jpg", "detections": []}], "meta": {"timestamp": "x"}}
            p.write_text(json.dumps(payload))
            entries, meta = load_predictions_payload(p)
            self.assertEqual(entries[0]["image"], "a.jpg")
            self.assertEqual(meta, {"timestamp": "x"})

    def test_validate_wrapped_meta_contract_ok(self):
        payload = {
            "predictions": [{"image": "a.jpg", "detections": []}],
            "meta": {
                "timestamp": "2026-02-05T00:00:00Z",
                "adapter": "dummy",
                "config": "rtdetr_pose/configs/base.json",
                "checkpoint": None,
                "images": 1,
                "tta": {
                    "enabled": False,
                    "seed": None,
                    "flip_prob": 0.5,
                    "norm_only": False,
                    "warnings": [],
                    "summary": None,
                },
                "ttt": {
                    "enabled": False,
                    "method": "tent",
                    "steps": 1,
                    "batch_size": 1,
                    "lr": 1e-4,
                    "update_filter": "all",
                    "include": None,
                    "exclude": None,
                    "max_batches": 1,
                    "seed": None,
                    "mim": {"mask_prob": 0.6, "patch_size": 16, "mask_value": 0.0},
                    "report": None,
                },
            },
        }
        res = validate_predictions_payload(payload, strict=False)
        self.assertIsInstance(res.warnings, list)

    def test_validate_wrapped_meta_contract_rejects_bad_types(self):
        payload = {
            "predictions": [{"image": "a.jpg", "detections": []}],
            "meta": {
                "timestamp": "2026-02-05T00:00:00Z",
                "adapter": "dummy",
                "config": "rtdetr_pose/configs/base.json",
                "images": "1",
                "tta": {"enabled": False, "seed": None, "flip_prob": 0.5, "norm_only": False, "warnings": [], "summary": None},
                "ttt": {
                    "enabled": False,
                    "method": "tent",
                    "steps": 1,
                    "batch_size": 1,
                    "lr": 1e-4,
                    "update_filter": "all",
                    "include": None,
                    "exclude": None,
                    "max_batches": 1,
                    "seed": None,
                    "mim": {"mask_prob": 0.6, "patch_size": 16, "mask_value": 0.0},
                    "report": None,
                },
            },
        }
        with self.assertRaises(ValueError) as ctx:
            validate_predictions_payload(payload, strict=False)
        self.assertIn("meta.images", str(ctx.exception))

    def test_load_index_adds_basename_alias(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = [{"image": "/tmp/0001.jpg", "detections": [{"score": 0.9, "bbox": {"cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1}}]}]
            p.write_text(json.dumps(payload))
            idx = load_predictions_index(p)
            self.assertIn("/tmp/0001.jpg", idx)
            self.assertIn("0001.jpg", idx)

    def test_load_index_normalizes_windows_slashes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preds.json"
            payload = [{"image": r"C:\tmp\0001.jpg", "detections": [{"score": 0.9, "bbox": {"cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1}}]}]
            p.write_text(json.dumps(payload))
            idx = load_predictions_index(p)
            self.assertIn(r"C:\tmp\0001.jpg", idx)
            self.assertIn("C:/tmp/0001.jpg", idx)
            self.assertIn("0001.jpg", idx)

    def test_validator_warnings_non_strict(self):
        entries = [{"image": "a.jpg", "detections": [{"score": 0.1, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
        res = validate_predictions_entries(entries, strict=False)
        self.assertTrue(any("missing 'class_id'" in w for w in res.warnings))

    def test_validator_strict_requires_numeric(self):
        entries = [{"image": "a.jpg", "detections": [{"class_id": 1, "score": "0.1", "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
        with self.assertRaises(ValueError):
            validate_predictions_entries(entries, strict=True)

    def test_validator_requires_image_string(self):
        entries = [{"image": 123, "detections": []}]
        with self.assertRaises(ValueError) as ctx:
            validate_predictions_entries(entries, strict=False)
        self.assertIn("image must be a non-empty string", str(ctx.exception))

    def test_canonicalize_enforces_finite_and_stable_sort(self):
        entries = [
            {
                "image": "a.jpg",
                "detections": [
                    {"class_id": 2, "score": 0.1, "bbox": {"cx": 0.2, "cy": 0.2, "w": 0.2, "h": 0.2}},
                    {"class_id": 1, "score": 0.9, "bbox": {"cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1}},
                ],
            }
        ]
        out = canonicalize_predictions(entries, strict=True, policy="error")
        dets = out.entries[0]["detections"]
        self.assertEqual(int(dets[0]["class_id"]), 1)
        self.assertEqual(int(dets[1]["class_id"]), 2)

        bad = [
            {
                "image": "a.jpg",
                "detections": [
                    {"class_id": 1, "score": float("nan"), "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}
                ],
            }
        ]
        with self.assertRaises(ValueError):
            canonicalize_predictions(bad, strict=True, policy="error")

    def test_canonicalize_rejects_duplicate_images(self):
        entries = [
            {"image": "a.jpg", "detections": []},
            {"image": "a.jpg", "detections": []},
        ]
        with self.assertRaises(ValueError):
            canonicalize_predictions(entries, strict=False, policy="clamp")

    def test_canonicalize_unknown_keys_warn_or_error(self):
        entries = [
            {
                "image": "a.jpg",
                "extra_entry": 1,
                "detections": [
                    {
                        "class_id": 0,
                        "score": 0.9,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "extra_det": 1,
                    }
                ],
            }
        ]
        warned = canonicalize_predictions(entries, strict=True, policy="error", unknown_keys="warn")
        self.assertTrue(any("unknown keys" in w for w in warned.warnings))
        with self.assertRaises(ValueError):
            canonicalize_predictions(entries, strict=True, policy="error", unknown_keys="error")


if __name__ == "__main__":
    unittest.main()
