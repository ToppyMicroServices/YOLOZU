import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.coco_eval import build_coco_ground_truth, predictions_to_coco_detections
from yolozu.dataset import build_manifest


def _write_stub_png(path: Path, *, width: int, height: int) -> None:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = (13).to_bytes(4, "big", signed=False)
    ihdr = b"IHDR"
    w = int(width).to_bytes(4, "big", signed=False)
    h = int(height).to_bytes(4, "big", signed=False)
    rest = bytes([8, 2, 0, 0, 0])
    path.write_bytes(sig + ihdr_len + ihdr + w + h + rest)


class TestCocoEvalConversion(unittest.TestCase):
    def test_build_ground_truth_schema(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "coco128"
        if not dataset_root.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        manifest = build_manifest(dataset_root)
        records = manifest["images"][:3]

        gt, index = build_coco_ground_truth(records)
        self.assertIn("images", gt)
        self.assertIn("annotations", gt)
        self.assertIn("categories", gt)
        self.assertEqual(len(gt["images"]), len(records))

        self.assertTrue(index.image_key_to_id)
        self.assertTrue(index.class_id_to_category_id)

    def test_predictions_to_detections_mapping(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "coco128"
        if not dataset_root.is_dir():
            self.skipTest("coco128 missing; run: bash tools/fetch_coco128.sh")
        manifest = build_manifest(dataset_root)
        records = manifest["images"][:2]

        gt, index = build_coco_ground_truth(records)
        image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

        # One dummy detection per image, in normalized cxcywh.
        preds = []
        for rec in records:
            preds.append(
                {
                    "image": Path(rec["image"]).name,
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            )

        dt = predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format="cxcywh_norm")
        self.assertEqual(len(dt), len(records))
        for det in dt:
            self.assertIn("image_id", det)
            self.assertIn("category_id", det)
            self.assertIn("bbox", det)
            self.assertIn("score", det)

    def test_predictions_to_detections_windows_style_image_key(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            img_path = root / "0001.png"
            _write_stub_png(img_path, width=64, height=32)

            records = [
                {
                    "image": str(img_path),
                    "labels": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}],
                }
            ]
            gt, index = build_coco_ground_truth(records)
            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}

            preds = [
                {
                    "image": str(img_path).replace("/", "\\"),
                    "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                }
            ]
            dt = predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format="cxcywh_norm")
            self.assertEqual(len(dt), 1)
            self.assertEqual(int(dt[0]["image_id"]), 1)

    def test_predictions_to_detections_rejects_non_list_detections(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            img_path = root / "0001.png"
            _write_stub_png(img_path, width=16, height=16)

            records = [{"image": str(img_path), "labels": []}]
            gt, index = build_coco_ground_truth(records)
            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt["images"]}
            preds = [{"image": str(img_path), "detections": {"class_id": 0}}]

            with self.assertRaises(ValueError) as ctx:
                predictions_to_coco_detections(preds, coco_index=index, image_sizes=image_sizes, bbox_format="cxcywh_norm")
            self.assertIn("detections must be a list", str(ctx.exception))

    def test_ambiguous_basename_cannot_select_a_different_image(self):
        records = [
            {"image": "/dataset/a/sample.jpg", "labels": []},
            {"image": "/dataset/b/sample.jpg", "labels": []},
        ]
        with patch("yolozu.eval.coco_eval.get_image_size", return_value=(64, 32)):
            _, index = build_coco_ground_truth(records)
        sizes = {1: (64, 32), 2: (64, 32)}
        prediction = {"image": "/relocated/b/sample.jpg", "detections": [
            {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}
        ]}
        with self.assertRaisesRegex(ValueError, "ambiguous prediction image key"):
            predictions_to_coco_detections([prediction], coco_index=index, image_sizes=sizes)
        prediction["image"] = records[1]["image"]
        detections = predictions_to_coco_detections([prediction], coco_index=index, image_sizes=sizes)
        self.assertEqual(detections[0]["image_id"], 2)


if __name__ == "__main__":
    unittest.main()
