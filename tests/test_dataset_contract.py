import json
from pathlib import Path
import tempfile
import unittest

from yolozu.datasets.dataset import build_manifest
from yolozu.datasets.dataset_contract import normalize_label_bbox


class TestDatasetContractV1(unittest.TestCase):
    def test_xyxy_abs_derives_yolo_and_coco_views(self):
        label = normalize_label_bbox(
            {"class_id": 3, "bbox": {"format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50, "y2": 80}},
            image_wh=(200, 100),
            bbox_field="preserve",
        )

        self.assertEqual(label["bbox_format"], "xyxy_abs")
        self.assertEqual(label["bbox_xywh_abs"]["x"], 10.0)
        self.assertEqual(label["bbox_xywh_abs"]["w"], 40.0)
        self.assertAlmostEqual(label["bbox_cxcywh_norm"]["cx"], 0.15)
        self.assertAlmostEqual(label["bbox_cxcywh_norm"]["cy"], 0.5)
        self.assertAlmostEqual(label["bbox_cxcywh_norm"]["w"], 0.2)
        self.assertAlmostEqual(label["bbox_cxcywh_norm"]["h"], 0.6)

    def test_coco_manifest_keeps_xyxy_abs_as_dataset_contract_view(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            (images_dir / "0001.jpg").write_bytes(b"")
            coco = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 200, "height": 100}],
                "categories": [{"id": 7, "name": "thing"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 40, 60]}],
            }
            (ann_dir / "instances_val2017.json").write_text(json.dumps(coco), encoding="utf-8")

            manifest = build_manifest(root, split="val2017")

        label = manifest["images"][0]["labels"][0]
        self.assertEqual(manifest["images"][0]["dataset_contract_version"], "1")
        self.assertEqual(label["bbox"]["format"], "xyxy_abs")
        self.assertEqual(label["bbox_xyxy_abs"]["x1"], 10.0)
        self.assertEqual(label["bbox_xyxy_abs"]["x2"], 50.0)
        self.assertEqual(label["bbox_xywh_abs"]["w"], 40.0)
        self.assertAlmostEqual(label["bbox_cxcywh_norm"]["h"], 0.6)


if __name__ == "__main__":
    unittest.main()
