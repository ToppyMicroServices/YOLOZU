import json
from pathlib import Path
import tempfile
import unittest

from rtdetr_pose.dataset import build_manifest


class TestRTDETRDatasetDescriptors(unittest.TestCase):
    def test_dataset_json_wrapper_with_external_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            images_dir = source / "images" / "val2017"
            labels_dir = root / "wrapper" / "labels" / "val2017"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "0001.jpg").write_bytes(b"")
            (labels_dir / "0001.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            wrapper = root / "wrapper"
            (wrapper / "dataset.json").write_text(
                json.dumps(
                    {
                        "format": "yolozu_dataset_wrapper",
                        "images_dir": str(images_dir),
                        "labels_dir": str(labels_dir),
                        "split": "val2017",
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_manifest(wrapper, split="val2017")
            manifest_from_file = build_manifest(wrapper / "dataset.json", split="val2017")

        self.assertEqual(len(manifest["images"]), 1)
        record = manifest["images"][0]
        self.assertEqual(record["image_path"], str(images_dir / "0001.jpg"))
        self.assertEqual(record["labels"][0]["bbox"]["w"], 0.25)
        self.assertEqual(manifest_from_file["images"][0]["image_path"], str(images_dir / "0001.jpg"))

    def test_data_yaml_resolves_yolo_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = root / "dataset"
            images_dir = dataset / "images" / "train"
            labels_dir = dataset / "labels" / "train"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (images_dir / "sample.png").write_bytes(b"")
            (labels_dir / "sample.txt").write_text("1 0.4 0.6 0.2 0.3\n", encoding="utf-8")
            (dataset / "data.yaml").write_text(
                "path: .\ntrain: images/train\nnames:\n  0: background\n  1: object\n",
                encoding="utf-8",
            )

            manifest = build_manifest(dataset / "data.yaml", split="train")

        self.assertEqual(len(manifest["images"]), 1)
        label = manifest["images"][0]["labels"][0]
        self.assertEqual(label["class_id"], 1)
        self.assertEqual(label["bbox"]["h"], 0.3)


if __name__ == "__main__":
    unittest.main()
