from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeArray:
    def __init__(self, data):
        self._data = data
        self.ndim = 2

    def astype(self, _dtype: str):
        return self

    def __mul__(self, _other):
        return self


class _FakeNumpy:
    @staticmethod
    def asarray(data):
        if not isinstance(data, list):
            raise TypeError("expected list-like mask")
        return _FakeArray(data)


class _FakeImageWriter:
    def save(self, path):
        Path(path).write_bytes(b"mask")


class _FakeImageModule:
    @staticmethod
    def fromarray(_arr, mode="L"):
        if mode != "L":
            raise ValueError("unexpected mode")
        return _FakeImageWriter()


class TestCocoInstanceSegConversionTools(unittest.TestCase):
    def test_convert_predictions_main_skips_invalid_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "convert_coco_instance_seg_predictions_test",
            repo_root / "tools" / "convert_coco_instance_seg_predictions.py",
        )

        class FakeMaskUtils:
            @staticmethod
            def frPyObjects(seg, h, w):
                if not isinstance(seg, list) or h <= 0 or w <= 0:
                    raise ValueError("bad polygon")
                return seg

            @staticmethod
            def merge(rles):
                return rles

            @staticmethod
            def decode(rle):
                if rle == {"bad": True}:
                    raise ValueError("bad rle")
                return [[1, 0], [0, 1]]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds_path = root / "preds.json"
            inst_path = root / "instances.json"
            out_path = root / "predictions.json"
            masks_dir = root / "masks"

            preds_path.write_text(
                json.dumps(
                    [
                        {"image_id": "bad", "category_id": 1, "score": 0.8, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                        {"image_id": 1, "category_id": "bad", "score": 0.8, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                        {"image_id": 1, "category_id": 1, "score": 0.8, "segmentation": {"bad": True}},
                        {"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": [[0, 0, 1, 0, 1, 1]]},
                    ]
                ),
                encoding="utf-8",
            )
            inst_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": "bad", "file_name": "ignore.jpg", "width": 10, "height": 10},
                            {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
                        ],
                        "categories": [{"id": "bad", "name": "ignore"}, {"id": 1, "name": "thing"}],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(module, "_try_import_deps", return_value=(_FakeNumpy, _FakeImageModule, FakeMaskUtils)):
                rc = module.main(
                    [
                        "--predictions",
                        str(preds_path),
                        "--instances-json",
                        str(inst_path),
                        "--output",
                        str(out_path),
                        "--masks-dir",
                        str(masks_dir),
                        "--min-score",
                        "0.1",
                    ]
                )

            self.assertEqual(rc, None)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["predictions"]), 1)
            self.assertEqual(len(payload["predictions"][0]["instances"]), 1)
            self.assertTrue((masks_dir / payload["predictions"][0]["instances"][0]["mask"].split("/")[-1]).exists())

    def test_prepare_instance_seg_main_skips_invalid_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "prepare_coco_instance_seg_test",
            repo_root / "tools" / "prepare_coco_instance_seg.py",
        )

        class FakeCOCO:
            def __init__(self, _path: str):
                pass

            def getAnnIds(self, imgIds=None):
                return [1, 2, 3]

            def loadAnns(self, ann_ids):
                return [
                    {"id": 3, "category_id": "bad", "segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
                    {"id": 2, "category_id": 1, "segmentation": "bad", "iscrowd": 0},
                    {"id": 1, "category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
                ]

            def annToMask(self, ann):
                if ann.get("segmentation") == "bad":
                    raise ValueError("bad segmentation")
                return [[1, 0], [0, 1]]

        def fake_convert_coco_instances_to_yolo_labels(*, instances_json, images_dir, labels_dir, include_crowd):
            self.assertIsInstance(instances_json, dict)
            self.assertTrue(Path(images_dir).exists())
            self.assertIn(include_crowd, (True, False))
            Path(labels_dir).mkdir(parents=True, exist_ok=True)
            (Path(labels_dir) / "classes.json").write_text(
                json.dumps({"category_id_to_class_id": {"bad": 9, "1": 0}}),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "0001.jpg").write_bytes(b"")
            ann_dir = root / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": "bad", "file_name": "ignore.jpg"},
                            {"id": 1, "file_name": "0001.jpg"},
                        ],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "thing"}],
                    }
                ),
                encoding="utf-8",
            )
            out_root = root / "out"

            with (
                mock.patch.object(module, "convert_coco_instances_to_yolo_labels", side_effect=fake_convert_coco_instances_to_yolo_labels),
                mock.patch.object(module, "_try_import_deps", return_value=(_FakeNumpy, _FakeImageModule, FakeCOCO)),
            ):
                rc = module.main(
                    [
                        "--coco-root",
                        str(root),
                        "--split",
                        "val2017",
                        "--out",
                        str(out_root),
                    ]
                )

            self.assertEqual(rc, None)
            label_json = json.loads((out_root / "labels" / "val2017" / "0001.json").read_text(encoding="utf-8"))
            self.assertEqual(label_json["mask_classes"], [0])
            self.assertEqual(len(label_json["mask_path"]), 1)
            self.assertTrue((out_root / label_json["mask_path"][0]).exists())


if __name__ == "__main__":
    unittest.main()
