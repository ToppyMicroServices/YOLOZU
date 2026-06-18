import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtdetr_pose.train_records import load_train_records, resolve_val_records


class TestTrainRecords(unittest.TestCase):
    def test_val_records_json_overrides_val_split_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            val_path = root / "val_records.json"
            val_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"image_path": str(root / "a.jpg"), "labels": []},
                            {"image_path": str(root / "b.jpg"), "labels": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(val_records_json=str(val_path), val_split="val2017", val_max_images=1)

            def _build_manifest(*_args, **_kwargs):
                raise AssertionError("val_records_json should bypass val-split manifest scan")

            val_records, val_records_map = resolve_val_records(
                args=args,
                dataset_root=root / "dataset",
                workspace_root=root,
                build_manifest_fn=_build_manifest,
                flatten_records_for_map_fn=lambda records: [{"image_path": r["image_path"]} for r in records],
            )

        self.assertEqual(len(val_records), 1)
        self.assertEqual(len(val_records_map), 1)
        self.assertTrue(val_records[0]["image_path"].endswith("a.jpg"))

    def test_train_records_json_and_extra_records_share_loader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_path = root / "train_records.json"
            extra_path = root / "extra_records.json"
            train_path.write_text(json.dumps([{"image_path": "train.jpg", "labels": []}]), encoding="utf-8")
            extra_path.write_text(json.dumps({"images": [{"image_path": "extra.jpg", "labels": []}]}), encoding="utf-8")
            args = SimpleNamespace(records_json=str(train_path), extra_records_json=str(extra_path))

            records, keypoint_names, keypoint_skeleton = load_train_records(
                args=args,
                dataset_root=root / "dataset",
                workspace_root=root,
                build_manifest_fn=lambda *_args, **_kwargs: {"images": []},
                extract_manifest_keypoints_meta_fn=lambda _manifest: (["unused"], [[1, 2]]),
            )

        self.assertEqual([r["image_path"] for r in records], ["train.jpg", "extra.jpg"])
        self.assertEqual(keypoint_names, [])
        self.assertEqual(keypoint_skeleton, [])


if __name__ == "__main__":
    unittest.main()
