from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPrepareCocoKeypointsYOLOZUTool(unittest.TestCase):
    def test_skips_invalid_rows_and_writes_valid_labels(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tool = repo_root / "tools" / "prepare_coco_keypoints_yolozu.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            work = Path(td)
            coco_root = work / "coco"
            images_dir = coco_root / "val2017"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "0001.jpg").write_bytes(b"")

            annotations = {
                "images": [
                    {"id": "bad", "file_name": "bad.jpg", "width": 100, "height": 100},
                    {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"image_id": "bad", "category_id": 1, "bbox": [0, 0, 20, 20], "keypoints": [10, 10, 2]},
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "keypoints": ["bad", 10, 2]},
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "keypoints": [10, 10, 2]},
                ],
                "categories": [
                    {"id": "bad", "name": "junk", "keypoints": ["nose"]},
                    {"id": 1, "name": "person", "keypoints": ["nose"], "skeleton": [["x", 2], [1, 1], [1, 2]]},
                ],
            }
            ann_path = coco_root / "annotations.json"
            ann_path.write_text(json.dumps(annotations), encoding="utf-8")

            out_root = work / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--coco-root",
                    str(coco_root),
                    "--annotations",
                    str(ann_path.relative_to(coco_root)),
                    "--images-dir",
                    "val2017",
                    "--out",
                    str(out_root),
                    "--category-id",
                    "1",
                ],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"prepare_coco_keypoints_yolozu failed:\n{proc.stdout}\n{proc.stderr}")

            dataset = json.loads((out_root / "dataset.json").read_text(encoding="utf-8"))
            classes = json.loads((out_root / "labels" / "val2017" / "classes.json").read_text(encoding="utf-8"))
            label_lines = (out_root / "labels" / "val2017" / "0001.txt").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(dataset["task"], "keypoints")
            self.assertEqual(dataset["category_id"], 1)
            self.assertEqual(classes["keypoint_names"], ["nose"])
            self.assertEqual(classes.get("skeleton", []), [])
            self.assertEqual(len(label_lines), 1)
            self.assertTrue(label_lines[0].startswith("0 "))


if __name__ == "__main__":
    unittest.main()
