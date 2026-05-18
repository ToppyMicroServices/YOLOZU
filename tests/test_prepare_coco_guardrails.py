import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPrepareCocoGuardrails(unittest.TestCase):
    def _make_empty_instances_json(self, root: Path, split: str) -> Path:
        ann = root / "annotations"
        ann.mkdir(parents=True, exist_ok=True)
        path = ann / f"instances_{split}.json"
        path.write_text(json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8")
        return path

    def _make_instances_json(self, root: Path, split: str, *, file_name: str) -> Path:
        ann = root / "annotations"
        ann.mkdir(parents=True, exist_ok=True)
        path = ann / f"instances_{split}.json"
        path.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": file_name, "width": 100, "height": 100}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 2, 10, 20], "iscrowd": 0}
                    ],
                    "categories": [{"id": 7, "name": "thing"}],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_prepare_coco_yolo_rejects_empty_instances_images(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_coco_yolo.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "val2017").mkdir(parents=True, exist_ok=True)
            self._make_empty_instances_json(root, "val2017")
            out = root / "out"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--coco-root",
                    str(root),
                    "--split",
                    "val2017",
                    "--out",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_COCO_IMAGES_EMPTY]", proc.stderr)
            self.assertIn("instances JSON has no images entries", proc.stderr)

    def test_prepare_coco_yolo_rejects_traversal_file_name(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_coco_yolo.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "val2017").mkdir(parents=True, exist_ok=True)
            (root / "secret.jpg").write_text("not an image", encoding="utf-8")
            self._make_instances_json(root, "val2017", file_name="../../secret.jpg")
            out = root / "out"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--coco-root",
                    str(root),
                    "--split",
                    "val2017",
                    "--out",
                    str(out),
                    "--copy-images",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unsafe COCO file_name", proc.stderr)
            self.assertFalse((out / "images" / "val2017" / "secret.jpg").exists())

    def test_prepare_coco_instance_seg_rejects_empty_instances_images(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_coco_instance_seg.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "val2017").mkdir(parents=True, exist_ok=True)
            self._make_empty_instances_json(root, "val2017")
            out = root / "out"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--coco-root",
                    str(root),
                    "--split",
                    "val2017",
                    "--out",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_COCO_IMAGES_EMPTY]", proc.stderr)
            self.assertIn("instances JSON has no images entries", proc.stderr)


if __name__ == "__main__":
    unittest.main()
