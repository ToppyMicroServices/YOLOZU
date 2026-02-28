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
