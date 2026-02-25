import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import os
import json


class TestDemoCliDefaults(unittest.TestCase):
    def test_demo_defaults_to_demo_suite(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            # Default should run the demo suite (at least instance-seg synthetic).
            self.assertIn("instance-seg demo:", proc.stdout)
            self.assertIn("== instance-seg (synthetic) ==", proc.stdout)
            self.assertIn("== instance-seg (coco-instances) ==", proc.stdout)
            self.assertIn("output_dir:", proc.stdout)
            # Default output folder should be demo_output.
            self.assertIn("demo_output", proc.stdout)

    def test_demo_suite_can_enable_coco_instances_via_top_level_args(self):
        try:
            import numpy as _np  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest("numpy not installed") from exc
        try:
            from PIL import Image
        except Exception as exc:
            raise unittest.SkipTest("Pillow not installed") from exc

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            td_path = Path(td)
            images_dir = td_path / "coco_images"
            images_dir.mkdir(parents=True, exist_ok=True)

            img_path = images_dir / "000000000001.jpg"
            Image.new("RGB", (64, 48), (240, 240, 240)).save(img_path)

            instances_path = td_path / "instances_val.json"
            coco = {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "iscrowd": 0,
                        "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                    }
                ],
                "categories": [{"id": 3, "name": "thing"}],
            }
            instances_path.write_text(json.dumps(coco), encoding="utf-8")

            cwd = td_path / "run"
            cwd.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "demo",
                    "--coco-instances-json",
                    str(instances_path),
                    "--coco-images-dir",
                    str(images_dir),
                ],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("== instance-seg (coco-instances) ==", proc.stdout)


if __name__ == "__main__":
    unittest.main()
