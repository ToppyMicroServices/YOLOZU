import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import os
import json


class TestDemoCliDefaults(unittest.TestCase):
    def test_demo_keypoints_help_works(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo", "keypoints", "--help"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("--score-threshold", proc.stdout)

    def test_demo_depth_help_works(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo", "depth", "--help"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("--model", proc.stdout)
            self.assertIn("depth_anything", proc.stdout)
            self.assertIn("--compare", proc.stdout)

    def test_demo_pose_help_works(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo", "pose", "--help"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("--pattern-cols", proc.stdout)
            self.assertIn("--square-size", proc.stdout)
            self.assertIn("--backend", proc.stdout)
            self.assertIn("--aruco-dict", proc.stdout)
            self.assertIn("--densefusion-root", proc.stdout)

    def test_demo_train_help_works(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo", "train", "--help"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("--max-steps", proc.stdout)

    def test_demo_defaults_to_demo_suite(self):
        try:
            import numpy as _np  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest("numpy not installed") from exc
        try:
            from PIL import Image as _Image  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest("Pillow not installed") from exc

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
            # Default should run the demo suite (at least the instance-seg coco-instances section).
            self.assertIn("instance-seg demo:", proc.stdout)
            self.assertIn("== instance-seg (coco-instances) ==", proc.stdout)
            self.assertIn("output_dir:", proc.stdout)
            self.assertIn("suite_config:", proc.stdout)
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

    def test_demo_instance_seg_coco_instances_uses_default_paths_when_omitted(self):
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

            # Create default COCO paths under cwd.
            images_dir = td_path / "data" / "coco" / "images" / "val2017"
            ann_dir = td_path / "data" / "coco" / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            img_path = images_dir / "000000000001.jpg"
            Image.new("RGB", (64, 48), (240, 240, 240)).save(img_path)

            instances_path = ann_dir / "instances_val2017.json"
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
                    "instance-seg",
                    "--background",
                    "coco-instances",
                    "--inference",
                    "none",
                ],
                cwd=str(td_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            self.assertIn("instance-seg demo:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
