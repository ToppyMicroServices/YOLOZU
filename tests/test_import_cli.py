import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestImportCLI(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_import_dataset_coco_instances_wrapper_validates(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "coco" / "images" / "val2017"
            ann_dir = root / "coco" / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            img_path = images_dir / "0001.png"
            # Minimal PNG header with valid signature + IHDR width/height.
            img_path.write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (64).to_bytes(4, "big") + (32).to_bytes(4, "big")
            )

            instances = {
                "images": [{"id": 1, "file_name": "0001.png", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            out_dir = root / "wrapper"
            proc = self._run(
                [
                    "import",
                    "dataset",
                    "--from",
                    "coco-instances",
                    "--instances",
                    str(instances_path),
                    "--images-dir",
                    str(images_dir),
                    "--split",
                    "val2017",
                    "--output",
                    str(out_dir),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"import dataset coco-instances failed:\n{proc.stdout}\n{proc.stderr}")

            proc2 = self._run(
                [
                    "validate",
                    "dataset",
                    str(out_dir),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--strict",
                ],
                cwd=repo_root,
            )
            if proc2.returncode != 0:
                self.fail(f"validate dataset on COCO wrapper failed:\n{proc2.stdout}\n{proc2.stderr}")

    def test_import_config_ultralytics_args_yaml(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text(
                "\n".join(
                    [
                        "imgsz: 640",
                        "batch: 16",
                        "epochs: 100",
                        "lr0: 0.01",
                        "weight_decay: 0.0005",
                        "optimizer: SGD",
                        "seed: 0",
                        "device: cpu",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            out_path = root / "train_config_import.json"
            proc = self._run(
                [
                    "import",
                    "config",
                    "--from",
                    "ultralytics",
                    "--args",
                    str(args_yaml),
                    "--output",
                    str(out_path),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"import config ultralytics failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_train_config_v1")
            self.assertEqual(int(payload.get("batch")), 16)
            self.assertEqual(int(payload.get("epochs")), 100)
            self.assertAlmostEqual(float(payload.get("lr")), 0.01, places=6)

    def test_import_dataset_coco_instances_keeps_keypoints_schema(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "coco" / "images" / "val2017"
            ann_dir = root / "coco" / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            (images_dir / "0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [
                    {
                        "id": 7,
                        "name": "person",
                        "keypoints": ["nose", "left_eye", "right_eye"],
                        "skeleton": [[1, 2], [1, 3]],
                    }
                ],
            }
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            out_dir = root / "wrapper"
            proc = self._run(
                [
                    "import",
                    "dataset",
                    "--from",
                    "coco-instances",
                    "--instances",
                    str(instances_path),
                    "--images-dir",
                    str(images_dir),
                    "--split",
                    "val2017",
                    "--output",
                    str(out_dir),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"import dataset coco-instances failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_json.get("keypoint_names"), ["nose", "left_eye", "right_eye"])
            self.assertEqual(dataset_json.get("skeleton"), [[1, 2], [1, 3]])

            classes_json = json.loads((out_dir / "labels" / "val2017" / "classes.json").read_text(encoding="utf-8"))
            self.assertEqual(classes_json.get("keypoint_names"), ["nose", "left_eye", "right_eye"])
            self.assertEqual(classes_json.get("skeleton"), [[1, 2], [1, 3]])

    def test_import_dataset_auto_from_coco_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        sample_root = repo_root / "data" / "conversion_tiny_coco"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out_dir = Path(td) / "wrapper_auto"

            proc = self._run(
                [
                    "import",
                    "dataset",
                    "--from",
                    "auto",
                    "--dataset",
                    str(sample_root),
                    "--split",
                    "val2017",
                    "--output",
                    str(out_dir),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"import dataset auto from coco root failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_json.get("format"), "coco_instances")
            self.assertEqual(dataset_json.get("split"), "val2017")


if __name__ == "__main__":
    unittest.main()
