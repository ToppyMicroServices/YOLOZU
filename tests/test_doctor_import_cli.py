import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDoctorImportCLI(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_doctor_import_dataset_coco_instances_stdout(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--dataset-from",
                    "coco-instances",
                    "--instances",
                    str(instances_path),
                    "--images-dir",
                    str(images_dir),
                    "--split",
                    "val2017",
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (dataset) failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertEqual(payload.get("kind"), "yolozu_doctor_import")
            ds = payload.get("dataset") or {}
            self.assertEqual(ds.get("from"), "coco-instances")
            counts = ds.get("counts") or {}
            self.assertEqual(int(counts.get("images")), 1)

    def test_doctor_import_dataset_auto_warns_category_id_zero(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            images_dir = root / "images" / "val2017"
            ann_dir = root / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 64, "height": 32}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [{"id": 0, "name": "zero_cls"}],
            }
            instances_path = ann_dir / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--dataset-from",
                    "auto",
                    "--instances",
                    str(instances_path),
                    "--images-dir",
                    str(images_dir),
                    "--split",
                    "val2017",
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (dataset auto) failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(proc.stdout)
            ds = payload.get("dataset") or {}
            self.assertEqual(ds.get("from"), "coco-instances")
            self.assertTrue(bool(ds.get("category_id_zero_present")))
            warnings = payload.get("warnings") or []
            self.assertTrue(any("category_id=0" in str(w) for w in warnings))

    def test_doctor_import_config_ultralytics_stdout(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text("imgsz: 640\nbatch: 16\nepochs: 1\nlr0: 0.01\n", encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--config-from",
                    "ultralytics",
                    "--args",
                    str(args_yaml),
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (config) failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(proc.stdout)
            cfg = payload.get("config") or {}
            train = cfg.get("train_config") or {}
            self.assertEqual(train.get("format"), "yolozu_train_config_v1")
            self.assertEqual(int(train.get("batch")), 16)

    def test_doctor_import_config_auto_ultralytics_stdout(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text("imgsz: 640\nbatch: 12\nepochs: 1\nlr0: 0.01\n", encoding="utf-8")

            proc = self._run(
                [
                    "doctor",
                    "import",
                    "--config-from",
                    "auto",
                    "--args",
                    str(args_yaml),
                    "--output",
                    "-",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"doctor import (config auto) failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(proc.stdout)
            cfg = payload.get("config") or {}
            self.assertEqual(cfg.get("from"), "ultralytics")
            warnings = payload.get("warnings") or []
            self.assertTrue(any("auto-detected" in str(w) for w in warnings))

    def test_train_import_ultralytics_preview_writes_resolved_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text("imgsz: 640\nbatch: 8\nepochs: 1\nlr0: 0.005\n", encoding="utf-8")
            out_path = root / "resolved_train_config.json"

            proc = self._run(
                [
                    "train",
                    "--import",
                    "ultralytics",
                    "--cfg",
                    str(args_yaml),
                    "--resolved-config-out",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --import preview failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_train_config_v1")
            self.assertEqual(int(payload.get("batch")), 8)

    def test_train_import_auto_preview_writes_resolved_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            args_yaml = root / "args.yaml"
            args_yaml.write_text("imgsz: 640\nbatch: 6\nepochs: 1\nlr0: 0.002\n", encoding="utf-8")
            out_path = root / "resolved_train_config_auto.json"

            proc = self._run(
                [
                    "train",
                    "--import",
                    "auto",
                    "--cfg",
                    str(args_yaml),
                    "--resolved-config-out",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --import auto preview failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_train_config_v1")
            self.assertEqual(int(payload.get("batch")), 6)

    def test_train_external_yolox_dry_run_writes_bridge_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "train_yolox_bridge.json"

            proc = self._run(
                [
                    "train",
                    "--external-backend",
                    "yolox",
                    "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --external-backend yolox --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_training_run_summary_v1")
            self.assertEqual(str(payload.get("task")), "train_yolox")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(((payload.get("backend") or {}).get("backend_id")), "yolox")
            self.assertEqual(str((payload.get("license_boundary") or {}).get("primary_lane")), "YOLOX-style external training bridge")

    def test_train_external_ultralytics_dry_run_writes_bridge_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "train_ultralytics_bridge.json"

            proc = self._run(
                [
                    "train",
                    "--external-backend",
                    "ultralytics",
                    "yolo11n.pt",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --external-backend ultralytics --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_training_run_summary_v1")
            self.assertEqual(str(payload.get("task")), "train_ultralytics")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(((payload.get("backend") or {}).get("backend_id")), "ultralytics")
            self.assertTrue(bool((payload.get("license_boundary") or {}).get("optional_bridge")))

    def test_train_external_detectron2_dry_run_writes_bridge_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "train_detectron2_bridge.json"

            proc = self._run(
                [
                    "train",
                    "--external-backend",
                    "detectron2",
                    "configs/examples/finetune_external/detectron2_finetune_smoke.yaml",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--task-family",
                    "keypoints",
                    "--train-opt",
                    "DATASETS.TRAIN",
                    "(\"dummy_train\",)",
                    "--output",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --external-backend detectron2 --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_training_run_summary_v1")
            self.assertEqual(str(payload.get("task")), "train_detectron2")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(((payload.get("backend") or {}).get("backend_id")), "detectron2")
            self.assertEqual(str(payload.get("task_family")), "keypoints")

    def test_train_external_hf_detr_dry_run_writes_bridge_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out_path = root / "train_hf_detr_bridge.json"

            proc = self._run(
                [
                    "train",
                    "--external-backend",
                    "hf-detr",
                    "facebook/detr-resnet-50",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(out_path),
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"train --external-backend hf-detr --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_training_run_summary_v1")
            self.assertEqual(str(payload.get("task")), "train_hf_detr")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertEqual(((payload.get("backend") or {}).get("backend_id")), "hf-detr")
            self.assertEqual(str(payload.get("model_id")), "facebook/detr-resnet-50")


if __name__ == "__main__":
    unittest.main()
