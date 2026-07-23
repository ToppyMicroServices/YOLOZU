import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestYoloMigrationTools(unittest.TestCase):
    def test_import_ultralytics_data_yaml_writes_classes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "import_ultralytics_data_yaml.py"
        self.assertTrue(script.is_file(), "missing tools/import_ultralytics_data_yaml.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset_src"
            for split in ("train", "val"):
                (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
                (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
                (dataset_root / "images" / split / "000001.jpg").write_bytes(b"")
                (dataset_root / "labels" / split / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {dataset_root}",
                        "train: images/train",
                        "val: images/val",
                        "names:",
                        "  0: person",
                        "  1: bicycle",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out_root = root / "wrapped"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--data-yaml",
                    str(data_yaml),
                    "--split",
                    "val",
                    "--output",
                    str(out_root),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"import_ultralytics_data_yaml.py failed:\n{proc.stdout}\n{proc.stderr}")

            dataset_json = out_root / "dataset.json"
            classes_json = out_root / "labels" / "val" / "classes.json"
            classes_txt = out_root / "labels" / "val" / "classes.txt"
            self.assertTrue(dataset_json.is_file())
            self.assertTrue(classes_json.is_file())
            self.assertTrue(classes_txt.is_file())

            classes = json.loads(classes_json.read_text(encoding="utf-8"))
            self.assertEqual(classes.get("class_names"), ["person", "bicycle"])
            self.assertEqual(classes.get("class_id_to_category_id"), {"0": 0, "1": 1})

    def test_export_predictions_yolov5_generates_strict_wrapped_payload(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_yolov5.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_yolov5.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_yolov5.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--labels-dir",
                    str(dataset / "labels" / "val"),
                    "--protocol",
                    "nms_applied",
                    "--output",
                    str(out),
                    "--strict",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions_yolov5.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out.is_file())

            proc_validate = subprocess.run(
                [sys.executable, str(validator), str(out), "--strict"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_validate.returncode != 0:
                self.fail(f"validate_predictions.py failed:\n{proc_validate.stdout}\n{proc_validate.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("predictions", payload)
            self.assertIn("meta", payload)
            self.assertEqual(payload["meta"].get("adapter"), "yolov5")
            self.assertEqual(payload["meta"].get("ttt", {}).get("enabled"), False)

    def test_export_predictions_ultralytics_dry_run_generates_strict_payload(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_ultralytics.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_ultralytics.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_ultralytics.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--model",
                    "yolo11n.pt",
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--max-images",
                    "1",
                    "--protocol",
                    "nms_applied",
                    "--dry-run",
                    "--wrap",
                    "--strict",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions_ultralytics.py failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out.is_file())

            proc_validate = subprocess.run(
                [sys.executable, str(validator), str(out), "--strict"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc_validate.returncode != 0:
                self.fail(f"validate_predictions.py failed:\n{proc_validate.stdout}\n{proc_validate.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("predictions", payload)
            self.assertIn("meta", payload)
            self.assertEqual(payload["meta"].get("adapter"), "ultralytics")
            extra = payload.get("meta", {}).get("extra", {}) or {}
            self.assertTrue(bool(extra.get("dry_run")))
            self.assertEqual(extra.get("execution_status"), "dry_run")
            self.assertFalse(bool(extra.get("runtime_executed")))
            self.assertEqual(extra.get("inference_calls"), 0)


if __name__ == "__main__":
    unittest.main()
