import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsDetectron2MMDetTools(unittest.TestCase):
    def test_detectron2_dry_run_exports_strict_predictions(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_detectron2.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_detectron2.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_detectron2.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--config",
                    "dummy_detectron2.yaml",
                    "--weights",
                    "dummy_model_final.pth",
                    "--dry-run",
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
                self.fail(f"export_predictions_detectron2.py failed:\n{proc.stdout}\n{proc.stderr}")

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
            self.assertNotIn("WARN:", proc_validate.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema_version"), 1)
            self.assertTrue(payload["predictions"])
            self.assertTrue(
                all(entry.get("schema_version") == 2 for entry in payload["predictions"])
            )
            extra = payload["meta"]["extra"]
            self.assertEqual(extra.get("execution_status"), "dry_run")
            self.assertFalse(bool(extra.get("runtime_executed")))
            self.assertEqual(extra.get("inference_calls"), 0)

    def test_mmdet_dry_run_exports_strict_predictions(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_mmdet.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_mmdet.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_mmdet.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--config",
                    "dummy_mmdet.py",
                    "--checkpoint",
                    "dummy_epoch_12.pth",
                    "--dry-run",
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
                self.fail(f"export_predictions_mmdet.py failed:\n{proc.stdout}\n{proc.stderr}")

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
            self.assertNotIn("WARN:", proc_validate.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema_version"), 1)
            self.assertTrue(payload["predictions"])
            self.assertTrue(
                all(entry.get("schema_version") == 2 for entry in payload["predictions"])
            )
            extra = payload["meta"]["extra"]
            self.assertEqual(extra.get("execution_status"), "dry_run")
            self.assertFalse(bool(extra.get("runtime_executed")))
            self.assertEqual(extra.get("inference_calls"), 0)

    def test_detectron2_non_dry_missing_runtime_inputs_fails_without_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_detectron2.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "pred_detectron2.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(repo_root / "data" / "smoke"),
                    "--split",
                    "val",
                    "--config",
                    str(root / "missing.yaml"),
                    "--weights",
                    str(root / "missing.pth"),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("file not found", proc.stderr)
            self.assertFalse(out.exists())

    def test_mmdet_non_dry_missing_runtime_inputs_fails_without_output(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_mmdet.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "pred_mmdet.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(repo_root / "data" / "smoke"),
                    "--split",
                    "val",
                    "--config",
                    str(root / "missing.py"),
                    "--checkpoint",
                    str(root / "missing.pth"),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("file not found", proc.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
