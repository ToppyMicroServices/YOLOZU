import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsYoloXTool(unittest.TestCase):
    def test_yolox_export_dry_run_is_schema_valid(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_yolox.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_yolox.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_yolox.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
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
                self.fail(f"export_predictions_yolox.py failed:\n{proc.stdout}\n{proc.stderr}")
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
            self.assertEqual(payload.get("meta", {}).get("adapter"), "yolox")
            extra = payload.get("meta", {}).get("extra", {})
            self.assertEqual(extra.get("execution_status"), "dry_run")
            self.assertFalse(bool(extra.get("runtime_executed")))
            self.assertEqual(extra.get("inference_calls"), 0)

    def test_yolox_non_dry_requires_exp_and_weights(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_yolox.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_yolox.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(repo_root / "data" / "smoke"),
                    "--split",
                    "val",
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
            self.assertIn("requires --exp", proc.stderr)
            self.assertIn("requires --weights", proc.stderr)
            self.assertFalse(out.exists())

    def test_yolozu_export_backend_yolox_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        dataset = repo_root / "data" / "smoke"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_from_yolozu_cli.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "yolox",
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--dry-run",
                    "--output",
                    str(out),
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"tools/yolozu.py export --backend yolox failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("predictions", payload)
            self.assertIsInstance(payload.get("predictions"), list)


if __name__ == "__main__":
    unittest.main()
