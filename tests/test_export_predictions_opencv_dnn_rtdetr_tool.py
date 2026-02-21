import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsOpenCVDNNRTDETRTool(unittest.TestCase):
    def test_dry_run_writes_predictions_and_meta(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_opencv_dnn_rtdetr.py"
        self.assertTrue(script.is_file(), "missing tools/export_predictions_opencv_dnn_rtdetr.py")

        dataset = repo_root / "data" / "smoke"
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "pred_rtdetr_opencv_dnn.json"
            meta = root / "pred_rtdetr_opencv_dnn.json.meta.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--max-images",
                    "2",
                    "--strict",
                    "--dry-run",
                    "--output",
                    str(out),
                    "--meta-output",
                    str(meta),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions_opencv_dnn_rtdetr.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out.is_file(), "missing predictions output")
            self.assertTrue(meta.is_file(), "missing meta output")

            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_obj.get("exporter"), "opencv_dnn_rtdetr")
            self.assertTrue(bool(meta_obj.get("dry_run", False)))
            export_settings = meta_obj.get("export_settings") or {}
            self.assertIn("preprocess", export_settings)

            payload = json.loads(out.read_text(encoding="utf-8"))
            entries = payload.get("predictions") or []
            self.assertTrue(isinstance(entries, list) and entries, "expected predictions entries")
            entry0 = entries[0]
            self.assertIn("preprocess", entry0)

            proc2 = subprocess.run(
                [sys.executable, str(repo_root / "tools" / "validate_predictions.py"), str(out), "--strict"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc2.returncode != 0:
                self.fail(f"validate_predictions.py failed:\n{proc2.stdout}\n{proc2.stderr}")


if __name__ == "__main__":
    unittest.main()

