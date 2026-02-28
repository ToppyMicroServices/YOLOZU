import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsExecuTorchTool(unittest.TestCase):
    def test_executorch_exporter_dry_run_schema_valid(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_executorch.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_executorch.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_executorch.json"
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
                    "--wrap",
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
                self.fail(f"export_predictions_executorch.py failed:\n{proc.stdout}\n{proc.stderr}")

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
            meta = payload.get("meta", {})
            self.assertEqual(meta.get("adapter"), "executorch")
            self.assertEqual((meta.get("extra") or {}).get("exporter"), "executorch")
            self.assertTrue((meta.get("extra") or {}).get("dry_run"))


if __name__ == "__main__":
    unittest.main()
