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

    def test_executorch_exporter_requires_declared_runtime_output_non_dry(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_executorch.py"
        dataset = repo_root / "data" / "smoke"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            model = root / "model.pte"
            model.write_bytes(b"pte")
            out = root / "pred_executorch.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--model",
                    str(model),
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
            self.assertIn("--runtime-output-json is required", proc.stderr)
            self.assertFalse(out.exists())

    def test_executorch_exporter_decodes_runtime_output_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_executorch.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            model = root / "model.pte"
            model.write_bytes(b"pte")
            runtime_output = root / "runtime_outputs.json"
            runtime_output.write_text(
                json.dumps(
                    {
                        "000000000009.jpg": [[0.1, 0.2, 0.5, 0.7, 0.9, 3]],
                        "000000000025.jpg": [[0.0, 0.0, 1.0, 1.0, 0.01, 1]],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "pred_executorch.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--max-images",
                    "1",
                    "--model",
                    str(model),
                    "--runtime-output-json",
                    str(runtime_output),
                    "--min-score",
                    "0.1",
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
                self.fail(f"export_predictions_executorch.py decode failed:\n{proc.stdout}\n{proc.stderr}")

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
            entries = payload.get("predictions") or []
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["detections"][0]["class_id"], 3)
            self.assertAlmostEqual(entries[0]["detections"][0]["bbox"]["cx"], 0.3)
            extra = payload.get("meta", {}).get("extra", {})
            self.assertEqual(extra.get("runtime_decode", {}).get("contract"), "combined_xyxy_score_class")

    def test_executorch_exporter_fails_unsupported_decoder_shape(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_executorch.py"
        dataset = repo_root / "data" / "smoke"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            model = root / "model.pte"
            model.write_bytes(b"pte")
            runtime_output = root / "runtime_outputs.json"
            runtime_output.write_text(json.dumps({"000000000009.jpg": [[0.1, 0.2, 0.5]]}), encoding="utf-8")
            out = root / "pred_executorch.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--max-images",
                    "1",
                    "--model",
                    str(model),
                    "--runtime-output-json",
                    str(runtime_output),
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
            self.assertIn("ExecuTorch decoder error", proc.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
