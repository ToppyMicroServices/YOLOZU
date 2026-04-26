import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsTorchScriptTool(unittest.TestCase):
    def test_torchscript_exporter_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_torchscript.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"export_predictions_torchscript.py --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--combined-output", proc.stdout)
        self.assertIn("--boxes-scale", proc.stdout)

    def test_torchscript_exporter_real_combined_output_or_skip(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"torch unavailable: {exc}")
        try:
            import PIL  # noqa: F401
            import numpy  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"image preprocessing deps unavailable: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_torchscript.py"
        validator = repo_root / "tools" / "validate_predictions.py"
        dataset = repo_root / "data" / "smoke"

        class FixedDetector(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[[0.1, 0.2, 0.5, 0.7, 0.9, 2.0]]], dtype=x.dtype, device=x.device)

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            model_path = root / "fixed.torchscript"
            traced = torch.jit.trace(FixedDetector().eval(), torch.zeros((1, 3, 640, 640), dtype=torch.float32))
            traced.save(str(model_path))
            out = root / "pred_torchscript.json"
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
                    str(model_path),
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
                self.fail(f"export_predictions_torchscript.py failed:\n{proc.stdout}\n{proc.stderr}")

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
            self.assertEqual(payload.get("meta", {}).get("adapter"), "torchscript")
            entries = payload.get("predictions") or []
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["detections"][0]["class_id"], 2)


if __name__ == "__main__":
    unittest.main()
