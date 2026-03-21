import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsTTTCLI(unittest.TestCase):
    def test_help_includes_ttt_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions.py"
        self.assertTrue(script.is_file())

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        out = proc.stdout + proc.stderr
        self.assertIn("--ttt", out)
        self.assertIn("--ttt-preset", out)
        self.assertIn("--ttt-method", out)
        self.assertIn("--ttt-reset", out)
        self.assertIn("--ttt-steps", out)
        self.assertIn("--ttt-max-update-norm", out)
        self.assertIn("--ttt-cotta-ema-momentum", out)
        self.assertIn("--ttt-cotta-augmentations", out)
        self.assertIn("--ttt-cotta-restore-prob", out)
        self.assertIn("--ttt-eata-conf-min", out)
        self.assertIn("--ttt-eata-anchor-lambda", out)
        self.assertIn("--ttt-eata-max-skip-streak", out)
        self.assertIn("--ttt-sar-rho", out)
        self.assertIn("--ttt-sar-adaptive", out)
        self.assertIn("--ttt-sar-first-step-scale", out)
        self.assertIn("--ttt-sdft-task", out)
        self.assertIn("--ttt-aux-pose-weight", out)
        self.assertIn("--ttt-aux-keypoints-weight", out)
        self.assertIn("--ttt-aux-depth-weight", out)
        self.assertIn("--ttt-aux-seg-weight", out)
        self.assertIn("--ttt-aux-temperature", out)
        self.assertIn("pose_safe", out)
        self.assertIn("keypoints_safe", out)

    def test_dummy_adapter_smoke_and_ttt_unsupported(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            (tmp_root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (tmp_root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            Image = None
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover
                self.skipTest(f"PIL not available: {exc}")

            img_path = tmp_root / "images" / "train2017" / "000000000001.jpg"
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_path)

            out_path = tmp_root / "preds.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--adapter",
                    "dummy",
                    "--dataset",
                    str(tmp_root),
                    "--output",
                    str(out_path.relative_to(repo_root)),
                    "--wrap",
                    "--max-images",
                    "1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions dummy failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text())
            self.assertIn("predictions", payload)
            self.assertIn("meta", payload)
            self.assertFalse(payload["meta"]["ttt"]["enabled"])

            proc2 = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--adapter",
                    "dummy",
                    "--dataset",
                    str(tmp_root),
                    "--output",
                    str(out_path.relative_to(repo_root)),
                    "--wrap",
                    "--max-images",
                    "1",
                    "--ttt",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertNotEqual(proc2.returncode, 0)
            msg = proc2.stdout + proc2.stderr
            self.assertIn("TTT failed", msg)


if __name__ == "__main__":
    unittest.main()
