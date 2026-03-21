import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
repo_root = Path(__file__).resolve().parents[1]


def _write_tiny_rtdetr_pose_config(path: Path) -> None:
    cfg = {
        "dataset": {"root": ".", "split": "val2017", "format": "yolo"},
        "model": {
            "num_classes": 3,
            "hidden_dim": 64,
            "num_queries": 10,
            "use_uncertainty": False,
            "backbone_name": "tiny_cnn",
            "stem_channels": 8,
            "backbone_channels": [16, 32, 64],
            "stage_blocks": [1, 1, 1],
            "num_encoder_layers": 0,
            "num_decoder_layers": 1,
            "nhead": 8,
        },
        "loss": {"name": "default", "task_aligner": "none", "weights": {}},
        "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
    }
    path.write_text(json.dumps(cfg))


class TestExportPredictionsLoRACLI(unittest.TestCase):
    def test_help_includes_lora_flags(self):
        script = repo_root / "tools" / "export_predictions.py"
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
        for flag in (
            "--lora-r",
            "--lora-alpha",
            "--lora-dropout",
            "--lora-target",
            "--lora-freeze-base",
            "--lora-train-bias",
            "--infer-batch-size",
            "--torch-compile",
            "--torch-compile-backend",
            "--torch-compile-mode",
            "--torch-amp",
            "--torch-channels-last",
            "--torch-inference-mode",
            "--tta-flip-keypoints",
            "--tta-flip-pose-offsets",
        ):
            self.assertIn(flag, out)

    def test_dummy_adapter_rejects_lora_flags(self):
        script = repo_root / "tools" / "export_predictions.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--adapter", "dummy", "--lora-r", "2", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        # argparse exits with code 0 for --help, so this isn't a great place to assert errors.
        # Ensure the flags exist; behavior is covered by the runtime check below.
        self.assertEqual(proc.returncode, 0)

        proc2 = subprocess.run(
            [sys.executable, str(script), "--adapter", "dummy", "--lora-r", "2"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertNotEqual(proc2.returncode, 0)
        msg = proc2.stdout + proc2.stderr
        self.assertIn("--lora-* flags are only supported", msg)

    def test_dummy_adapter_rejects_compile_and_batch_flags(self):
        script = repo_root / "tools" / "export_predictions.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--adapter",
                "dummy",
                "--torch-compile",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        msg = proc.stdout + proc.stderr
        self.assertIn("--torch-compile*/--torch-amp/--torch-channels-last", msg)

    def test_rtdetr_pose_export_smoke_with_lora(self):
        if torch is None:
            self.skipTest("torch not installed")
        Image = None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"PIL not available: {exc}")

        script = repo_root / "tools" / "export_predictions.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            images = dataset_root / "images" / "val2017"
            labels = dataset_root / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            img_path = images / "000001.jpg"
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_path)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            cfg_path = root / "tiny_rtdetr_pose.json"
            _write_tiny_rtdetr_pose_config(cfg_path)

            out_path = root / "preds.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--adapter",
                    "rtdetr_pose",
                    "--dataset",
                    str(dataset_root),
                    "--config",
                    str(cfg_path),
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--score-threshold",
                    "1.1",  # force empty detections for stability
                    "--max-detections",
                    "5",
                    "--infer-batch-size",
                    "1",
                    "--torch-compile",
                    "--torch-compile-backend",
                    "inductor",
                    "--torch-compile-mode",
                    "reduce-overhead",
                    "--torch-amp",
                    "bf16",
                    "--no-torch-channels-last",
                    "--torch-inference-mode",
                    "--max-images",
                    "1",
                    "--wrap",
                    "--lora-r",
                    "2",
                    "--lora-freeze-base",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            meta = payload.get("meta") or {}
            self.assertTrue(meta.get("lora", {}).get("enabled"))
            self.assertTrue(meta.get("lora", {}).get("freeze_base"))
            self.assertEqual(int((meta.get("inference") or {}).get("infer_batch_size", 0)), 1)
            self.assertTrue(((meta.get("inference") or {}).get("torch_compile") or {}).get("enabled"))
            self.assertEqual(str((meta.get("inference") or {}).get("torch_amp")), "bf16")
            self.assertFalse(bool((meta.get("inference") or {}).get("torch_channels_last")))
            self.assertTrue(bool((meta.get("inference") or {}).get("torch_inference_mode")))

            report = meta.get("lora", {}).get("report")
            self.assertIsInstance(report, dict)
            self.assertTrue(report.get("enabled"))
            self.assertGreater(int(report.get("replaced", 0)), 0)

            trainable_info = report.get("trainable_info")
            self.assertIsInstance(trainable_info, dict)
            self.assertGreater(int(trainable_info.get("lora_params", 0)), 0)


if __name__ == "__main__":
    unittest.main()
