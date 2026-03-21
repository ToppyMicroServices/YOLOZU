import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.adapter import RTDETRPoseAdapter  # noqa: E402
from yolozu.predictions import validate_predictions_entries, validate_predictions_payload  # noqa: E402


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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


@unittest.skipUnless(_has_cuda(), "CUDA not available")
class TestCudaSmokeRTDETRPose(unittest.TestCase):
    def test_adapter_predict_on_cuda(self):
        torch = None
        Image = None
        try:
            import torch
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_path = root / "tiny_rtdetr_pose.json"
            _write_tiny_rtdetr_pose_config(cfg_path)

            img_path = root / "img.jpg"
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_path)

            adapter = RTDETRPoseAdapter(
                config_path=str(cfg_path),
                checkpoint_path=None,
                device="cuda",
                image_size=(32, 32),
                score_threshold=1.1,  # force empty detections for stability
                max_detections=5,
            )
            model = adapter.get_model()
            param = next(model.parameters())
            self.assertTrue(param.is_cuda)

            outputs = adapter.predict([{"image": str(img_path)}])
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]["image"], str(img_path))
            self.assertIsInstance(outputs[0].get("detections"), list)

            res = validate_predictions_entries(outputs, strict=False)
            self.assertIsInstance(res.warnings, list)

            torch.cuda.synchronize()

    def test_export_predictions_cli_rtdetr_pose_cuda_smoke(self):
        Image = None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

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
            script = repo_root / "tools" / "export_predictions.py"

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
                    "cuda",
                    "--image-size",
                    "32",
                    "--score-threshold",
                    "1.1",  # force empty detections for stability
                    "--max-detections",
                    "5",
                    "--max-images",
                    "1",
                    "--wrap",
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
            res = validate_predictions_payload(payload, strict=False)
            self.assertIsInstance(res.warnings, list)

    def test_export_predictions_cli_rtdetr_pose_cuda_smoke_with_lora(self):
        Image = None
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

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
            script = repo_root / "tools" / "export_predictions.py"

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
                    "cuda",
                    "--image-size",
                    "32",
                    "--score-threshold",
                    "1.1",  # force empty detections for stability
                    "--max-detections",
                    "5",
                    "--max-images",
                    "1",
                    "--wrap",
                    "--lora-r",
                    "2",
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
            report = meta.get("lora", {}).get("report")
            self.assertIsInstance(report, dict)
            self.assertTrue(report.get("enabled"))
            self.assertGreater(int(report.get("replaced", 0)), 0)


if __name__ == "__main__":
    unittest.main()
