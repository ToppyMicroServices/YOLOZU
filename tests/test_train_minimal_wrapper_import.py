import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


class TestTrainMinimalWrapperImport(unittest.TestCase):
    def test_train_minimal_wrapper_importable(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import rtdetr_pose.tools.train_minimal as tm; assert callable(getattr(tm, 'main', None))",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"import failed:\n{proc.stdout}\n{proc.stderr}")

    @unittest.skipIf(torch is None, "torch not installed")
    @unittest.skipIf(Image is None, "Pillow not installed")
    def test_cpu_smoke_train_from_records_json(self):
        repo_root = Path(__file__).resolve().parents[1]
        from rtdetr_pose.tools import train_minimal

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            image_path = root / "image.png"
            Image.new("RGB", (32, 32), color=(120, 80, 40)).save(image_path)
            records_path = root / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image_path": str(image_path),
                                "labels": [{"class_id": 0, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            metrics_path = root / "metrics.jsonl"
            result = train_minimal.main(
                [
                    "--records-json",
                    str(records_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--max-steps",
                    "1",
                    "--image-size",
                    "32",
                    "--device",
                    "cpu",
                    "--real-images",
                    "--strict-task-data",
                    "--num-classes",
                    "1",
                    "--num-queries",
                    "2",
                    "--hidden-dim",
                    "16",
                    "--run-dir",
                    str(run_dir),
                    "--metrics-jsonl",
                    str(metrics_path),
                    "--no-export-onnx",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(metrics_path.is_file())
            self.assertTrue((run_dir / "run_record.json").is_file())


if __name__ == "__main__":
    unittest.main()
