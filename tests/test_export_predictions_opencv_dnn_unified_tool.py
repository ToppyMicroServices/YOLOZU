import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsOpenCVDnnUnifiedTool(unittest.TestCase):
    def test_unified_dry_run_writes_predictions_and_dump_io(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_opencv_dnn_unified.py"
        dataset = repo_root / "data" / "smoke"

        self.assertTrue(script.is_file(), "missing tools/export_predictions_opencv_dnn_unified.py")
        self.assertTrue(dataset.is_dir(), "missing data/smoke dataset")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_opencv_unified.json"
            io_dump = Path(td) / "pred_opencv_unified.io.json"
            meta = Path(td) / "pred_opencv_unified.meta.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--onnx",
                    "dummy.onnx",
                    "--decode",
                    "yolo_84",
                    "--preprocess",
                    "yolo_letterbox_640",
                    "--dry-run",
                    "--dump-io",
                    str(io_dump),
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
                self.fail(f"export_predictions_opencv_dnn_unified.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out.is_file())
            self.assertTrue(meta.is_file())
            self.assertTrue(io_dump.is_file())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)
            self.assertIn("predictions", payload)
            self.assertIsInstance(payload.get("predictions"), list)

            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_obj.get("unified", {}).get("decode"), "yolo_84")
            self.assertEqual(meta_obj.get("unified", {}).get("preprocess_preset"), "yolo_letterbox_640")

    def test_yolozu_export_backend_opencv_dnn_dry_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "yolozu.py"
        dataset = repo_root / "data" / "smoke"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "pred_from_cli.json"
            io_dump = Path(td) / "pred_from_cli.io.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "export",
                    "--backend",
                    "opencv-dnn",
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--onnx",
                    "dummy.onnx",
                    "--decode",
                    "rtdetr",
                    "--preprocess",
                    "rtdetr_resize_640",
                    "--dry-run",
                    "--dump-io",
                    str(io_dump),
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
                self.fail(f"tools/yolozu.py export failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out.is_file())
            self.assertTrue(io_dump.is_file())

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)
            self.assertIn("predictions", payload)
            self.assertIsInstance(payload.get("predictions"), list)


if __name__ == "__main__":
    unittest.main()
