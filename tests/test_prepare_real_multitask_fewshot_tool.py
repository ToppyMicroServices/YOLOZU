import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPrepareRealMultitaskFewshotTool(unittest.TestCase):
    def test_prepare_tool_smoke(self) -> None:
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            self.skipTest("NumPy/Pillow not installed")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_real_multitask_fewshot.py"
        instances = repo_root / "data" / "coco" / "annotations" / "instances_val2017.json"
        images_dir = repo_root / "data" / "coco" / "images" / "val2017"

        if not instances.exists() or not images_dir.exists():
            self.skipTest("COCO fixture data missing")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fewshot"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--instances-json",
                    str(instances),
                    "--images-dir",
                    str(images_dir),
                    "--out",
                    str(out),
                    "--train-images",
                    "1",
                    "--val-images",
                    "1",
                    "--force",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"prepare_real_multitask_fewshot failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")

            summary = out / "prepare_summary.json"
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            counts = payload.get("counts") or {}
            self.assertEqual(int(counts.get("train_images", -1)), 1)
            self.assertEqual(int(counts.get("val_images", -1)), 1)
            provenance = payload.get("label_provenance") or {}
            self.assertEqual(str(provenance.get("bbox")), "coco_instances_gt")
            self.assertFalse(bool(provenance.get("model_inference_used", True)))
            self.assertTrue((out / "labels" / "train" / "classes.json").exists())
            self.assertTrue((out / "labels" / "val" / "classes.json").exists())

    def test_prepare_requires_manual_download_by_default(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_real_multitask_fewshot.py"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "fewshot"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--instances-json",
                    str(root / "missing" / "instances_val2017.json"),
                    "--images-dir",
                    str(root / "missing" / "val2017"),
                    "--out",
                    str(out),
                    "--train-images",
                    "1",
                    "--val-images",
                    "1",
                    "--force",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("automatic download is disabled by default", proc.stderr)
            self.assertIn("download_coco_instances_tiny.py", proc.stderr)

    def test_prepare_auto_download_requires_license_ack(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "prepare_real_multitask_fewshot.py"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "fewshot"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--instances-json",
                    str(root / "missing" / "instances_val2017.json"),
                    "--images-dir",
                    str(root / "missing" / "val2017"),
                    "--out",
                    str(out),
                    "--train-images",
                    "1",
                    "--val-images",
                    "1",
                    "--download-if-missing",
                    "--allow-auto-download",
                    "--force",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--accept-dataset-license", proc.stderr)


if __name__ == "__main__":
    unittest.main()
