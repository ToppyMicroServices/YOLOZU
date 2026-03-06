import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestInstanceSegYoloBboxBackground(unittest.TestCase):
    def test_demo_instance_seg_yolo_bbox_background_runs(self):
        repo_root = Path(__file__).resolve().parents[1]
        try:
            import numpy as _np  # noqa: F401
            from PIL import Image as _Image  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        dataset_root = repo_root / "data" / "smoke"
        if not dataset_root.is_dir():
            self.skipTest("data/smoke missing")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            run_dir = Path(td) / "run"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "demo",
                    "instance-seg",
                    "--background",
                    "yolo-bbox",
                    "--yolo-root",
                    str(dataset_root),
                    "--yolo-split",
                    "val",
                    "--inference",
                    "none",
                    "--num-images",
                    "1",
                    "--max-instances",
                    "1",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"demo instance-seg yolo-bbox failed:\n{proc.stdout}\n{proc.stderr}")

            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            self.assertTrue(lines, "demo instance-seg produced no stdout")
            report_path = Path(lines[-1])
            self.assertTrue(report_path.is_file(), f"demo report missing: {report_path}")
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            meta = payload.get("meta") or {}
            self.assertEqual(meta.get("background"), "yolo-bbox")
            dataset = meta.get("dataset") or {}
            self.assertIn("yolo_root", dataset)
            self.assertIn("yolo_split", dataset)
            artifacts = payload.get("artifacts") or {}
            overlays_dir = artifacts.get("overlays_dir")
            self.assertTrue(isinstance(overlays_dir, str) and overlays_dir, "missing overlays_dir in demo artifacts")
            overlays = sorted(Path(overlays_dir).glob("*.png"))
            self.assertTrue(overlays, "demo should generate at least one overlay PNG")


if __name__ == "__main__":
    unittest.main()
