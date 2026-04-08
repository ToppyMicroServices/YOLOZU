import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class TestEvalDepthTool(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "tools" / "eval_depth.py"

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--pred-depth", proc.stdout)
        self.assertIn("--gt-depth", proc.stdout)
        self.assertIn("--align", proc.stdout)

    def test_cli_writes_report_with_median_scale_alignment(self) -> None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable for depth CLI test: {exc}")
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            td_path = Path(td)
            gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            pred = gt * 2.0
            mask = np.array([[1, 1], [0, 1]], dtype=np.uint8)
            gt_path = td_path / "gt.npy"
            pred_path = td_path / "pred.npy"
            mask_path = td_path / "mask.npy"
            out_path = td_path / "depth_eval.json"
            np.save(gt_path, gt)
            np.save(pred_path, pred)
            np.save(mask_path, mask)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script),
                    "--pred-depth",
                    str(pred_path),
                    "--gt-depth",
                    str(gt_path),
                    "--mask",
                    str(mask_path),
                    "--align",
                    "median_scale",
                    "--output",
                    str(out_path),
                ],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "yolozu_depth_eval_report")
            self.assertEqual(payload.get("alignment"), "median_scale")
            self.assertAlmostEqual(float(payload.get("scale_factor")), 0.5, places=6)
            counts = payload.get("counts") or {}
            self.assertEqual(counts.get("valid_pixels"), 3)
            metrics = payload.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("mae")), 0.0, places=6)
            self.assertAlmostEqual(float(metrics.get("rmse")), 0.0, places=6)
            self.assertAlmostEqual(float(metrics.get("abs_rel")), 0.0, places=6)
            self.assertAlmostEqual(float(metrics.get("delta1")), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
