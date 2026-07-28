import json
import math
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


@unittest.skipIf(torch is None, "torch not installed")
class TestRefinePredictionsHessianCLI(unittest.TestCase):
    def test_cli_smoke_refine_offsets(self):
        script = repo_root / "tools" / "refine_predictions_hessian.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds_path = root / "preds.json"
            out_path = root / "preds_refined.json"
            log_path = root / "refine_log.json"

            preds = [
                {
                    "image": "000001.jpg",
                    # Inline aux maps (H=W=5) so the tool can exercise real refinement without a dataset.
                    "depth": [[0, 1, 2, 3, 4]] * 5,
                    "mask": [[1, 1, 1, 1, 1]] * 5,
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                            # Target a depth value to force offsets refinement along +x (depth(u)=u).
                            "log_z": math.log(4.0),
                            "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            "offsets": [0.0, 0.0],
                        }
                    ],
                }
            ]
            preds_path.write_text(json.dumps({"predictions": preds}, indent=2))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--predictions",
                    str(preds_path),
                    "--output",
                    str(out_path),
                    "--enable",
                    "--refine-offsets",
                    "--wrap",
                    "--steps",
                    "3",
                    "--log-output",
                    str(log_path),
                    "--log-steps",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"refine_predictions_hessian.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("predictions", payload)
            det0 = payload["predictions"][0]["detections"][0]
            self.assertIn("hessian_refinement", det0)
            self.assertIn("offsets", det0.get("hessian_refinement", {}))

            # Should nudge offsets toward +x (depth(u)=u) in a deterministic way.
            offsets_after = det0.get("offsets")
            self.assertIsInstance(offsets_after, list)
            self.assertEqual(len(offsets_after), 2)
            self.assertGreater(float(offsets_after[0]), 0.5)

            report = det0["hessian_refinement"]["offsets"]
            self.assertGreaterEqual(int(report.get("steps_run", 0)), 1)
            self.assertIsInstance(str(report.get("stop_reason", "")), str)

            log_payload = json.loads(log_path.read_text())
            self.assertIn("images", log_payload)
            report = log_payload.get("research_report") or {}
            self.assertEqual(report.get("kind"), "research_lane_report")
            self.assertEqual(report.get("lane"), "hessian")
            self.assertEqual((report.get("promotion_gate") or {}).get("decision"), "review_required")
            self.assertEqual(len((report.get("artifact_hashes") or {}).get("output_sha256", "")), 64)
            self.assertGreaterEqual(float((report.get("latency_overhead") or {}).get("total", -1.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
