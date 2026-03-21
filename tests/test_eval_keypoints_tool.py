import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalKeypointsTool(unittest.TestCase):
    def test_eval_keypoints_pck_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_keypoints.py"
        self.assertTrue(script.is_file())

        Image = None
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Pillow not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            img_path = root / "images" / "train2017" / "000000000001.jpg"
            Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

            # GT: 2 labeled keypoints (v>0).
            (root / "labels" / "train2017" / "000000000001.txt").write_text(
                "0 0.5 0.5 0.4 0.4 0.50 0.50 2 0.60 0.50 2\n",
                encoding="utf-8",
            )

            pred_path = root / "predictions.json"
            pred_path.write_text(
                json.dumps(
                    [
                        {
                            "image": str(img_path),
                            "detections": [
                                {
                                    "class_id": 0,
                                    "score": 1.0,
                                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                                    # kp0 close, kp1 far -> PCK 0.5 at threshold 0.1 (scale=max(w,h)=0.4).
                                    "keypoints": [0.52, 0.50, 2, 0.70, 0.50, 2],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            out_path = root / "keypoints_eval.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(root),
                    "--split",
                    "train2017",
                    "--predictions",
                    str(pred_path),
                    "--output",
                    str(out_path),
                    "--pck-threshold",
                    "0.1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"eval_keypoints.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            self.assertAlmostEqual(float(metrics.get("pck")), 0.5, places=6)
            self.assertEqual(int(metrics.get("keypoints_labeled")), 2)
            self.assertEqual(int(metrics.get("keypoints_correct")), 1)


if __name__ == "__main__":
    unittest.main()
