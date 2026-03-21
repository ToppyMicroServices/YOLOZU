import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalInstanceSegmentationTool(unittest.TestCase):
    def test_eval_instance_segmentation_smoke_with_html_and_overlays(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_instance_segmentation.py"
        self.assertTrue(script.is_file())

        np = None
        Image = None
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"deps not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "dataset"
            split = "val2017"
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "masks" / split).mkdir(parents=True, exist_ok=True)

            sample_id = "000001"
            img_path = dataset_root / "images" / split / f"{sample_id}.png"
            Image.new("RGB", (4, 4), color=(0, 0, 0)).save(img_path)

            # Two GT instances: class 0 (top-left 2x2) and class 1 (bottom-right 2x2).
            gt0 = np.zeros((4, 4), dtype=np.uint8)
            gt0[:2, :2] = 255
            gt1 = np.zeros((4, 4), dtype=np.uint8)
            gt1[2:, 2:] = 255
            gt0_path = dataset_root / "masks" / split / f"{sample_id}_c0.png"
            gt1_path = dataset_root / "masks" / split / f"{sample_id}_c1.png"
            Image.fromarray(gt0, mode="L").save(gt0_path)
            Image.fromarray(gt1, mode="L").save(gt1_path)

            # Sidecar metadata under labels/<split>/<stem>.json
            sidecar = dataset_root / "labels" / split / f"{sample_id}.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "mask_path": [
                            f"masks/{split}/{gt0_path.name}",
                            f"masks/{split}/{gt1_path.name}",
                        ],
                        "mask_classes": [0, 1],
                    }
                ),
                encoding="utf-8",
            )

            preds_root = root / "preds"
            preds_root.mkdir(parents=True, exist_ok=True)
            pred0_path = preds_root / f"{sample_id}_pred_c0.png"
            pred1_path = preds_root / f"{sample_id}_pred_c1.png"
            Image.fromarray(gt0, mode="L").save(pred0_path)
            Image.fromarray(gt1, mode="L").save(pred1_path)

            pred_json = preds_root / "instance_seg_predictions.json"
            pred_json.write_text(
                json.dumps(
                    [
                        {
                            "image": img_path.name,
                            "instances": [
                                {"class_id": 0, "score": 0.9, "mask": pred0_path.name},
                                {"class_id": 1, "score": 0.8, "mask": pred1_path.name},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            out_json = root / "instance_seg_eval.json"
            out_html = root / "instance_seg_eval.html"
            overlays_dir = root / "overlays"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_root.relative_to(repo_root)),
                    "--split",
                    split,
                    "--predictions",
                    str(pred_json.relative_to(repo_root)),
                    "--pred-root",
                    str(preds_root.relative_to(repo_root)),
                    "--output",
                    str(out_json.relative_to(repo_root)),
                    "--html",
                    str(out_html.relative_to(repo_root)),
                    "--overlays-dir",
                    str(overlays_dir.relative_to(repo_root)),
                    "--max-overlays",
                    "1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"eval_instance_segmentation.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_json.is_file())
            report = json.loads(out_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("map50")), 1.0, places=6)
            self.assertAlmostEqual(float(metrics.get("map50_95")), 1.0, places=6)

            self.assertTrue(out_html.is_file())
            html = out_html.read_text(encoding="utf-8")
            self.assertIn("<html>", html)

            self.assertTrue(overlays_dir.is_dir())
            self.assertGreaterEqual(len(list(overlays_dir.glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
