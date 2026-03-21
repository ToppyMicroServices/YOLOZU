import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalSegmentationTool(unittest.TestCase):
    def test_eval_segmentation_smoke_with_ignore_index_and_html(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_segmentation.py"
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
            (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (dataset_root / "masks" / "train").mkdir(parents=True, exist_ok=True)

            sample_id = "000001"
            img_path = dataset_root / "images" / "train" / f"{sample_id}.png"
            mask_gt_path = dataset_root / "masks" / "train" / f"{sample_id}.png"

            Image.new("RGB", (2, 2), color=(0, 0, 0)).save(img_path)

            gt = np.array([[0, 0], [1, 255]], dtype=np.uint8)
            Image.fromarray(gt, mode="L").save(mask_gt_path)

            preds_root = root / "preds"
            preds_root.mkdir(parents=True, exist_ok=True)
            mask_pred_path = preds_root / f"{sample_id}.png"
            pred = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            Image.fromarray(pred, mode="L").save(mask_pred_path)

            dataset_json = dataset_root / "dataset.json"
            dataset_json.write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "task": "semantic_segmentation",
                        "split": "train",
                        "mode": "copy",
                        "path_type": "relative",
                        "ignore_index": 255,
                        "classes": ["background", "obj"],
                        "samples": [
                            {
                                "id": sample_id,
                                "image": f"images/train/{sample_id}.png",
                                "mask": f"masks/train/{sample_id}.png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            pred_json = preds_root / "seg_predictions.json"
            pred_json.write_text(json.dumps({sample_id: str(mask_pred_path.name)}), encoding="utf-8")

            out_json = root / "seg_eval.json"
            out_html = root / "seg_eval.html"
            overlays_dir = root / "overlays"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-json",
                    str(dataset_json.relative_to(repo_root)),
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
                self.fail(f"eval_segmentation.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_json.is_file())
            report = json.loads(out_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            self.assertAlmostEqual(float(metrics.get("miou")), 0.5, places=6)
            self.assertAlmostEqual(float(metrics.get("pixel_accuracy")), 2.0 / 3.0, places=6)

            self.assertTrue(out_html.is_file())
            html = out_html.read_text(encoding="utf-8")
            self.assertIn("<html>", html)

            overlay_path = overlays_dir / f"{sample_id}.png"
            self.assertTrue(overlay_path.is_file())


if __name__ == "__main__":
    unittest.main()
