import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPackageSegmentationPredictionsTool(unittest.TestCase):
    def test_package_mask_dir_into_predictions_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "package_segmentation_predictions.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_json = root / "dataset.json"
            masks_dir = root / "pred_masks"
            out = root / "seg_predictions.json"
            masks_dir.mkdir(parents=True, exist_ok=True)

            dataset_json.write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "task": "semantic_segmentation",
                        "split": "val",
                        "mode": "manifest",
                        "path_type": "relative",
                        "ignore_index": 255,
                        "classes": ["bg", "obj"],
                        "samples": [
                            {"id": "sample_a", "image": "images/sample_a.jpg", "mask": "masks/sample_a.png"},
                            {"id": "sample_b", "image": "images/sample_b.jpg", "mask": "masks/sample_b.png"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            for sample_id in ("sample_a", "sample_b"):
                (masks_dir / f"{sample_id}.png").write_bytes(b"not-read-by-this-tool")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-json",
                    str(dataset_json),
                    "--masks-dir",
                    str(masks_dir),
                    "--output",
                    str(out),
                    "--relative-to-output",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"package_segmentation_predictions.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in payload], ["sample_a", "sample_b"])
            self.assertTrue(payload[0]["mask"].endswith("pred_masks/sample_a.png"))


if __name__ == "__main__":
    unittest.main()
