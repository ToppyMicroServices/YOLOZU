import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestExportPredictionsCocoKeypointsTool(unittest.TestCase):
    def test_convert_coco_keypoints_results_to_predictions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "export_predictions_coco_keypoints.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            instances = root / "instances.json"
            results = root / "results.json"
            out = root / "predictions.json"

            instances.write_text(
                json.dumps(
                    {
                        "images": [{"id": 7, "file_name": "img001.jpg", "width": 100, "height": 50}],
                        "annotations": [],
                        "categories": [{"id": 1, "name": "person"}],
                    }
                ),
                encoding="utf-8",
            )
            results.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 7,
                            "category_id": 1,
                            "bbox": [10, 5, 40, 20],
                            "score": 0.9,
                            "keypoints": [20, 10, 2, 40, 20, 2],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, str(script), "--results-json", str(results), "--instances-json", str(instances), "--output", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"export_predictions_coco_keypoints.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["image"], "img001.jpg")
            det = payload[0]["detections"][0]
            self.assertEqual(det["class_id"], 0)
            self.assertAlmostEqual(det["bbox"]["cx"], 0.3)
            self.assertEqual(len(det["keypoints"]), 2)
            self.assertAlmostEqual(det["keypoints"][0]["x"], 0.2)


if __name__ == "__main__":
    unittest.main()
