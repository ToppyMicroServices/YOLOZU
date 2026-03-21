import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMigrateCoco(unittest.TestCase):
    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yolozu", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_migrate_coco_dataset_wrapper_manifest(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            coco_root = root / "coco"
            (coco_root / "images" / "val2017").mkdir(parents=True, exist_ok=True)
            (coco_root / "annotations").mkdir(parents=True, exist_ok=True)

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 100, "height": 200}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "iscrowd": 0}],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = coco_root / "annotations" / "instances_val2017.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            out_root = root / "out"
            proc = self._run(
                [
                    "migrate",
                    "dataset",
                    "--from",
                    "coco",
                    "--coco-root",
                    str(coco_root),
                    "--split",
                    "val2017",
                    "--output",
                    str(out_root),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"migrate dataset --from coco failed:\n{proc.stdout}\n{proc.stderr}")

            wrapper = out_root / "dataset.json"
            self.assertTrue(wrapper.is_file())
            label_path = out_root / "labels" / "val2017" / "0001.txt"
            self.assertTrue(label_path.is_file())
            self.assertTrue((out_root / "labels" / "val2017" / "classes.txt").is_file())

            proc2 = self._run(
                [
                    "validate",
                    "dataset",
                    str(out_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--no-check-images",
                ],
                cwd=repo_root,
            )
            if proc2.returncode != 0:
                self.fail(f"validate dataset (no-check-images) failed:\n{proc2.stdout}\n{proc2.stderr}")

    def test_migrate_coco_results_predictions_and_validate(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            instances = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 100, "height": 200}],
                "annotations": [],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = root / "instances.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            results = [{"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "score": 0.9}]
            results_path = root / "results.json"
            results_path.write_text(json.dumps(results), encoding="utf-8")

            out_preds = root / "predictions.json"
            proc = self._run(
                [
                    "migrate",
                    "predictions",
                    "--from",
                    "coco-results",
                    "--results",
                    str(results_path),
                    "--instances",
                    str(instances_path),
                    "--output",
                    str(out_preds),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"migrate predictions --from coco-results failed:\n{proc.stdout}\n{proc.stderr}")

            proc2 = self._run(["validate", "predictions", str(out_preds), "--strict"], cwd=repo_root)
            if proc2.returncode != 0:
                self.fail(f"validate predictions --strict failed:\n{proc2.stdout}\n{proc2.stderr}")

            payload = json.loads(out_preds.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["image"], "0001.jpg")
            det = payload[0]["detections"][0]
            self.assertEqual(int(det["class_id"]), 0)
            bbox = det["bbox"]
            self.assertAlmostEqual(float(bbox["cx"]), 0.05, places=6)
            self.assertAlmostEqual(float(bbox["cy"]), 0.05, places=6)
            self.assertAlmostEqual(float(bbox["w"]), 0.1, places=6)
            self.assertAlmostEqual(float(bbox["h"]), 0.1, places=6)

    def test_migrate_coco_results_predictions_skips_invalid_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)

            instances = {
                "images": [
                    {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 200},
                    {"id": "bad", "file_name": "ignored.jpg", "width": 10, "height": 10},
                ],
                "annotations": [],
                "categories": [{"id": 7, "name": "thing"}],
            }
            instances_path = root / "instances.json"
            instances_path.write_text(json.dumps(instances), encoding="utf-8")

            results = [
                {"image_id": "bad", "category_id": 7, "bbox": [0, 0, 10, 20], "score": 0.8},
                {"image_id": 1, "category_id": "bad", "bbox": [0, 0, 10, 20], "score": 0.8},
                {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "score": True},
                {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 20], "score": 0.7},
            ]
            results_path = root / "results.json"
            results_path.write_text(json.dumps(results), encoding="utf-8")

            out_preds = root / "predictions.json"
            proc = self._run(
                [
                    "migrate",
                    "predictions",
                    "--from",
                    "coco-results",
                    "--results",
                    str(results_path),
                    "--instances",
                    str(instances_path),
                    "--output",
                    str(out_preds),
                    "--force",
                ],
                cwd=repo_root,
            )
            if proc.returncode != 0:
                self.fail(f"migrate predictions --from coco-results failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_preds.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(len(payload[0]["detections"]), 1)
            self.assertEqual(float(payload[0]["detections"][0]["score"]), 0.7)


if __name__ == "__main__":
    unittest.main()
