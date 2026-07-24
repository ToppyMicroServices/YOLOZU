import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalCLIGuardrails(unittest.TestCase):
    @staticmethod
    def _invalid_predictions() -> dict:
        return {
            "schema_version": 1,
            "predictions": [
                {
                    "schema_version": 2,
                    "image": "000000000009.jpg",
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 1.5,
                            "bbox": {"cx": 1.2, "cy": 0.5, "w": 0.2, "h": 0.2},
                        }
                    ],
                }
            ],
        }

    def test_eval_coco_rejects_empty_dataset(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_coco.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = root / "dataset"
            (dataset / "images" / "val").mkdir(parents=True, exist_ok=True)
            (dataset / "labels" / "val").mkdir(parents=True, exist_ok=True)
            preds = root / "preds.json"
            preds.write_text(json.dumps([]), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "-d",
                    str(dataset),
                    "-s",
                    "val",
                    "-p",
                    str(preds),
                    "--dry-run",
                    "-o",
                    str(root / "report.json"),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_DATASET_EMPTY]", proc.stderr)
            self.assertIn("no dataset images resolved", proc.stderr)

    def test_eval_coco_rejects_empty_predictions_entries(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "eval_coco.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = repo_root / "data" / "smoke"
            preds = root / "preds.json"
            preds.write_text(json.dumps([]), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "-d",
                    str(dataset),
                    "-s",
                    "val",
                    "-p",
                    str(preds),
                    "--dry-run",
                    "--output",
                    str(root / "report.json"),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_PREDICTIONS_EMPTY]", proc.stderr)
            self.assertIn("no prediction entries found", proc.stderr)

    def test_eval_coco_strict_default_overwrites_stale_success_with_failure(self):
        repo_root = Path(__file__).resolve().parents[1]
        output = None

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds = root / "invalid.json"
            output = root / "report.json"
            preds.write_text(json.dumps(self._invalid_predictions()), encoding="utf-8")
            output.write_text(json.dumps({"status": "ok", "ok": True}), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "eval-coco",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--predictions",
                    str(preds),
                    "--dry-run",
                    "--output",
                    str(output),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_PREDICTIONS_INVALID]", proc.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["code"], "E_PREDICTIONS_INVALID")
            self.assertEqual(report["counts"]["detections"], 0)

    def test_eval_coco_explicit_repair_succeeds_and_records_repairs(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds = root / "invalid.json"
            output = root / "report.json"
            preds.write_text(json.dumps(self._invalid_predictions()), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "tools" / "eval_coco.py"),
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--predictions",
                    str(preds),
                    "--dry-run",
                    "-r",
                    "-o",
                    str(output),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["validation"]["mode"], "repair")
            self.assertTrue(any("score: out of range" in warning for warning in report["warnings"]))
            self.assertTrue(any("bbox.cx: out of range" in warning for warning in report["warnings"]))

    def test_eval_coco_max_images_excludes_known_unselected_predictions(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            output = Path(td) / "report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "eval-coco",
                    "-d",
                    "data/smoke",
                    "-s",
                    "val",
                    "-p",
                    "data/smoke/predictions/predictions_dummy.json",
                    "-n",
                    "2",
                    "--dry-run",
                    "-o",
                    str(output),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["dataset_images_total"], 10)
            self.assertEqual(report["counts"]["images"], 2)
            self.assertEqual(report["counts"]["prediction_images_excluded"], 8)
            self.assertEqual(report["counts"]["detections_excluded"], 40)
            self.assertTrue(any("known but unselected" in warning for warning in report["warnings"]))

    def test_eval_coco_max_images_still_rejects_full_dataset_unknown(self):
        repo_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (repo_root / "data" / "smoke" / "predictions" / "predictions_dummy.json").read_text(encoding="utf-8")
        )
        payload["predictions"][0]["image"] = "not-in-dataset.jpg"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            preds = root / "unknown.json"
            output = root / "report.json"
            preds.write_text(json.dumps(payload), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "eval-coco",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--predictions",
                    str(preds),
                    "--max-images",
                    "2",
                    "--dry-run",
                    "--output",
                    str(output),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("[E_PREDICTION_UNKNOWN_IMAGE]", proc.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["error"]["code"], "E_PREDICTION_UNKNOWN_IMAGE")


if __name__ == "__main__":
    unittest.main()
