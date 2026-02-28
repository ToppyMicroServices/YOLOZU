import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvalCLIGuardrails(unittest.TestCase):
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
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--predictions",
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
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--predictions",
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


if __name__ == "__main__":
    unittest.main()
