import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDistillPredictionsCLI(unittest.TestCase):
    def test_rejects_invalid_teacher_min_score(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "distill_predictions.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            student = root / "student.json"
            teacher = root / "teacher.json"
            out = root / "out.json"
            report = root / "report.json"
            student.write_text(json.dumps([{"image": "img.jpg", "detections": []}]))
            teacher.write_text(json.dumps([{"image": "img.jpg", "detections": []}]))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--student",
                    str(student),
                    "--teacher",
                    str(teacher),
                    "--output",
                    str(out),
                    "--output-report",
                    str(report),
                    "--teacher-min-score",
                    "1.2",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--teacher-min-score must be in [0.0, 1.0]", proc.stderr)

    def test_rejects_negative_max_added_per_image(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "distill_predictions.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            student = root / "student.json"
            teacher = root / "teacher.json"
            out = root / "out.json"
            report = root / "report.json"
            student.write_text(json.dumps([{"image": "img.jpg", "detections": []}]))
            teacher.write_text(json.dumps([{"image": "img.jpg", "detections": []}]))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--student",
                    str(student),
                    "--teacher",
                    str(teacher),
                    "--output",
                    str(out),
                    "--output-report",
                    str(report),
                    "--max-added-per-image",
                    "-1",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--max-added-per-image must be >= 0", proc.stderr)


if __name__ == "__main__":
    unittest.main()
