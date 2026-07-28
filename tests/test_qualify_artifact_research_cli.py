import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestQualifyArtifactResearchCLI(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "tools" / "qualify_artifact_research.py"

    def test_help(self):
        proc = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("three deterministic", proc.stdout)

    def test_refuses_existing_output_before_running(self):
        with tempfile.TemporaryDirectory(dir=self.repo_root) as temp_dir:
            proc = subprocess.run(
                [sys.executable, str(self.script), "--output-dir", temp_dir],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing to replace existing output directory", proc.stderr)

    def test_rejects_source_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            student = Path(temp_dir) / "student.json"
            student.write_text("[]", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(self.script), "--student", str(student)],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("student must stay inside the repository", proc.stderr)

    def test_requires_three_repetitions(self):
        proc = subprocess.run(
            [sys.executable, str(self.script), "--repeats", "2"],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--repeats must be >= 3", proc.stderr)


if __name__ == "__main__":
    unittest.main()
