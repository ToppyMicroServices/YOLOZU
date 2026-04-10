import subprocess
import sys
import unittest
from pathlib import Path


class TestSupportExternalTrainingTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        self.assertTrue(script.is_file(), "missing tools/support_external_training.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"support_external_training --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("train-yolox", proc.stdout)
        self.assertIn("train-ultralytics", proc.stdout)
        self.assertIn("train-hf-detr", proc.stdout)


if __name__ == "__main__":
    unittest.main()
