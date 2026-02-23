import subprocess
import sys
import unittest
from pathlib import Path


class TestSchemaCompatibilityTool(unittest.TestCase):
    def test_schema_compatibility_tool_passes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "check_schema_compatibility.py"
        self.assertTrue(script.is_file())

        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"check_schema_compatibility.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        self.assertIn("schema compatibility gate passed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
