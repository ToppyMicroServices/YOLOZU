import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import os


class TestDemoCliDefaults(unittest.TestCase):
    def test_demo_defaults_to_demo_suite(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            cwd = Path(td)
            env = dict(os.environ)
            py_path = str(repo_root)
            if env.get("PYTHONPATH"):
                py_path = py_path + os.pathsep + str(env["PYTHONPATH"])
            env["PYTHONPATH"] = py_path
            proc = subprocess.run(
                [sys.executable, "-m", "yolozu", "demo"],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            # Default should run the demo suite (at least instance-seg synthetic).
            self.assertIn("instance-seg demo:", proc.stdout)
            self.assertIn("== instance-seg (synthetic) ==", proc.stdout)
            self.assertIn("output_dir:", proc.stdout)
            # Default output folder should be demo_output.
            self.assertIn("demo_output", proc.stdout)


if __name__ == "__main__":
    unittest.main()
