import subprocess
import unittest
from pathlib import Path


class TestSmokeCliHelp(unittest.TestCase):
    def test_smoke_script_supports_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "smoke.sh"
        self.assertTrue(script.exists(), "missing scripts/smoke.sh")

        proc = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"smoke --help failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("Usage:", out)
        self.assertIn("--dataset", out)
        self.assertIn("--report", out)
        self.assertIn("--synthgen-root", out)
        self.assertIn("--demo-run-dir", out)
        self.assertIn("--skip-demo", out)


if __name__ == "__main__":
    unittest.main()
