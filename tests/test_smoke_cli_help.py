import os
import subprocess
import tempfile
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
        self.assertIn("--torch-device", out)
        self.assertIn("--profile", out)
        self.assertIn("--walkthrough-report", out)

    def test_smoke_script_falls_back_when_yolozu_python_candidate_is_broken(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "smoke.sh"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            fake_python = Path(td) / "fake-python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            fake_python.chmod(0o755)

            env = dict(os.environ)
            env["YOLOZU_PYTHON"] = str(fake_python)
            proc = subprocess.run(
                ["bash", str(script), "--help"],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"smoke --help fallback failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self.assertIn("Usage:", out)
            self.assertIn("--profile", out)

    def test_smoke_script_accepts_repo_local_python_via_pythonpath_bootstrap(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "smoke.sh"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            wrapper = Path(td) / "python-wrapper"
            wrapper.write_text(
                """#!/usr/bin/env bash
real_python="$(command -v python3)"
if [[ "${1:-}" == "-" ]]; then
  exec "$real_python" -I "$@"
fi
exec "$real_python" "$@"
""",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

            env = dict(os.environ)
            env["YOLOZU_PYTHON"] = str(wrapper)
            proc = subprocess.run(
                ["bash", str(script), "--help"],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(
                    "smoke --help pythonpath bootstrap failed:\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self.assertIn("Usage:", out)
            self.assertIn("--walkthrough-report", out)


if __name__ == "__main__":
    unittest.main()
