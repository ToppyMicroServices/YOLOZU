import json
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

    def test_dod_cpu_smoke_script_supports_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "dod_cpu_smoke.sh"
        self.assertTrue(script.exists(), "missing scripts/dod_cpu_smoke.sh")

        proc = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"dod_cpu_smoke --help failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("Usage:", out)
        self.assertIn("--run-dir", out)
        self.assertIn("--installed-package", out)
        self.assertIn("doctor --proof", out)

    def test_fresh_install_journey_script_supports_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "fresh_install_journey.sh"
        self.assertTrue(script.exists(), "missing scripts/fresh_install_journey.sh")

        proc = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(
                "fresh_install_journey --help failed:\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("Usage:", out)
        self.assertIn("--python", out)
        self.assertIn("--package", out)
        self.assertIn("--run-dir", out)

    def test_dod_cpu_smoke_script_runs_public_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "dod_cpu_smoke.sh"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            run_dir = Path(td) / "dod"
            proc = subprocess.run(
                ["bash", str(script), "--run-dir", str(run_dir)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"dod_cpu_smoke failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            report = run_dir / "dod_cpu_smoke_report.json"
            self.assertTrue(report.is_file())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "yolozu_dod_cpu_smoke")
            self.assertEqual(payload.get("status"), "pass")
            self.assertEqual(payload.get("execution", {}).get("mode"), "repo_checkout")
            steps = payload.get("execution", {}).get("steps", [])
            self.assertEqual(
                [step.get("name") for step in steps],
                ["doctor_proof", "demo_instance_seg", "validate_dataset", "validate_predictions", "eval_coco"],
            )
            self.assertTrue(all(step.get("exit_code") == 0 for step in steps))


if __name__ == "__main__":
    unittest.main()
