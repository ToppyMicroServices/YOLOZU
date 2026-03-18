import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestBenchmarkModelTool(unittest.TestCase):
    def test_tool_help_lists_phase1_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "benchmark_model.py"

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"benchmark_model.py --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--format", proc.stdout)
        self.assertIn("--runtime-lock", proc.stdout)
        self.assertIn("--latency-source", proc.stdout)
        self.assertIn("--predictions-output", proc.stdout)

    def test_module_cli_dry_run_writes_stable_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report = root / "benchmark_report.json"
            artifact_dir = root / "artifacts"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--model",
                    "runs/foo/model.pt",
                    "--data",
                    "data/smoke",
                    "--format",
                    "engine",
                    "--dry-run",
                    "--output",
                    str(report),
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu benchmark --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(report.is_file(), "expected benchmark report JSON")
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("kind"), "yolozu_benchmark_report")
            self.assertEqual(payload.get("schema_version"), 1)
            self.assertEqual(payload.get("format"), ["engine"])
            self.assertEqual((payload.get("run_meta") or {}).get("runtime_lock"), "none")

            results = payload.get("results") or []
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.get("format"), "engine")
            self.assertIn(result.get("status"), ("dry_run", "skipped"))
            if result.get("status") == "skipped":
                self.assertTrue(result.get("skip_reason"))

            for rel in (
                artifact_dir / "predictions_engine.json",
                artifact_dir / "eval_engine.json",
                artifact_dir / "parity_engine.json",
                root / "export_settings_engine.json",
            ):
                self.assertTrue(rel.is_file(), f"missing benchmark artifact: {rel}")


if __name__ == "__main__":
    unittest.main()
