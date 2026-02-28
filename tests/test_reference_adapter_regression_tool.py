import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestReferenceAdapterRegressionTool(unittest.TestCase):
    def test_write_and_check_baseline(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_reference_adapter_regression.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline_path = root / "baseline.json"
            write_report = root / "write_report.json"
            check_report = root / "check_report.json"

            common = [
                "--dataset",
                "data/smoke",
                "--split",
                "val",
                "--max-images",
                "1",
                "--device",
                "cpu",
                "--image-size",
                "96",
                "--score-threshold",
                "0.05",
                "--max-detections",
                "10",
                "--init-seed",
                "2026",
                "--baseline",
                str(baseline_path.relative_to(repo_root)),
            ]

            write_cmd = [
                sys.executable,
                str(script),
                *common,
                "--output",
                str(write_report.relative_to(repo_root)),
                "--write-baseline",
            ]
            write_proc = subprocess.run(
                write_cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if write_proc.returncode != 0:
                self.fail(
                    "run_reference_adapter_regression --write-baseline failed:\n"
                    f"STDOUT:\n{write_proc.stdout}\nSTDERR:\n{write_proc.stderr}"
                )

            check_cmd = [
                sys.executable,
                str(script),
                *common,
                "--output",
                str(check_report.relative_to(repo_root)),
            ]
            check_proc = subprocess.run(
                check_cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if check_proc.returncode != 0:
                self.fail(
                    "run_reference_adapter_regression baseline check failed:\n"
                    f"STDOUT:\n{check_proc.stdout}\nSTDERR:\n{check_proc.stderr}"
                )

            report = json.loads(check_report.read_text(encoding="utf-8"))
            self.assertTrue(report.get("ok"))
            gates = report.get("gates") or {}
            self.assertTrue(gates.get("schema_drift", {}).get("ok"))
            self.assertTrue(gates.get("consistency_drift", {}).get("ok"))
            self.assertTrue(gates.get("metric_drift", {}).get("ok"))
            self.assertTrue(gates.get("speed_drift", {}).get("ok"))


if __name__ == "__main__":
    unittest.main()
