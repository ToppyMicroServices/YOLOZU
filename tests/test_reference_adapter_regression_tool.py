import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestReferenceAdapterRegressionTool(unittest.TestCase):
    def _run(self, cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

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
            write_proc = self._run(write_cmd, cwd=repo_root)
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
            check_proc = self._run(check_cmd, cwd=repo_root)
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
            self.assertIn("run_meta", report)
            self.assertIn("baseline_meta", report)
            self.assertIn("gate_policy", report)
            self.assertIn("protocol", report)

    def test_behavior_gate_warn_then_hard(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_reference_adapter_regression.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline_path = root / "baseline.json"
            write_report = root / "write_report.json"
            warn_report = root / "warn_report.json"
            hard_report = root / "hard_report.json"

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
                "--init-seed",
                "2026",
                "--baseline",
                str(baseline_path.relative_to(repo_root)),
                "--perf-gate-mode",
                "off",
            ]

            write_cmd = [
                sys.executable,
                str(script),
                *common,
                "--max-detections",
                "10",
                "--output",
                str(write_report.relative_to(repo_root)),
                "--write-baseline",
            ]
            write_proc = self._run(write_cmd, cwd=repo_root)
            if write_proc.returncode != 0:
                self.fail(
                    "baseline write failed:\n"
                    f"STDOUT:\n{write_proc.stdout}\nSTDERR:\n{write_proc.stderr}"
                )

            warn_cmd = [
                sys.executable,
                str(script),
                *common,
                "--max-detections",
                "1",
                "--score-gate-mode",
                "warn",
                "--output",
                str(warn_report.relative_to(repo_root)),
            ]
            warn_proc = self._run(warn_cmd, cwd=repo_root)
            if warn_proc.returncode != 0:
                self.fail(
                    "warn-mode comparison should not fail process:\n"
                    f"STDOUT:\n{warn_proc.stdout}\nSTDERR:\n{warn_proc.stderr}"
                )
            warn_payload = json.loads(warn_report.read_text(encoding="utf-8"))
            self.assertTrue(bool(warn_payload.get("ok")))
            self.assertFalse(bool((warn_payload.get("gates") or {}).get("metric_drift", {}).get("ok")))
            self.assertTrue(any("[metric_drift]" in x for x in (warn_payload.get("soft_failures") or [])))

            hard_cmd = [
                sys.executable,
                str(script),
                *common,
                "--max-detections",
                "1",
                "--score-gate-mode",
                "hard",
                "--output",
                str(hard_report.relative_to(repo_root)),
            ]
            hard_proc = self._run(hard_cmd, cwd=repo_root)
            self.assertNotEqual(
                hard_proc.returncode,
                0,
                msg=(
                    "hard-mode comparison must fail when behavior drifts\n"
                    f"STDOUT:\n{hard_proc.stdout}\nSTDERR:\n{hard_proc.stderr}"
                ),
            )
            hard_payload = json.loads(hard_report.read_text(encoding="utf-8"))
            self.assertFalse(bool(hard_payload.get("ok")))
            self.assertTrue(any("[metric_drift]" in x for x in (hard_payload.get("hard_failures") or [])))

    def test_contract_gate_skips_weights_hash_without_checkpoint(self):
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
                "--score-gate-mode",
                "off",
                "--perf-gate-mode",
                "off",
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
            write_proc = self._run(write_cmd, cwd=repo_root)
            if write_proc.returncode != 0:
                self.fail(
                    "baseline write failed:\n"
                    f"STDOUT:\n{write_proc.stdout}\nSTDERR:\n{write_proc.stderr}"
                )

            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_meta = baseline_payload.get("baseline_meta") or {}
            baseline_meta["checkpoint_hash"] = None
            baseline_meta["weights_hash"] = "deadbeef"
            baseline_payload["baseline_meta"] = baseline_meta
            baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            check_cmd = [
                sys.executable,
                str(script),
                *common,
                "--output",
                str(check_report.relative_to(repo_root)),
            ]
            check_proc = self._run(check_cmd, cwd=repo_root)
            if check_proc.returncode != 0:
                self.fail(
                    "checkpoint-free weights_hash mismatch should not fail hard gate:\n"
                    f"STDOUT:\n{check_proc.stdout}\nSTDERR:\n{check_proc.stderr}"
                )

            payload = json.loads(check_report.read_text(encoding="utf-8"))
            consistency = (payload.get("gates") or {}).get("consistency_drift") or {}
            self.assertTrue(bool(consistency.get("ok")))
            warnings = (consistency.get("details") or {}).get("warnings") or []
            self.assertTrue(
                any("weights_hash differs in checkpoint-free comparison" in str(item) for item in warnings),
                msg=f"expected checkpoint-free weights warning, got: {warnings}",
            )


if __name__ == "__main__":
    unittest.main()
