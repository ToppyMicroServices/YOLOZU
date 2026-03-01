import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestReferenceAdapterRegressionTool(unittest.TestCase):
    def _load_tool_module(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_reference_adapter_regression.py"
        spec = importlib.util.spec_from_file_location("reference_adapter_regression_tool", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _run(self, cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_matrix_baseline_path_resolution(self):
        repo_root = Path(__file__).resolve().parents[1]
        mod = self._load_tool_module()
        args = mod._parse_args(
            [
                "--baseline-layout",
                "matrix",
                "--baseline-root",
                "baselines/reference_adapter",
                "--adapter-id",
                "rtdetr_pose",
                "--backend-id",
                "onnxrt",
                "--device",
                "cuda:0",
                "--baseline-version",
                "v3",
                "--profile",
                "full",
            ]
        )
        path = mod._resolve_baseline_path(args=args, cwd=repo_root)
        expected = repo_root / "baselines/reference_adapter/rtdetr_pose/onnxrt/cuda-0/v3/full.json"
        self.assertEqual(path, expected)

    def test_collect_provenance_minimal(self):
        mod = self._load_tool_module()
        provenance = mod._collect_provenance(
            capture_mode="minimal",
            runtime_lock={"sha256": "abc123", "path": "requirements-ci.lock"},
        )
        self.assertEqual(provenance.get("capture_mode"), "minimal")
        snapshot = provenance.get("snapshot") or {}
        self.assertTrue(bool(snapshot.get("enabled")))
        self.assertEqual(snapshot.get("runtime_lock_sha256"), "abc123")
        self.assertIn("pip_freeze_sha256", snapshot)
        self.assertIn("generator", provenance)

    def test_metric_gate_checks_robust_metrics(self):
        mod = self._load_tool_module()
        baseline_payload = {
            "baseline": {
                "summary": {
                    "total_detections": 10,
                    "score_sum": 5.0,
                    "score_mean": 0.5,
                    "bbox_checksum": 2.0,
                },
                "robust_metrics": {
                    "map50": 0.8,
                },
                "speed": {"fps": 10.0},
                "contract": {},
            },
            "thresholds": {
                "metric": {
                    "total_detections_abs": 0.0,
                    "score_sum_abs": 0.0,
                    "score_mean_abs": 0.0,
                    "bbox_checksum_abs": 0.0,
                    "map50_abs": 0.01,
                },
                "speed": {"min_fps_ratio": 0.0, "absolute_floor_fps": 0.0},
            },
            "baseline_meta": {},
        }
        failure_records: list[dict[str, object]] = []
        gates, hard_failures, soft_failures = mod._compare_against_baseline(
            baseline_payload=baseline_payload,
            summary={
                "total_detections": 10,
                "score_sum": 5.0,
                "score_mean": 0.5,
                "bbox_checksum": 2.0,
                "class_hist": {},
            },
            robust_metrics={"map50": 0.6},
            speed={"fps": 10.0, "images": 1, "seconds": 0.1},
            contract={},
            run_meta={},
            schema_warnings=[],
            schema_errors=[],
            consistency_errors=[],
            contract_errors=[],
            gate_policy={
                mod.GATE_SCHEMA: "off",
                mod.GATE_CONSISTENCY: "off",
                mod.GATE_METRIC: "hard",
                mod.GATE_SPEED: "off",
            },
            predictions=[],
            enforce_runtime_lock=False,
            enforce_weights_hash=False,
            peer_robust_metrics=None,
            backend_parity={"mode": "off"},
            failure_records=failure_records,
        )
        self.assertFalse(bool(gates[mod.GATE_METRIC]["ok"]))
        self.assertTrue(any("map50" in item for item in hard_failures))
        self.assertEqual(soft_failures, [])

    def test_record_preflight_classifies_missing_image_as_e_io(self):
        mod = self._load_tool_module()
        records = [{"image": "data/does_not_exist_for_preflight.jpg", "labels": []}]
        _meta, errors = mod._preflight_records(
            records,
            dataset_root=Path(__file__).resolve().parents[1],
            image_size=(160, 160),
        )
        self.assertTrue(any(str(item).startswith("E_IO:") for item in errors), msg=f"errors={errors}")

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

    def test_contract_gate_enforces_weights_hash_when_flag_enabled(self):
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
            baseline_meta["weights_hash"] = "deadbeef"
            baseline_payload["baseline_meta"] = baseline_meta
            baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            check_cmd = [
                sys.executable,
                str(script),
                *common,
                "--enforce-weights-hash",
                "--output",
                str(check_report.relative_to(repo_root)),
            ]
            check_proc = self._run(check_cmd, cwd=repo_root)
            self.assertNotEqual(
                check_proc.returncode,
                0,
                msg=(
                    "enforced weights_hash mismatch must fail hard gate:\n"
                    f"STDOUT:\n{check_proc.stdout}\nSTDERR:\n{check_proc.stderr}"
                ),
            )

            payload = json.loads(check_report.read_text(encoding="utf-8"))
            hard_failures = payload.get("hard_failures") or []
            self.assertTrue(any("E_CANON_WEIGHTS_HASH" in str(item) for item in hard_failures))

    def test_contract_gate_enforces_runtime_lock_hash_when_flag_enabled(self):
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
                "--runtime-lock",
                "requirements-ci.lock",
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
            baseline_meta["runtime_lock_sha256"] = "deadbeef"
            baseline_payload["baseline_meta"] = baseline_meta
            baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            check_cmd = [
                sys.executable,
                str(script),
                *common,
                "--enforce-runtime-lock",
                "--output",
                str(check_report.relative_to(repo_root)),
            ]
            check_proc = self._run(check_cmd, cwd=repo_root)
            self.assertNotEqual(
                check_proc.returncode,
                0,
                msg=(
                    "enforced runtime lock mismatch must fail hard gate:\n"
                    f"STDOUT:\n{check_proc.stdout}\nSTDERR:\n{check_proc.stderr}"
                ),
            )

            payload = json.loads(check_report.read_text(encoding="utf-8"))
            hard_failures = payload.get("hard_failures") or []
            self.assertTrue(any("E_CANON_RUNTIME_LOCK" in str(item) for item in hard_failures))

    def test_fixed_real_scenario_write_and_check(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        repo_root = Path(__file__).resolve().parents[1]
        dataset_root = repo_root / "data" / "real_multitask_fewshot"
        if not dataset_root.exists():
            self.skipTest("real_multitask_fewshot dataset is not available")

        script = repo_root / "tools" / "run_reference_adapter_regression.py"
        self.assertTrue(script.is_file())

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            baseline_path = root / "real_scenario_baseline.json"
            write_report = root / "real_scenario_write_report.json"
            check_report = root / "real_scenario_check_report.json"

            common = [
                "--dataset",
                "data/real_multitask_fewshot",
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
                "--write-baseline",
                "--output",
                str(write_report.relative_to(repo_root)),
            ]
            write_proc = self._run(write_cmd, cwd=repo_root)
            if write_proc.returncode != 0:
                self.fail(
                    "fixed real scenario baseline write failed:\n"
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
                    "fixed real scenario baseline check failed:\n"
                    f"STDOUT:\n{check_proc.stdout}\nSTDERR:\n{check_proc.stderr}"
                )

            payload = json.loads(check_report.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("ok"), msg=f"expected fixed real scenario to pass: {payload}")
            self.assertEqual(str((payload.get("run") or {}).get("dataset")), "data/real_multitask_fewshot")
            gates = payload.get("gates") or {}
            self.assertTrue(gates.get("schema_drift", {}).get("ok"))
            self.assertTrue(gates.get("consistency_drift", {}).get("ok"))


if __name__ == "__main__":
    unittest.main()
