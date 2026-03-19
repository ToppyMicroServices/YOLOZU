import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from yolozu.eval import benchmark_mode


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
        self.assertIn("--torch-model", proc.stdout)
        self.assertIn("--onnx-model", proc.stdout)
        self.assertIn("--engine-model", proc.stdout)
        self.assertIn("--protocol", proc.stdout)
        self.assertIn("--task", proc.stdout)
        self.assertIn("pose6d", proc.stdout)

    def test_torchscript_is_accepted_as_benchmark_format(self):
        self.assertIn("torchscript", benchmark_mode.PHASE1_FORMATS)

    def test_task_alias_pose_canonicalizes_to_keypoints(self):
        args = self._args(format="torchscript", model="runs/foo/model.torchscript", task="pose", dry_run=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["task"], "keypoints")
        self.assertEqual(report["task_requested"], "pose")
        self.assertEqual(report["task_semantics"]["metric_family"], "oks_map")
        self.assertIn("pose", report["task_semantics"]["accepted_aliases"])
        result = report["results"][0]
        self.assertEqual(result["task"], "keypoints")
        self.assertEqual(result["task_requested"], "pose")

    def test_classification_task_records_topk_semantics(self):
        args = self._args(format="torchscript", model="runs/foo/model.torchscript", task="classification", dry_run=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["task"], "classification")
        self.assertEqual(report["task_semantics"]["metric_family"], "topk_accuracy")
        self.assertEqual(report["task_semantics"]["support_level"], "documented_planned")
        self.assertEqual(report["task_semantics"]["expected_metric_keys"], ["top1", "top5", "accuracy"])

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

            artifacts = result.get("artifacts") or {}
            for key in ("predictions", "eval", "parity", "export_settings"):
                artifact_path = Path(str(artifacts.get(key)))
                self.assertTrue(artifact_path.is_file(), f"missing benchmark artifact: {artifact_path}")

    def test_module_cli_dry_run_supports_torchscript(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report = root / "benchmark_report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--model",
                    "runs/foo/model.torchscript",
                    "--data",
                    "data/smoke",
                    "--format",
                    "torchscript",
                    "--dry-run",
                    "--output",
                    str(report),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"yolozu benchmark --format torchscript --dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), ["torchscript"])
            results = payload.get("results") or []
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.get("format"), "torchscript")
            self.assertIn(result.get("status"), ("dry_run", "skipped"))
            if result.get("status") == "skipped":
                self.assertEqual(result.get("skip_reason"), "missing_runtime_dependency")
            artifacts = result.get("artifacts") or {}
            self.assertTrue(str(artifacts.get("predictions", "")).endswith("predictions_torchscript.json"))
            self.assertTrue(str(artifacts.get("eval", "")).endswith("eval_torchscript.json"))
            self.assertTrue(str(artifacts.get("parity", "")).endswith("parity_torchscript.json"))

    def _args(self, **overrides):
        root = Path(__file__).resolve().parents[1]
        base = dict(
            model="runs/foo/model.pt",
            torch_model=None,
            onnx_model=None,
            engine_model=None,
            data=str(root / "data" / "smoke"),
            imgsz=640,
            half=False,
            int8=False,
            device="cpu",
            verbose=False,
            format="torch",
            task="detect",
            split="val",
            protocol=None,
            max_images=2,
            dry_run=False,
            strict=False,
            repro_policy="relaxed",
            runtime_lock="none",
            run_id="unit-test-run",
            output=str(root / "tmp_benchmark_report.json"),
            history=None,
            predictions_output=None,
            eval_output=None,
            parity_output=None,
            batch=1,
            dynamic=False,
            nms=False,
            simplify=False,
            opset=17,
            workspace=4.0,
            fraction=1.0,
            latency_source="auto",
            iterations=50,
            warmup=5,
            sleep_s=0.0,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_real_torch_backend_orchestration_runs_export_and_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_path = root / "benchmark_report.json"

            def fake_run(cmd, cwd, capture_output, text, check):
                cmd = [str(x) for x in cmd]
                if cmd[1].endswith("export_predictions_ultralytics.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {
                        "predictions": [
                            {"image": "images/val/000001.jpg", "detections": []},
                            {"image": "images/val/000002.jpg", "detections": []},
                        ],
                        "meta": {"adapter": "ultralytics"},
                    }
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                if cmd[1].endswith("eval_suite.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {"metrics": {"bbox_mAP50": 0.42}}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                self.fail(f"unexpected subprocess command: {cmd}")

            def fake_module_available(name):
                return name == "ultralytics"

            args = self._args(output=str(report_path), data=str(repo_root / "data" / "smoke"), format="torch")
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=fake_module_available):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            result = report["results"][0]
            self.assertEqual(result["format"], "torch")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["latency_source"], "dataset_pass_wall_time")
            self.assertEqual(result["throughput"]["images"], 2)
            self.assertEqual(result["eval_metrics"]["bbox_mAP50"], 0.42)
            self.assertTrue(Path(result["artifacts"]["predictions"]).is_file())
            self.assertTrue(Path(result["artifacts"]["eval"]).is_file())
            self.assertTrue(Path(result["artifacts"]["parity"]).is_file())

    def test_real_onnx_backend_skips_without_backend_artifact(self):
        args = self._args(format="onnx", model="runs/foo/model.pt", latency_source="dataset_pass_wall_time")
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "onnxruntime"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "skipped")
        result = report["results"][0]
        self.assertEqual(result["format"], "onnx")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["skip_reason"], "model_artifact_required")

    def test_torchscript_supported_runtime_uses_synthetic_semantics_for_now(self):
        args = self._args(format="torchscript", model="runs/foo/model.torchscript", latency_source="auto")
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "ok")
        result = report["results"][0]
        self.assertEqual(result["format"], "torchscript")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["latency_source"], "synthetic_step")
        self.assertTrue(result["artifacts"]["predictions"].endswith("predictions_torchscript.json"))

    def test_real_torch_and_onnx_backends_write_real_parity_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        image_1 = str((repo_root / "data" / "smoke" / "images" / "val" / "000000000009.jpg").resolve())
        image_2 = str((repo_root / "data" / "smoke" / "images" / "val" / "000000000025.jpg").resolve())
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_path = root / "benchmark_report.json"

            def fake_run(cmd, cwd, capture_output, text, check):
                cmd = [str(x) for x in cmd]
                if cmd[1].endswith("export_predictions_ultralytics.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {
                        "predictions": [
                            {
                                "image": image_1,
                                "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}}],
                            },
                            {
                                "image": image_2,
                                "detections": [{"class_id": 1, "score": 0.8, "bbox": {"cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}}],
                            },
                        ],
                    }
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                if cmd[1].endswith("export_predictions_onnxrt.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {
                        "predictions": [
                            {
                                "image": image_1,
                                "detections": [{"class_id": 0, "score": 0.90001, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}}],
                            },
                            {
                                "image": image_2,
                                "detections": [{"class_id": 1, "score": 0.80001, "bbox": {"cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}}],
                            },
                        ],
                    }
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                if cmd[1].endswith("eval_suite.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {"metrics": {"bbox_mAP50": 0.42}}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                self.fail(f"unexpected subprocess command: {cmd}")

            def fake_module_available(name):
                return name in {"ultralytics", "onnxruntime"}

            args = self._args(
                output=str(report_path),
                data=str(repo_root / "data" / "smoke"),
                format="torch,onnx",
                onnx_model="exports/example.onnx",
            )
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=fake_module_available):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            results = {item["format"]: item for item in report["results"]}
            self.assertEqual(results["torch"]["parity"]["reference_backend"], "torch")
            self.assertEqual(results["torch"]["parity"]["candidate_backends"], ["onnx"])
            self.assertTrue(Path(results["torch"]["artifacts"]["parity"]).is_file())
            torch_parity_payload = json.loads(Path(results["torch"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(torch_parity_payload["kind"], "benchmark_parity_reference")

            self.assertTrue(Path(results["onnx"]["artifacts"]["parity"]).is_file())
            onnx_parity_payload = json.loads(Path(results["onnx"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(onnx_parity_payload["kind"], "benchmark_parity_report")
            self.assertEqual(onnx_parity_payload["reference_backend"], "torch")
            self.assertEqual(onnx_parity_payload["candidate_backend"], "onnx")
            self.assertTrue(onnx_parity_payload["summary"]["ok"])


if __name__ == "__main__":
    unittest.main()
