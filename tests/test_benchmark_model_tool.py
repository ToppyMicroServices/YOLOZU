import json
import subprocess
import sys
import tempfile
from base64 import b64decode
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, mock

from yolozu import cli_entry
from yolozu.eval import benchmark_mode


class TestBenchmarkModelTool(TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self._root_artifacts = [
            self.repo_root / "tmp_benchmark_report.json",
            self.repo_root / "export_settings_engine.json",
            self.repo_root / "export_settings_onnx.json",
            self.repo_root / "export_settings_openvino.json",
            self.repo_root / "export_settings_opencv_dnn.json",
            self.repo_root / "export_settings_torch.json",
            self.repo_root / "export_settings_torchscript.json",
        ]
        for path in self._root_artifacts:
            path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        for path in self._root_artifacts:
            path.unlink(missing_ok=True)

    def test_tool_help_lists_phase1_flags(self):
        repo_root = self.repo_root
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
        self.assertIn("--openvino-model", proc.stdout)
        self.assertIn("--task", proc.stdout)
        self.assertIn("pose6d", proc.stdout)
        self.assertIn("--depth-mask", proc.stdout)
        self.assertIn("--depth-align", proc.stdout)
        self.assertIn("--segmentation-parity-mismatch-atol", proc.stdout)
        self.assertIn("--parity-reference-backend", proc.stdout)
        self.assertIn("artifact_eval", proc.stdout)
        self.assertIn("--keypoints-parity-iou-thresh", proc.stdout)
        self.assertIn("--keypoints-parity-kp-atol", proc.stdout)
        self.assertIn("--pose-parity-rot-deg-atol", proc.stdout)
        self.assertIn("--pose-parity-trans-atol", proc.stdout)
        normalized_help = " ".join(proc.stdout.split())
        self.assertIn("Must remain disabled when the effective latency source is artifact_eval", normalized_help)
        self.assertIn("Must remain 1 when the effective latency source is artifact_eval", normalized_help)

    def test_torchscript_is_accepted_as_benchmark_format(self):
        self.assertIn("torchscript", benchmark_mode.PHASE1_FORMATS)

    def test_openvino_is_accepted_as_conditional_benchmark_format(self):
        self.assertIn("openvino", benchmark_mode.PHASE1_FORMATS)
        self.assertIn("openvino", benchmark_mode.REAL_BACKEND_FORMATS)

    def test_artifact_eval_tasks_do_not_require_backend_runtimes(self):
        artifact_tasks = ("classification", "obb", "segmentation", "keypoints", "depth", "pose6d")
        expected_support_level = "artifact_backed_real_for_torch_onnx_engine_torchscript_openvino"
        with mock.patch.object(benchmark_mode, "_module_available", return_value=False):
            for task_label in artifact_tasks:
                self.assertEqual(
                    benchmark_mode._task_semantics(task_label)["support_level"],
                    expected_support_level,
                )
                for fmt in benchmark_mode.REAL_BACKEND_FORMATS:
                    with self.subTest(task=task_label, fmt=fmt):
                        supported, reason = benchmark_mode._support_status_for_format(
                            fmt,
                            device="cpu",
                            task_label=task_label,
                        )
                        self.assertTrue(supported)
                        self.assertIsNone(reason)

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
        self.assertEqual(report["task_semantics"]["support_level"], "artifact_backed_real_for_torch_onnx_engine_torchscript_openvino")
        self.assertEqual(report["task_semantics"]["expected_metric_keys"], ["top1", "top5", "accuracy"])
        self.assertEqual(report["execution_semantics"]["by_format"]["torchscript"]["execution_mode"], "dry_run_planning")
        self.assertEqual(report["results"][0]["status"], "dry_run")
        self.assertEqual(report["results"][0]["support_status"], "skipped")
        self.assertEqual(report["support_summary"]["by_format"]["torchscript"], "skipped")

    def test_dod_semantics_keep_requested_obb_formats_when_artifacts_missing(self):
        args = self._args(format="torch,onnx,engine,torchscript", task="obb", dry_run=False)
        with mock.patch.object(
            benchmark_mode.subprocess,
            "run",
            side_effect=AssertionError("artifact-backed OBB benchmark should not launch backends"),
        ):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        requested = ["torch", "onnx", "engine", "torchscript"]
        self.assertEqual(report["format"], requested)
        self.assertEqual(report["support_summary"]["requested_formats"], requested)
        self.assertEqual(report["support_summary"]["reported_formats"], requested)
        self.assertEqual(report["support_summary"]["missing_formats"], [])
        self.assertEqual(report["support_summary"]["counts"], {"real": 0, "artifact-backed": 0, "skipped": 4})
        for result in report["results"]:
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["skip_reason"], "model_artifact_required")
            self.assertEqual(result["support_status"], "skipped")
            self.assertEqual(result["support_reason"], "model_artifact_required")
            self.assertEqual(result["artifact_status"]["predictions"], "real")
            self.assertEqual(result["artifact_status"]["eval"], "real")
            self.assertEqual(result["artifact_status"]["parity"], "skipped")
            self.assertIn("available", result["runtime"])
            self.assertIn("predictions", result["artifacts"])
            self.assertIn("eval", result["artifacts"])
            self.assertIn("parity", result["artifacts"])

    def test_artifact_parity_expectations_match_shipped_task_support(self):
        for task_label in ("classification", "obb"):
            with self.subTest(task=task_label):
                semantics = benchmark_mode._task_execution_semantics(
                    task_label,
                    fmt="torch",
                    benchmark_source="artifact_eval",
                    dry_run=False,
                )
                self.assertEqual(semantics["artifact_expectation"]["parity"], "skipped")

        for task_label in ("segmentation", "keypoints", "depth", "pose6d"):
            with self.subTest(task=task_label):
                semantics = benchmark_mode._task_execution_semantics(
                    task_label,
                    fmt="torch",
                    benchmark_source="artifact_eval",
                    dry_run=False,
                )
                self.assertEqual(semantics["artifact_expectation"]["parity"], "real_when_comparable")

    def test_dod_semantics_keep_requested_detect_formats_when_runtimes_missing(self):
        args = self._args(format="torch,onnx,engine,torchscript", task="detect", dry_run=False)
        with mock.patch.object(benchmark_mode, "_module_available", return_value=False):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        requested = ["torch", "onnx", "engine", "torchscript"]
        self.assertEqual(report["support_summary"]["reported_formats"], requested)
        self.assertEqual(report["support_summary"]["missing_formats"], [])
        self.assertEqual(report["support_summary"]["counts"]["skipped"], 4)
        for result in report["results"]:
            self.assertEqual(result["support_status"], "skipped")
            self.assertTrue(result["support_reason"])
            self.assertIn("latency_source", result["runtime"])

    def test_openvino_missing_runtime_reports_skipped_without_install_requirement(self):
        args = self._args(format="openvino", task="detect", model="exports/example.xml", dry_run=False)
        with mock.patch.object(benchmark_mode, "_module_available", return_value=False):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["validation_summary"]["openvino_applicable"], True)
        self.assertEqual(report["support_summary"]["by_format"]["openvino"], "skipped")
        result = report["results"][0]
        self.assertEqual(result["format"], "openvino")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["skip_reason"], "missing_runtime_dependency")
        self.assertEqual(result["runtime"]["available"], False)
        self.assertEqual(result["runtime"]["reason"], "missing_runtime_dependency")

    def test_obb_task_writes_missing_artifact_placeholders_without_launching_backend(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_path = root / "benchmark_report.json"
            artifact_dir = root / "artifacts"
            args = self._args(
                format="torchscript",
                model="runs/foo/model.torchscript",
                task="obb",
                dry_run=False,
                output=str(report_path),
                predictions_output=str(artifact_dir),
                eval_output=str(artifact_dir),
                parity_output=str(artifact_dir),
                run_id="unwired-task-test",
            )

            with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(
                        benchmark_mode.subprocess,
                        "run",
                        side_effect=AssertionError("backend should not launch"),
                    ):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "skipped")
            self.assertTrue(report_path.is_file(), "expected benchmark report to be written")
            result = report["results"][0]
            self.assertEqual(result["format"], "torchscript")
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["skip_reason"], "model_artifact_required")
            self.assertEqual(result["execution_semantics"]["execution_mode"], "real_artifact_eval")

            expected_artifacts = {
                "predictions": "benchmark_predictions_placeholder",
                "eval": "benchmark_eval_placeholder",
                "parity": "benchmark_parity_placeholder",
            }
            for key, expected_kind in expected_artifacts.items():
                artifact_path = Path(result["artifacts"][key])
                self.assertTrue(artifact_path.is_file(), f"missing skipped artifact: {artifact_path}")
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["kind"], expected_kind)
                self.assertEqual(payload["format"], "torchscript")
                self.assertEqual(payload["status"], "skipped")
                self.assertEqual(payload["reason"], "model_artifact_required")
                self.assertEqual(payload["run_meta"]["backend"], "torchscript")
                self.assertEqual(payload["run_meta"]["run_id"], "unwired-task-test")

            export_settings = json.loads((root / "export_settings_torchscript.json").read_text(encoding="utf-8"))
            self.assertEqual(export_settings["status"], "supported")
            self.assertEqual(export_settings["skip_reason"], "model_artifact_required")
            self.assertEqual(export_settings["execution_semantics"]["execution_mode"], "real_artifact_eval")

    def test_unwired_benchmark_formats_report_skipped_not_synthetic_placeholder(self):
        args = self._args(format="opencv_dnn", model="runs/foo/model.onnx", task="detect", dry_run=False)
        with mock.patch.object(benchmark_mode, "_module_available", return_value=True):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "skipped")
        result = report["results"][0]
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["skip_reason"], "benchmark_format_not_wired")
        self.assertEqual(result["execution_semantics"]["execution_mode"], "unsupported_skipped")
        self.assertEqual(result["execution_semantics"]["artifact_expectation"]["predictions"], "skipped")
        format_notes = report["validation_summary"]["by_format"]["opencv_dnn"]["format_notes"]
        self.assertIn("not wired", format_notes)
        self.assertIn("unsupported/skipped", format_notes)
        self.assertNotIn("synthetic", format_notes)
        skipped_artifact = json.loads(Path(result["artifacts"]["predictions"]).read_text(encoding="utf-8"))
        self.assertEqual(skipped_artifact["kind"], "benchmark_predictions_skipped")

    def test_depth_task_records_yolozu_native_execution_semantics(self):
        args = self._args(format="torchscript", model="runs/foo/model.torchscript", task="depth", dry_run=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["task"], "depth")
        self.assertTrue(report["task_semantics"]["yolozu_native_extension"])
        by_format = report["execution_semantics"]["by_format"]["torchscript"]
        self.assertEqual(by_format["execution_mode"], "dry_run_planning")
        self.assertEqual(by_format["eval_expectation"]["metric_family"], "depth_error")
        self.assertEqual(by_format["artifact_expectation"]["eval"], "placeholder")

    def test_pose6d_task_records_yolozu_native_execution_semantics(self):
        args = self._args(format="torchscript", model="runs/foo/model.torchscript", task="pose6d", dry_run=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["task"], "pose6d")
        self.assertTrue(report["task_semantics"]["yolozu_native_extension"])
        result = report["results"][0]
        self.assertEqual(result["execution_semantics"]["eval_expectation"]["metric_family"], "pose6d_error")
        self.assertEqual(result["execution_semantics"]["artifact_expectation"]["parity"], "placeholder")

    def test_auto_prefers_artifact_eval_for_artifact_backed_tasks(self):
        args = self._args(latency_source="auto")
        for task_label in ("classification", "obb", "segmentation", "keypoints", "depth", "pose6d"):
            self.assertEqual(
                benchmark_mode._selected_benchmark_source(args, fmt="torch", task_label=task_label),
                "artifact_eval",
            )
            self.assertEqual(
                benchmark_mode._selected_benchmark_source(args, fmt="onnx", task_label=task_label),
                "artifact_eval",
            )
            self.assertEqual(
                benchmark_mode._selected_benchmark_source(args, fmt="engine", task_label=task_label),
                "artifact_eval",
            )

    def test_classification_task_supports_real_artifact_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            labels = root / "classification_labels.json"
            torch_pred = root / "classification_torch.json"
            onnx_pred = root / "classification_onnx.json"
            labels.write_text(
                json.dumps(
                    {
                        "classes": ["cat", "dog", "bird"],
                        "samples": [
                            {"id": "img0", "label": "cat"},
                            {"id": "img1", "label": "dog"},
                            {"id": "img2", "label": 2},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            torch_pred.write_text(
                json.dumps(
                    {
                        "classes": ["cat", "dog", "bird"],
                        "predictions": [
                            {"id": "img0", "scores": [0.9, 0.05, 0.05]},
                            {"id": "img1", "scores": [0.1, 0.8, 0.1]},
                            {"id": "img2", "scores": [0.1, 0.2, 0.7]},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            onnx_pred.write_text(
                json.dumps(
                    {
                        "classes": ["cat", "dog", "bird"],
                        "predictions": [
                            {"id": "img0", "scores": [0.9, 0.05, 0.05]},
                            {"id": "img1", "scores": [0.6, 0.3, 0.1]},
                            {"id": "img2", "scores": [0.1, 0.2, 0.7]},
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = root / "benchmark_classification_report.json"
            artifact_dir = root / "artifacts"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "classification",
                    "--model",
                    str(torch_pred),
                    "--onnx-model",
                    str(onnx_pred),
                    "--data",
                    str(labels),
                    "--format",
                    "torch,onnx,opencv_dnn",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark classification artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "classification")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["opencv_dnn"]["execution_mode"], "unsupported_skipped")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "ok")
            self.assertEqual(results["opencv_dnn"]["status"], "skipped")
            self.assertEqual(results["opencv_dnn"]["skip_reason"], "benchmark_format_not_wired")
            self.assertEqual(results["torch"]["support_status"], "artifact-backed")
            self.assertEqual(results["onnx"]["support_status"], "artifact-backed")
            self.assertEqual(payload["support_summary"]["counts"]["artifact-backed"], 2)
            self.assertEqual(payload["support_summary"]["counts"]["skipped"], 1)
            self.assertEqual(results["torch"]["eval_metrics"]["top1"], 1.0)
            self.assertEqual(results["onnx"]["eval_metrics"]["top1"], 2 / 3)
            self.assertEqual(results["onnx"]["eval_metrics"]["top5"], 1.0)
            self.assertEqual(
                json.loads(Path(results["torch"]["artifacts"]["eval"]).read_text(encoding="utf-8"))["kind"],
                "benchmark_classification_eval_report",
            )

    def test_obb_task_supports_real_artifact_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            labels = root / "obb_labels.json"
            torch_pred = root / "obb_torch.json"
            onnx_pred = root / "obb_onnx.json"
            labels.write_text(
                json.dumps(
                    {
                        "classes": ["ship"],
                        "samples": [
                            {
                                "id": "img0",
                                "objects": [
                                    {
                                        "class_id": 0,
                                        "obb": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2, "angle_deg": 30.0},
                                    }
                                ],
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            torch_pred.write_text(
                json.dumps(
                    {
                        "classes": ["ship"],
                        "predictions": [
                            {
                                "id": "img0",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.99,
                                        "obb": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2, "angle_deg": 30.0},
                                    }
                                ],
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            onnx_pred.write_text(
                json.dumps(
                    {
                        "classes": ["ship"],
                        "predictions": [
                            {
                                "id": "img0",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.99,
                                        "obb": {"cx": 0.52, "cy": 0.5, "w": 0.4, "h": 0.2, "angle_deg": 31.0},
                                    }
                                ],
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = root / "benchmark_obb_report.json"
            artifact_dir = root / "artifacts"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "obb",
                    "--model",
                    str(torch_pred),
                    "--onnx-model",
                    str(onnx_pred),
                    "--data",
                    str(labels),
                    "--format",
                    "torch,onnx,opencv_dnn",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark OBB artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "obb")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["opencv_dnn"]["execution_mode"], "unsupported_skipped")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "ok")
            self.assertEqual(results["opencv_dnn"]["status"], "skipped")
            self.assertEqual(results["opencv_dnn"]["skip_reason"], "benchmark_format_not_wired")
            self.assertEqual(results["torch"]["support_status"], "artifact-backed")
            self.assertEqual(results["onnx"]["support_status"], "artifact-backed")
            self.assertEqual(payload["support_summary"]["counts"]["artifact-backed"], 2)
            self.assertEqual(payload["support_summary"]["counts"]["skipped"], 1)
            self.assertEqual(results["torch"]["eval_metrics"]["obb_mAP50"], 1.0)
            self.assertGreater(results["onnx"]["eval_metrics"]["obb_mAP50"], 0.0)
            self.assertEqual(
                json.loads(Path(results["torch"]["artifacts"]["eval"]).read_text(encoding="utf-8"))["kind"],
                "benchmark_obb_eval_report",
            )

    def test_obb_task_rejects_invalid_angle_artifact(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            labels = root / "obb_labels.json"
            pred = root / "obb_bad.json"
            labels.write_text(
                json.dumps(
                    {
                        "classes": ["ship"],
                        "samples": [
                            {
                                "id": "img0",
                                "objects": [
                                    {
                                        "class_id": 0,
                                        "obb": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2, "angle_deg": 0.0},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pred.write_text(
                json.dumps(
                    {
                        "classes": ["ship"],
                        "predictions": [
                            {
                                "id": "img0",
                                "detections": [
                                    {
                                        "class_id": 0,
                                        "score": 0.9,
                                        "obb": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2, "angle_deg": 270.0},
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = root / "benchmark_obb_report.json"
            args = self._args(
                format="torch",
                model=str(pred),
                task="obb",
                data=str(labels),
                output=str(report),
                predictions_output=str(root / "artifacts"),
                eval_output=str(root / "artifacts"),
                parity_output=str(root / "artifacts"),
            )

            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                payload, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            result = payload["results"][0]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["skip_reason"], "obb_artifact_invalid")
            self.assertIn("angle_deg", result["error"])

    def test_segmentation_task_supports_real_artifact_eval(self):
        np = None
        Image = None
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"segmentation benchmark deps unavailable: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "seg_dataset"
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "masks" / "val").mkdir(parents=True)

            image_path = dataset_root / "images" / "val" / "sample.png"
            mask_path = dataset_root / "masks" / "val" / "sample.png"
            image_path.write_bytes(
                b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGNkYPjPgA0wYRUdtBIAy0MBD1Y0SxIAAAAASUVORK5CYII="
                )
            )
            Image.fromarray(np.array([[0, 1], [1, 1]], dtype=np.uint8)).save(mask_path)
            (dataset_root / "dataset.json").write_text(
                json.dumps(
                    {
                        "dataset": "unit_segmentation",
                        "task": "semantic_segmentation",
                        "split": "val",
                        "mode": "manifest",
                        "path_type": "absolute",
                        "ignore_index": 255,
                        "classes": ["background", "fg"],
                        "samples": [{"id": "sample0", "image": str(image_path), "mask": str(mask_path)}],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            ref_mask = root / "pred_torch.png"
            cand_mask = root / "pred_onnx.png"
            Image.fromarray(np.array([[0, 1], [1, 1]], dtype=np.uint8)).save(ref_mask)
            Image.fromarray(np.array([[0, 1], [0, 1]], dtype=np.uint8)).save(cand_mask)
            ref_pred = root / "seg_torch.json"
            cand_pred = root / "seg_onnx.json"
            ref_pred.write_text(json.dumps({"sample0": ref_mask.name}, indent=2), encoding="utf-8")
            cand_pred.write_text(json.dumps({"sample0": cand_mask.name}, indent=2), encoding="utf-8")

            report = root / "benchmark_segmentation_report.json"
            artifact_dir = root / "artifacts"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "segmentation",
                    "--model",
                    str(ref_pred),
                    "--onnx-model",
                    str(cand_pred),
                    "--data",
                    str(dataset_root),
                    "--format",
                    "torch,onnx",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark segmentation artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "segmentation")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "partial")
            self.assertEqual(results["torch"]["support_status"], "artifact-backed")
            self.assertEqual(results["onnx"]["support_status"], "artifact-backed")
            self.assertEqual(payload["support_summary"]["counts"]["artifact-backed"], 2)
            self.assertEqual(results["torch"]["eval_metrics"]["miou"], 1.0)
            self.assertIn("max_mismatch_rate", results["onnx"]["parity"])

    def test_auto_uses_real_sources_for_torchscript_detect_and_artifact_tasks(self):
        args = self._args(latency_source="auto")
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="torch", task_label="detect"),
            "dataset_pass_wall_time",
        )
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="torchscript", task_label="detect"),
            "dataset_pass_wall_time",
        )
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="openvino", task_label="detect"),
            "dataset_pass_wall_time",
        )
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="torchscript", task_label="keypoints"),
            "artifact_eval",
        )
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="openvino", task_label="keypoints"),
            "artifact_eval",
        )

    def test_depth_task_supports_real_artifact_eval(self):
        np = None
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable for depth benchmark test: {exc}")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            pred_torch = gt.copy()
            pred_onnx = gt * 1.01
            gt_path = root / "gt.npy"
            torch_path = root / "torch_depth.npy"
            onnx_path = root / "onnx_depth.npy"
            report = root / "benchmark_report.json"
            artifact_dir = root / "artifacts"
            np.save(gt_path, gt)
            np.save(torch_path, pred_torch)
            np.save(onnx_path, pred_onnx)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "depth",
                    "--model",
                    str(torch_path),
                    "--onnx-model",
                    str(onnx_path),
                    "--data",
                    str(gt_path),
                    "--format",
                    "torch,onnx",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark depth artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "depth")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "ok")
            self.assertEqual(results["torch"]["eval_metrics"]["abs_rel"], 0.0)
            self.assertIn("mae", results["onnx"]["parity"]["metrics"])
            self.assertEqual(payload["parity_summary"]["reference_backend"], "torch")
            self.assertEqual(payload["parity_summary"]["comparisons"], 1)
            self.assertEqual(payload["parity_summary"]["ok"], 1)
            self.assertEqual(payload["parity_summary"]["drift"], 0)

    def test_keypoints_task_supports_real_artifact_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "keypoints_dataset"
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "labels" / "val").mkdir(parents=True)
            image_path = dataset_root / "images" / "val" / "sample.png"
            image_path.write_bytes(
                b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGNkYPjPgA0wYRUdtBIAy0MBD1Y0SxIAAAAASUVORK5CYII="
                )
            )
            (dataset_root / "labels" / "val" / "sample.txt").write_text(
                "0 0.5 0.5 0.5 0.5 0.25 0.25 2 0.75 0.25 2\n",
                encoding="utf-8",
            )

            ref_pred = root / "kp_torch.json"
            cand_pred = root / "kp_onnx.json"
            base_entry = {
                "image": str(image_path),
                "image_size": [4, 4],
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5},
                        "keypoints": [
                            {"x": 0.25, "y": 0.25, "v": 2},
                            {"x": 0.75, "y": 0.25, "v": 2},
                        ],
                    }
                ],
            }
            ref_pred.write_text(json.dumps({"predictions": [base_entry]}, indent=2), encoding="utf-8")
            cand_entry = json.loads(json.dumps(base_entry))
            cand_entry["detections"][0]["keypoints"][0]["x"] = 0.25005
            cand_pred.write_text(json.dumps({"predictions": [cand_entry]}, indent=2), encoding="utf-8")
            report = root / "benchmark_keypoints_report.json"
            artifact_dir = root / "artifacts"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "keypoints",
                    "--model",
                    str(ref_pred),
                    "--onnx-model",
                    str(cand_pred),
                    "--data",
                    str(dataset_root),
                    "--format",
                    "torch,onnx",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark keypoints artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "keypoints")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "ok")
            self.assertEqual(results["torch"]["eval_metrics"]["pck"], 1.0)
            self.assertIn("kp_abs_max", results["onnx"]["parity"])

    def test_pose6d_task_supports_real_artifact_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "pose_dataset"
            (dataset_root / "images" / "val").mkdir(parents=True)
            (dataset_root / "labels" / "val").mkdir(parents=True)
            image_path = dataset_root / "images" / "val" / "sample.png"
            image_path.write_bytes(
                b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGNkYPjPgA0wYRUdtBIAy0MBD1Y0SxIAAAAASUVORK5CYII="
                )
            )
            (dataset_root / "labels" / "val" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (dataset_root / "labels" / "val" / "sample.json").write_text(
                json.dumps(
                    {
                        "R_gt": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                        "t_gt": [[0.0, 0.0, 1.0]],
                        "cad_points": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                    }
                ),
                encoding="utf-8",
            )

            ref_pred = root / "pose_torch.json"
            cand_pred = root / "pose_onnx.json"
            base_entry = {
                "image": str(image_path),
                "image_size": [4, 4],
                "detections": [
                    {
                        "class_id": 0,
                        "score": 1.0,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5},
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t_xyz": [0.0, 0.0, 1.0],
                    }
                ],
            }
            ref_pred.write_text(json.dumps({"predictions": [base_entry]}, indent=2), encoding="utf-8")
            cand_entry = json.loads(json.dumps(base_entry))
            cand_entry["detections"][0]["t_xyz"] = [0.0, 0.0, 1.00005]
            cand_pred.write_text(json.dumps({"predictions": [cand_entry]}, indent=2), encoding="utf-8")
            report = root / "benchmark_pose_report.json"
            artifact_dir = root / "artifacts"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yolozu",
                    "benchmark",
                    "--task",
                    "pose6d",
                    "--model",
                    str(ref_pred),
                    "--onnx-model",
                    str(cand_pred),
                    "--data",
                    str(dataset_root),
                    "--format",
                    "torch,onnx",
                    "--latency-source",
                    "artifact_eval",
                    "--predictions-output",
                    str(artifact_dir),
                    "--eval-output",
                    str(artifact_dir),
                    "--parity-output",
                    str(artifact_dir),
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
                self.fail(f"yolozu benchmark pose6d artifact_eval failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("task"), "pose6d")
            by_format = payload.get("execution_semantics", {}).get("by_format", {})
            self.assertEqual(by_format["torch"]["execution_mode"], "real_artifact_eval")
            self.assertEqual(by_format["onnx"]["execution_mode"], "real_artifact_eval")
            results = {item["format"]: item for item in payload.get("results") or []}
            self.assertEqual(results["torch"]["status"], "ok")
            self.assertEqual(results["onnx"]["status"], "ok")
            self.assertEqual(results["torch"]["eval_metrics"]["pose_success"], 1.0)
            self.assertIn("trans_l2_max", results["onnx"]["parity"])

    def test_onnx_rejects_half_flag_early(self):
        args = self._args(format="onnx", onnx_model="exports/example.onnx", half=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "onnxruntime"):
            with self.assertRaisesRegex(ValueError, r"--half not supported for --format onnx"):
                benchmark_mode.run_benchmark_mode(args)

    def test_artifact_eval_rejects_each_inert_backend_flag_for_all_tasks(self):
        cases = (
            ("--half", {"half": True}),
            ("--batch", {"batch": 2}),
            ("--nms", {"nms": True}),
        )
        for task_label in sorted(benchmark_mode.ARTIFACT_EVAL_TASKS):
            for flag, overrides in cases:
                with self.subTest(task=task_label, flag=flag):
                    args = self._args(
                        format="torch",
                        task=task_label,
                        latency_source="artifact_eval",
                        **overrides,
                    )
                    with self.assertRaises(ValueError) as raised:
                        benchmark_mode.run_benchmark_mode(args)
                    message = str(raised.exception)
                    self.assertIn(flag, message)
                    self.assertIn(f"--task {task_label}", message)
                    self.assertIn("--latency-source artifact_eval", message)
                    self.assertIn("--format torch", message)
                    self.assertIn("consumes prepared artifacts", message)
                    self.assertFalse(Path(args.output).exists())

    def test_auto_resolves_artifact_tasks_before_flag_validation(self):
        for task_label in sorted(benchmark_mode.ARTIFACT_EVAL_TASKS):
            with self.subTest(task=task_label):
                args = self._args(
                    format="torch",
                    task=task_label,
                    latency_source="auto",
                    half=True,
                )
                with self.assertRaises(ValueError) as raised:
                    benchmark_mode.run_benchmark_mode(args)
                message = str(raised.exception)
                self.assertIn("--half", message)
                self.assertIn(f"--task {task_label}", message)
                self.assertIn("--latency-source auto (effective: artifact_eval)", message)
                self.assertFalse(Path(args.output).exists())

    def test_artifact_tasks_reject_dataset_pass_through_both_cli_surfaces(self):
        surfaces = {
            "canonical": [sys.executable, "-m", "yolozu", "benchmark"],
            "standalone": [sys.executable, str(self.repo_root / "tools" / "benchmark_model.py")],
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            for surface, prefix in surfaces.items():
                for task_label in sorted(benchmark_mode.ARTIFACT_EVAL_TASKS):
                    with self.subTest(surface=surface, task=task_label):
                        report = root / f"{surface}_{task_label}_dataset_pass.json"
                        proc = subprocess.run(
                            [
                                *prefix,
                                "--model",
                                "reports/prepared_artifact.json",
                                "--data",
                                "data/smoke",
                                "--format",
                                "torch",
                                "--task",
                                task_label,
                                "--latency-source",
                                "dataset_pass_wall_time",
                                "--no-half",
                                "--batch",
                                "1",
                                "--no-nms",
                                "--output",
                                str(report),
                            ],
                            cwd=str(self.repo_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                            text=True,
                        )
                        self.assertNotEqual(proc.returncode, 0)
                        output = f"{proc.stdout}\n{proc.stderr}"
                        self.assertIn(f"--task {task_label} uses artifact-backed evaluation", output)
                        self.assertIn("use --latency-source auto or artifact_eval", output)
                        self.assertFalse(report.exists(), "source validation must fail before writing the report")

    def test_artifact_eval_rejects_inert_flags_through_both_cli_surfaces(self):
        surfaces = {
            "canonical": [sys.executable, "-m", "yolozu", "benchmark"],
            "standalone": [sys.executable, str(self.repo_root / "tools" / "benchmark_model.py")],
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            for surface, prefix in surfaces.items():
                for task_label in sorted(benchmark_mode.ARTIFACT_EVAL_TASKS):
                    with self.subTest(surface=surface, task=task_label):
                        report = root / f"{surface}_{task_label}_rejected.json"
                        proc = subprocess.run(
                            [
                                *prefix,
                                "--model",
                                "reports/prepared_artifact.json",
                                "--data",
                                "data/smoke",
                                "--format",
                                "torch",
                                "--task",
                                task_label,
                                "--latency-source",
                                "artifact_eval",
                                "--half",
                                "--batch",
                                "2",
                                "--nms",
                                "--output",
                                str(report),
                            ],
                            cwd=str(self.repo_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                            text=True,
                        )
                        self.assertNotEqual(proc.returncode, 0)
                        output = f"{proc.stdout}\n{proc.stderr}"
                        for flag in ("--half", "--batch", "--nms"):
                            self.assertIn(flag, output)
                        self.assertIn(f"--task {task_label}", output)
                        self.assertIn("--latency-source artifact_eval", output)
                        self.assertIn("--format torch", output)
                        self.assertIn("consumes prepared artifacts", output)
                        self.assertFalse(report.exists(), "validation must fail before writing the report")

    def test_artifact_eval_accepts_explicit_defaults_through_both_cli_surfaces(self):
        surfaces = {
            "canonical": [sys.executable, "-m", "yolozu", "benchmark"],
            "standalone": [sys.executable, str(self.repo_root / "tools" / "benchmark_model.py")],
        }
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            for surface, prefix in surfaces.items():
                for task_label in sorted(benchmark_mode.ARTIFACT_EVAL_TASKS):
                    with self.subTest(surface=surface, task=task_label):
                        report = root / f"{surface}_{task_label}_defaults.json"
                        proc = subprocess.run(
                            [
                                *prefix,
                                "--model",
                                "reports/prepared_artifact.json",
                                "--data",
                                "data/smoke",
                                "--format",
                                "torch",
                                "--task",
                                task_label,
                                "--latency-source",
                                "artifact_eval",
                                "--no-half",
                                "--batch",
                                "1",
                                "--no-nms",
                                "--dry-run",
                                "--output",
                                str(report),
                            ],
                            cwd=str(self.repo_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                            text=True,
                        )
                        if proc.returncode != 0:
                            self.fail(
                                f"{surface} rejected artifact_eval defaults for {task_label}:\n"
                                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                            )
                        payload = json.loads(report.read_text(encoding="utf-8"))
                        self.assertEqual(payload["task"], task_label)
                        self.assertEqual(payload["validation_summary"]["nondefault_flags"], {})

    def test_engine_rejects_workspace_flag_early(self):
        args = self._args(format="engine", engine_model="exports/example.plan", workspace=8.0, device="cuda:0")
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name in {"tensorrt", "cuda"}):
            with self.assertRaisesRegex(ValueError, r"--workspace not supported for --format engine"):
                benchmark_mode.run_benchmark_mode(args)

    def test_torchscript_rejects_dynamic_flag_early(self):
        args = self._args(format="torchscript", model="exports/example.torchscript", dynamic=True)
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
            with self.assertRaisesRegex(ValueError, r"--dynamic not supported for --format torchscript"):
                benchmark_mode.run_benchmark_mode(args)

    def test_validation_summary_records_strict_format_policy(self):
        args = self._args(format="torch,onnx,torchscript", task="obb", strict=True)
        with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
            report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 2)
        summary = report["validation_summary"]
        self.assertEqual(summary["task"], "obb")
        self.assertEqual(summary["requested_formats"], ["torch", "onnx", "torchscript"])
        self.assertTrue(summary["strict"])
        self.assertEqual(summary["bad_flag_policy"], "fail_early")
        self.assertEqual(summary["unsupported_task_policy"], "report_skipped")
        self.assertTrue(summary["openvino_applicable"])
        for fmt in ("torch", "onnx", "torchscript"):
            item = summary["by_format"][fmt]
            self.assertEqual(item["execution_mode"], "real_artifact_eval")
            self.assertEqual(item["missing_runtime_policy"], "report_skipped")
            self.assertEqual(item["missing_artifact_policy"], "report_skipped")
            self.assertEqual(item["unsupported_nondefault_flags"], [])
            self.assertEqual(item["supported_nondefault_flags"], [])
            self.assertIn("consumes prepared artifacts", item["flag_applicability_reason"])
        self.assertEqual(
            summary["by_format"]["torch"]["format_supported_nondefault_flags"],
            ["batch", "half", "nms"],
        )

    def test_validation_summary_records_missing_artifact_skip_policy(self):
        args = self._args(format="onnx", model="runs/foo/model.pt", latency_source="dataset_pass_wall_time")
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "onnxruntime"):
            with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                report, code = benchmark_mode.run_benchmark_mode(args)

        self.assertEqual(code, 0)
        self.assertEqual(report["results"][0]["skip_reason"], "model_artifact_required")
        policy = report["validation_summary"]["by_format"]["onnx"]
        self.assertEqual(policy["execution_mode"], "real_backend_eval")
        self.assertEqual(policy["missing_artifact_policy"], "report_skipped")
        self.assertEqual(policy["missing_runtime_policy"], "report_skipped")

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

    def test_module_cli_benchmark_help_matches_real_surface(self):
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "yolozu", "benchmark", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"python -m yolozu benchmark --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--torchscript-model", proc.stdout)
        self.assertIn("--openvino-model", proc.stdout)
        self.assertIn("--segmentation-parity-mismatch-atol", proc.stdout)
        self.assertIn("--parity-reference-backend", proc.stdout)
        self.assertIn("--protocol", proc.stdout)
        self.assertIn("torchscript", proc.stdout)
        self.assertIn("openvino", proc.stdout)
        normalized_help = " ".join(proc.stdout.split())
        self.assertIn("Must remain disabled when the effective latency source is artifact_eval", normalized_help)
        self.assertIn("Must remain 1 when the effective latency source is artifact_eval", normalized_help)

    def test_short_and_long_help_work_on_both_benchmark_surfaces(self):
        commands = (
            [sys.executable, "-m", "yolozu", "benchmark"],
            [sys.executable, str(self.repo_root / "tools" / "benchmark_model.py")],
        )
        for command in commands:
            for help_flag in ("-h", "--help"):
                with self.subTest(command=command, help_flag=help_flag):
                    proc = subprocess.run(
                        [*command, help_flag],
                        cwd=str(self.repo_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertIn("--openvino-model", proc.stdout)
                    self.assertIn("openvino", proc.stdout)
                    self.assertIn(
                        "OpenVINO requires supplied artifacts and an available runtime",
                        " ".join(proc.stdout.split()),
                    )
                    self.assertIn(
                        "Return exit code 2 if any requested format is skipped or fails",
                        " ".join(proc.stdout.split()),
                    )

    def test_canonical_openvino_parser_surface_matches_standalone(self):
        required = ["--model", "runs/example/model.pt", "--data", "data/smoke"]
        standalone_parser = benchmark_mode.build_parser()
        self.assertEqual(
            cli_entry.BENCHMARK_PARITY_REFERENCE_BACKENDS,
            benchmark_mode.PARITY_REFERENCE_BACKENDS,
        )
        standalone_defaults = standalone_parser.parse_args(required)
        captured: list[object] = []

        with mock.patch.object(cli_entry, "_cmd_benchmark", side_effect=lambda args: captured.append(args) or 0):
            code = cli_entry.main(["benchmark", *required])

        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        canonical_defaults = captured.pop()
        canonical_default_values = vars(canonical_defaults).copy()
        canonical_default_values.pop("command", None)
        self.assertEqual(canonical_default_values, vars(standalone_defaults))
        self.assertEqual(canonical_defaults.openvino_model, standalone_defaults.openvino_model)
        self.assertIsNone(canonical_defaults.openvino_model)
        self.assertEqual(
            canonical_defaults.parity_reference_backend,
            standalone_defaults.parity_reference_backend,
        )
        self.assertEqual(canonical_defaults.parity_reference_backend, "auto")

        openvino_args = [
            *required,
            "--format",
            "torch,openvino",
            "--openvino-model",
            "exports/example.xml",
            "--parity-reference-backend",
            "openvino",
            "--dry-run",
        ]
        standalone_openvino = standalone_parser.parse_args(openvino_args)
        with mock.patch.object(cli_entry, "_cmd_benchmark", side_effect=lambda args: captured.append(args) or 0):
            code = cli_entry.main(["benchmark", *openvino_args])

        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        canonical_openvino = captured.pop()
        for attribute in (
            "format",
            "openvino_model",
            "parity_reference_backend",
            "dry_run",
        ):
            self.assertEqual(
                getattr(canonical_openvino, attribute),
                getattr(standalone_openvino, attribute),
            )
        self.assertEqual(canonical_openvino.openvino_model, "exports/example.xml")
        self.assertEqual(canonical_openvino.parity_reference_backend, "openvino")

        standalone_help = standalone_parser.format_help()
        for token in ("--openvino-model", "--parity-reference-backend", "openvino"):
            self.assertIn(token, standalone_help)

    def test_module_cli_routes_openvino_override_in_dry_run(self):
        repo_root = self.repo_root
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
                    "runs/example/model.pt",
                    "--openvino-model",
                    "exports/example.xml",
                    "--data",
                    "data/smoke",
                    "--format",
                    "torch,openvino",
                    "--parity-reference-backend",
                    "openvino",
                    "--dry-run",
                    "--predictions-output",
                    str(root / "predictions_{format}.json"),
                    "--eval-output",
                    str(root / "eval_{format}.json"),
                    "--parity-output",
                    str(root / "parity_{format}.json"),
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
                self.fail(f"canonical OpenVINO dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], ["torch", "openvino"])
            openvino_result = next(item for item in payload["results"] if item["format"] == "openvino")
            self.assertIn(openvino_result["status"], {"dry_run", "skipped"})
            if openvino_result["status"] == "skipped":
                self.assertIn(
                    openvino_result["skip_reason"],
                    {"missing_runtime_dependency", "model_artifact_required"},
                )
            self.assertTrue(Path(openvino_result["artifacts"]["export_settings"]).is_file())

    def _args(self, **overrides):
        root = Path(__file__).resolve().parents[1]
        base = dict(
            model="runs/foo/model.pt",
            torch_model=None,
            onnx_model=None,
            engine_model=None,
            torchscript_model=None,
            openvino_model=None,
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
            parity_reference_backend="auto",
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
                raise AssertionError(f"unexpected subprocess command: {cmd}")

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
            self.assertEqual(result["support_status"], "real")
            self.assertEqual(report["support_summary"]["counts"]["real"], 1)
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
        self.assertEqual(result["support_status"], "skipped")
        self.assertEqual(result["support_reason"], "model_artifact_required")
        self.assertTrue(result["runtime"]["available"])

    def test_torchscript_supported_runtime_uses_real_orchestration(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_path = root / "benchmark_report.json"

            def fake_run(cmd, cwd, capture_output, text, check):
                cmd = [str(x) for x in cmd]
                if cmd[1].endswith("export_predictions_torchscript.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {
                        "predictions": [
                            {"image": "images/val/000001.jpg", "detections": []},
                            {"image": "images/val/000002.jpg", "detections": []},
                        ],
                        "meta": {"adapter": "torchscript"},
                    }
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                if cmd[1].endswith("eval_suite.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {"metrics": {"bbox_mAP50": 0.37}}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                raise AssertionError(f"unexpected subprocess command: {cmd}")

            args = self._args(
                output=str(report_path),
                data=str(repo_root / "data" / "smoke"),
                format="torchscript",
                model="runs/foo/model.torchscript",
                latency_source="auto",
            )
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "torch"):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            result = report["results"][0]
            self.assertEqual(result["format"], "torchscript")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["latency_source"], "dataset_pass_wall_time")
            self.assertEqual(result["execution_semantics"]["execution_mode"], "real_backend_eval")
            self.assertEqual(result["eval_metrics"]["bbox_mAP50"], 0.37)
            self.assertTrue(result["artifacts"]["predictions"].endswith("predictions_torchscript.json"))

    def test_openvino_supported_runtime_uses_real_orchestration(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            report_path = root / "benchmark_report.json"

            def fake_run(cmd, cwd, capture_output, text, check):
                cmd = [str(x) for x in cmd]
                if cmd[1].endswith("export_predictions_openvino.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {
                        "predictions": [
                            {"image": "images/val/000001.jpg", "detections": []},
                            {"image": "images/val/000002.jpg", "detections": []},
                        ],
                        "meta": {"adapter": "openvino"},
                    }
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                if cmd[1].endswith("eval_suite.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {"metrics": {"bbox_mAP50": 0.39}}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                raise AssertionError(f"unexpected subprocess command: {cmd}")

            args = self._args(
                output=str(report_path),
                data=str(repo_root / "data" / "smoke"),
                format="openvino",
                model="runs/foo/model.xml",
                latency_source="auto",
            )
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name == "openvino"):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            result = report["results"][0]
            self.assertEqual(result["format"], "openvino")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["support_status"], "real")
            self.assertEqual(result["runtime"]["available"], True)
            self.assertEqual(result["latency_source"], "dataset_pass_wall_time")
            self.assertEqual(result["execution_semantics"]["execution_mode"], "real_backend_eval")
            self.assertEqual(result["eval_metrics"]["bbox_mAP50"], 0.39)
            self.assertTrue(result["artifacts"]["predictions"].endswith("predictions_openvino.json"))

    def test_detect_four_format_request_writes_real_parity_and_skips_engine(self):
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
                if cmd[1].endswith("export_predictions_torchscript.py"):
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
                raise AssertionError(f"unexpected subprocess command: {cmd}")

            def fake_module_available(name):
                return name in {"ultralytics", "onnxruntime", "torch"}

            args = self._args(
                output=str(report_path),
                data=str(repo_root / "data" / "smoke"),
                format="torch,onnx,engine,torchscript",
                onnx_model="exports/example.onnx",
                torchscript_model="exports/example.torchscript",
            )
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=fake_module_available):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "partial")
            results = {item["format"]: item for item in report["results"]}
            self.assertEqual(results["torch"]["parity"]["reference_backend"], "torch")
            self.assertEqual(results["torch"]["parity"]["candidate_backends"], ["onnx", "torchscript"])
            self.assertEqual(results["engine"]["status"], "skipped")
            self.assertEqual(results["engine"]["support_status"], "skipped")
            self.assertIn(results["engine"]["support_reason"], {"gpu_required", "platform_not_supported", "missing_runtime_dependency"})
            self.assertTrue(Path(results["torch"]["artifacts"]["parity"]).is_file())
            torch_parity_payload = json.loads(Path(results["torch"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(torch_parity_payload["kind"], "benchmark_parity_reference")

            self.assertTrue(Path(results["onnx"]["artifacts"]["parity"]).is_file())
            onnx_parity_payload = json.loads(Path(results["onnx"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(onnx_parity_payload["kind"], "benchmark_parity_report")
            self.assertEqual(onnx_parity_payload["reference_backend"], "torch")
            self.assertEqual(onnx_parity_payload["candidate_backend"], "onnx")
            self.assertTrue(onnx_parity_payload["summary"]["ok"])
            torchscript_parity_payload = json.loads(Path(results["torchscript"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(torchscript_parity_payload["candidate_backend"], "torchscript")
            self.assertTrue(torchscript_parity_payload["summary"]["ok"])
            self.assertEqual(report["parity_summary"]["reference_backend"], "torch")
            self.assertEqual(report["parity_summary"]["comparisons"], 2)
            self.assertEqual(report["parity_summary"]["ok"], 2)
            self.assertEqual(report["parity_summary"]["skipped"], 1)
            self.assertEqual(report["parity_summary"]["by_format"]["engine"]["status"], "skipped")

    def test_detect_parity_can_use_explicit_onnx_reference_backend(self):
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
                if cmd[1].endswith("eval_suite.py"):
                    out = Path(cmd[cmd.index("--output") + 1])
                    payload = {"metrics": {"bbox_mAP50": 0.42}}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(cmd, 0, stdout=str(out), stderr="")
                raise AssertionError(f"unexpected subprocess command: {cmd}")

            def fake_module_available(name):
                return name in {"ultralytics", "onnxruntime"}

            args = self._args(
                output=str(report_path),
                data=str(repo_root / "data" / "smoke"),
                format="torch,onnx",
                onnx_model="exports/example.onnx",
                parity_reference_backend="onnx",
            )
            with mock.patch.object(benchmark_mode, "_module_available", side_effect=fake_module_available):
                with mock.patch.object(benchmark_mode, "_git_head", return_value="deadbeef"):
                    with mock.patch.object(benchmark_mode.subprocess, "run", side_effect=fake_run):
                        report, code = benchmark_mode.run_benchmark_mode(args)

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            results = {item["format"]: item for item in report["results"]}
            self.assertEqual(results["onnx"]["parity"]["reference_backend"], "onnx")
            self.assertEqual(results["onnx"]["parity"]["candidate_backends"], ["torch"])
            onnx_parity_payload = json.loads(Path(results["onnx"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(onnx_parity_payload["kind"], "benchmark_parity_reference")
            torch_parity_payload = json.loads(Path(results["torch"]["artifacts"]["parity"]).read_text(encoding="utf-8"))
            self.assertEqual(torch_parity_payload["reference_backend"], "onnx")
            self.assertEqual(torch_parity_payload["candidate_backend"], "torch")
            self.assertTrue(torch_parity_payload["summary"]["ok"])
            self.assertEqual(report["parity_summary"]["reference_backend"], "onnx")
            self.assertEqual(report["parity_summary"]["comparisons"], 1)
            self.assertEqual(report["parity_summary"]["ok"], 1)


if __name__ == "__main__":
    main()
