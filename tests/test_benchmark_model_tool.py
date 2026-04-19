import json
import subprocess
import sys
import tempfile
from base64 import b64decode
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, mock

from yolozu.eval import benchmark_mode


class TestBenchmarkModelTool(TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self._root_artifacts = [
            self.repo_root / "tmp_benchmark_report.json",
            self.repo_root / "export_settings_onnx.json",
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
        self.assertEqual(report["execution_semantics"]["by_format"]["torchscript"]["execution_mode"], "dry_run_planning")

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

    def test_auto_prefers_artifact_eval_for_segmentation_keypoints_depth_and_pose6d(self):
        args = self._args(latency_source="auto")
        for task_label in ("segmentation", "keypoints", "depth", "pose6d"):
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
            self.assertEqual(results["torch"]["eval_metrics"]["miou"], 1.0)
            self.assertIn("max_mismatch_rate", results["onnx"]["parity"])

    def test_auto_keeps_detect_on_dataset_pass_and_torchscript_on_synthetic(self):
        args = self._args(latency_source="auto")
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="torch", task_label="detect"),
            "dataset_pass_wall_time",
        )
        self.assertEqual(
            benchmark_mode._selected_benchmark_source(args, fmt="torchscript", task_label="keypoints"),
            "synthetic_step",
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

    def test_engine_rejects_workspace_flag_early(self):
        args = self._args(format="engine", engine_model="exports/example.plan", workspace=8.0, device="cuda:0")
        with mock.patch.object(benchmark_mode, "_module_available", side_effect=lambda name: name in {"tensorrt", "cuda"}):
            with self.assertRaisesRegex(ValueError, r"--workspace not supported for --format engine"):
                benchmark_mode.run_benchmark_mode(args)

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
                raise AssertionError(f"unexpected subprocess command: {cmd}")

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


if __name__ == "__main__":
    main()
