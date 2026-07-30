import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestSupportExternalTrainingTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        self.assertTrue(script.is_file(), "missing tools/support_external_training.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"support_external_training --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("train-yolox", proc.stdout)
        self.assertIn("train-detectron2", proc.stdout)
        self.assertIn("train-mmdetection", proc.stdout)
        self.assertIn("train-mmpose", proc.stdout)
        self.assertIn("train-mmseg", proc.stdout)
        self.assertIn("train-tao", proc.stdout)
        self.assertIn("train-ultralytics", proc.stdout)
        self.assertIn("train-hf-detr", proc.stdout)

    def test_tao_dry_run_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_tao.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-tao",
                    "--config",
                    "configs/examples/finetune_external/tao_finetune_smoke.yaml",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--task-family",
                    "bbox",
                    "--dry-run",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"support_external_training train-tao failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["backend"]["backend_id"], "tao")
            self.assertIn("resume", payload["handoff_contracts"])
            self.assertIn("reports/resume_handoff.json", payload["run_output_contract"]["stable_artifacts"])

    def test_tao_missing_runtime_writes_machine_readable_failure(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_tao.json"
            work_dir = root / "tao_work"
            env = dict(os.environ)
            env["PATH"] = ""
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-tao",
                    "--preset",
                    "none",
                    "--config",
                    "configs/examples/finetune_external/tao_finetune_smoke.yaml",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--task-family",
                    "bbox",
                    "--work-dir",
                    str(work_dir),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_code"], "E_EXTERNAL_RUNTIME_MISSING")
            self.assertEqual(payload["execution_status"]["state"], "runtime_failed")
            self.assertFalse(payload["execution_status"]["real_training_executed"])
            self.assertIn("FileNotFoundError", payload["runtime_error"])
            self.assertEqual(payload["process"]["returncode"], 127)

    def test_mmdetection_auto_infers_coco_instances_as_bbox(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_mmdetection.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-mmdetection",
                    "--preset",
                    "none",
                    "--config",
                    "configs/examples/finetune_external/mmdetection_finetune_smoke.py",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--dry-run",
                    "--work-dir",
                    str(root / "work"),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(
                    "support_external_training train-mmdetection failed:"
                    f"\n{proc.stdout}\n{proc.stderr}"
                )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_family"], "bbox")
            self.assertEqual(payload["canonical_train_config"]["task"], "bbox")
            self.assertEqual(
                payload["handoff_contracts"]["export"]["output_contract"]["type"],
                "predictions interface contract",
            )

    def test_yolox_dry_run_dod_gate(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_yolox.json"
            work_dir = root / "yolox_work"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-yolox",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--exp",
                    "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                    "--dry-run",
                    "--work-dir",
                    str(work_dir),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"support_external_training train-yolox DoD dry-run failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("format"), "yolozu_training_run_summary_v1")
            self.assertTrue(bool(payload.get("dry_run")))
            self.assertFalse(bool(payload.get("training_executed")))
            execution_status = payload.get("execution_status") or {}
            self.assertEqual(execution_status.get("state"), "dry_run_handoff")
            self.assertFalse(bool(execution_status.get("real_training_executed")))
            self.assertTrue(bool(execution_status.get("handoff_ready")))
            self.assertTrue(bool(execution_status.get("requires_external_train_script")))

            artifact_plan = payload.get("artifact_plan") or {}
            self.assertEqual(artifact_plan.get("format"), "yolozu_external_training_artifact_plan_v1")
            self.assertEqual(artifact_plan.get("lane"), "yolox")
            self.assertIn("expected_outputs", artifact_plan.get("dry_run_validates") or [])
            runtime_boundary = artifact_plan.get("runtime_license_boundary") or {}
            self.assertEqual(runtime_boundary.get("repo_code"), "Apache-2.0")
            self.assertFalse(bool(runtime_boundary.get("vendored")))

            expected_outputs = artifact_plan.get("expected_outputs") or {}
            for key in ("predictions_json", "eval_report", "parity_report", "training_summary"):
                self.assertTrue(expected_outputs.get(key), key)

            next_commands = artifact_plan.get("next_commands") or {}
            self.assertIn("tools/export_predictions_yolox.py", next_commands.get("export", ""))
            self.assertIn("yolox_predictions.json", next_commands.get("eval", ""))
            self.assertIn("yolox_predictions.json", next_commands.get("parity", ""))

            next_steps_by_stage = {
                str(item.get("stage")): item for item in payload.get("next_steps") or [] if isinstance(item, dict)
            }
            for stage in ("resume", "export", "eval", "parity"):
                self.assertIn(stage, next_steps_by_stage)
                self.assertTrue(next_steps_by_stage[stage].get("command"))
                self.assertIsInstance(next_steps_by_stage[stage].get("input_contract"), dict)
                self.assertIsInstance(next_steps_by_stage[stage].get("output_contract"), dict)

            for filename in ("training_summary.json", "resume_handoff.json", "export_handoff.json", "eval_handoff.json", "parity_handoff.json"):
                self.assertTrue((work_dir / "reports" / filename).is_file(), filename)

            execution_payload = json.loads((work_dir / "reports" / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual((execution_payload.get("execution_status") or {}).get("state"), "dry_run_handoff")

    def test_yolox_non_dry_without_train_script_reports_status(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_yolox_non_dry.json"
            work_dir = root / "yolox_work"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-yolox",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--exp",
                    "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                    "--work-dir",
                    str(work_dir),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(out.is_file())

            payload = json.loads(out.read_text(encoding="utf-8"))
            execution_status = payload.get("execution_status") or {}
            self.assertEqual(execution_status.get("state"), "requires_external_train_script")
            self.assertTrue(bool(execution_status.get("handoff_ready")))
            self.assertFalse(bool(execution_status.get("real_training_executed")))
            self.assertIn("--train-script", str(execution_status.get("skip_reason")))

            execution_payload = json.loads((work_dir / "reports" / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual((execution_payload.get("execution_status") or {}).get("state"), "requires_external_train_script")

    def test_yolox_zero_exit_traceback_is_not_training_success(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            launcher = root / "swallowed_traceback.py"
            launcher.write_text(
                "import sys\n"
                "print('Traceback (most recent call last):', file=sys.stderr)\n"
                "print('RuntimeError: swallowed', file=sys.stderr)\n",
                encoding="utf-8",
            )
            out = root / "train_yolox.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-yolox",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--exp",
                    "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                    "--train-script",
                    str(launcher),
                    "--work-dir",
                    str(root / "work"),
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(payload["training_executed"])
            self.assertFalse(payload["execution_status"]["real_training_executed"])
            self.assertIn("exited zero", payload["runtime_error"])
            usage = payload["process"]["resource_usage"]
            self.assertGreaterEqual(usage["wall_seconds"], 0.0)
            self.assertGreaterEqual(usage["child_user_cpu_seconds"], 0.0)
            self.assertGreaterEqual(usage["child_peak_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
