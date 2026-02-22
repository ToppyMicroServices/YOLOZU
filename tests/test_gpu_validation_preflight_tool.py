import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGpuValidationPreflightTool(unittest.TestCase):
    def test_writes_split_report_with_expected_structure(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "gpu_validation_preflight.py"
        self.assertTrue(script.is_file(), "missing tools/gpu_validation_preflight.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out_path = Path(td) / "gpu_validation_preflight.json"
            proc = subprocess.run(
                [sys.executable, str(script), "--output", str(out_path)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"gpu_validation_preflight.py failed:\n{proc.stdout}\n{proc.stderr}")

            self.assertTrue(out_path.is_file(), "missing preflight report output")
            payload = json.loads(out_path.read_text(encoding="utf-8"))

            self.assertEqual(payload.get("kind"), "yolozu_gpu_validation_preflight")
            self.assertEqual(payload.get("schema_version"), 1)
            self.assertEqual(payload.get("issue"), "YOLOZU-zisn")

            summary = payload.get("summary") or {}
            for key in (
                "local_executable_steps",
                "local_preflight_ready_steps",
                "gpu_execution_required_steps",
                "gpu_execution_optional_steps",
                "ready_for_gpu_steps",
                "blocked_local_steps",
                "needs_gpu_runtime_steps",
            ):
                self.assertIn(key, summary)
                self.assertTrue(isinstance(summary.get(key), list), f"summary.{key} must be a list")

            expected_gpu_required = {
                "train_rtdetr_pose_amp_ddp_resume",
                "trt_build_export_eval_latency",
                "safe_ttt_ctta_stability",
                "opencv_dnn_backend_parity",
            }
            self.assertEqual(set(summary.get("gpu_execution_required_steps") or []), expected_gpu_required)

            steps = payload.get("steps") or []
            self.assertEqual(len(steps), 6)

            step_ids = {str(step.get("id")) for step in steps}
            self.assertEqual(
                step_ids,
                {
                    "train_rtdetr_pose_amp_ddp_resume",
                    "onnx_export_ort_parity",
                    "trt_build_export_eval_latency",
                    "safe_ttt_ctta_stability",
                    "opencv_dnn_backend_parity",
                    "gpu_doctor_matrix",
                },
            )

            allowed_status = {"ready_local", "ready_for_gpu", "needs_gpu_runtime", "blocked_local"}
            for step in steps:
                self.assertIn(step.get("status"), allowed_status)
                self.assertTrue(isinstance(step.get("local_checks"), list))
                self.assertTrue(isinstance(step.get("gpu_checks"), list))
                commands = step.get("commands") or {}
                self.assertTrue(isinstance(commands.get("local"), list))
                self.assertTrue(isinstance(commands.get("gpu"), list))
                self.assertTrue(isinstance(step.get("definition_of_done"), list))

            snapshot = payload.get("doctor_snapshot") or {}
            self.assertIn("runtime_capabilities", snapshot)


if __name__ == "__main__":
    unittest.main()
