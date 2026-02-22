import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "ci" / "summarize_gpu_ngc_run.py"
    spec = importlib.util.spec_from_file_location("summarize_gpu_ngc_run", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load summarize_gpu_ngc_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSummarizeGpuNgcRunTool(unittest.TestCase):
    def setUp(self):
        self.mod = _load_tool()

    def _base_args(self, out_dir: Path) -> list[str]:
        return [
            "--run-id",
            "123",
            "--run-attempt",
            "1",
            "--ref",
            "refs/heads/main",
            "--sha",
            "deadbeef",
            "--workflow",
            "gpu-ngc",
            "--job",
            "publish_logs",
            "--actor",
            "tester",
            "--check-runner-result",
            "success",
            "--has-runner",
            "false",
            "--probe-status",
            "http_403",
            "--no-runner-result",
            "success",
            "--gpu-job-result",
            "skipped",
            "--artifact-dir",
            str(out_dir / "artifacts"),
            "--out-run-info",
            str(out_dir / "run_info.json"),
            "--out-summary-json",
            str(out_dir / "dod_summary.json"),
            "--out-summary-md",
            str(out_dir / "dod_summary.md"),
        ]

    def test_skip_summary_on_runner_probe_403(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            code = self.mod.main(self._base_args(root))
            self.assertEqual(code, 0)
            summary = json.loads((root / "dod_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dod_status"], "skip")
            self.assertIn("RUNNER_DISCOVERY_TOKEN", "\n".join(summary.get("guidance") or []))

    def test_pass_summary_when_gpu_outputs_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "latency_trt.json").write_text("{}", encoding="utf-8")
            (artifacts / "pred_trt.json").write_text("{}", encoding="utf-8")
            (artifacts / "pred_onnxrt.json").write_text("{}", encoding="utf-8")
            (artifacts / "parity_trt_vs_onnxrt.log").write_text("ok", encoding="utf-8")
            (artifacts / "opencv_cuda_smoke.json").write_text(
                json.dumps({"status": "skipped", "reason": "opencv cuda backend unavailable"}),
                encoding="utf-8",
            )
            args = self._base_args(root)
            # has-runner / gpu-job-result switched to success
            args[args.index("--has-runner") + 1] = "true"
            args[args.index("--probe-status") + 1] = "ok"
            args[args.index("--gpu-job-result") + 1] = "success"

            code = self.mod.main(args)
            self.assertEqual(code, 0)
            summary = json.loads((root / "dod_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dod_status"], "pass")
            self.assertEqual(summary.get("checks", {}).get("opencv_cuda_status"), "skipped")

    def test_fail_summary_when_gpu_job_success_but_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            args = self._base_args(root)
            args[args.index("--has-runner") + 1] = "true"
            args[args.index("--probe-status") + 1] = "ok"
            args[args.index("--gpu-job-result") + 1] = "success"

            code = self.mod.main(args)
            self.assertEqual(code, 1)
            summary = json.loads((root / "dod_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dod_status"], "fail")
            findings = "\n".join(summary.get("findings") or [])
            self.assertIn("Missing expected artifacts", findings)


if __name__ == "__main__":
    unittest.main()
