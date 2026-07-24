import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from yolozu.integrations.layers import api as api_layer
from yolozu.integrations.layers.api import run_cli_tool
from yolozu.integrations.layers.core import fail_response, ok_response
from yolozu.integrations.layers.jobs import JobManager
from yolozu.integrations.tool_runner import jobs_list, jobs_status, runs_describe, runs_list


class TestIntegrationLayers(unittest.TestCase):
    def test_core_response_shapes(self):
        ok = ok_response("x", data={"a": 1})
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["tool"], "x")
        self.assertIn("summary", ok)

        err = fail_response("y", message="boom")
        self.assertFalse(err["ok"])
        self.assertEqual(err["tool"], "y")
        self.assertEqual(err["error"], "boom")

    def test_api_layer_blocks_path_traversal(self):
        out = run_cli_tool("validate_predictions", ["validate", "predictions", "../bad.json"])
        self.assertFalse(out["ok"])
        self.assertIn("path traversal", out.get("error", ""))

    def test_api_layer_allows_absolute_path_inside_explicit_workspace(self):
        completed = subprocess.CompletedProcess(args=["x"], returncode=0, stdout="{}", stderr="")
        with tempfile.TemporaryDirectory() as td:
            inside = Path(td) / "predictions.json"
            with patch(
                "yolozu.integrations.layers.api.subprocess.run",
                return_value=completed,
            ) as run:
                out = run_cli_tool(
                    "validate_predictions",
                    ["validate", "predictions", str(inside)],
                    workspace=td,
                )
        self.assertTrue(out["ok"])
        self.assertEqual(
            run.call_args.kwargs["cwd"],
            str(Path(td).resolve()),
        )

    def test_api_layer_rejects_absolute_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            out = run_cli_tool(
                "validate_predictions",
                ["validate", "predictions", "/tmp/outside.json"],
                workspace=td,
            )
        self.assertFalse(out["ok"])
        self.assertIn("path escapes workspace", out.get("error", ""))

    def test_api_layer_rejects_non_allowlisted_command(self):
        out = run_cli_tool("bad", ["rm", "-rf", "/tmp/x"])
        self.assertFalse(out["ok"])
        self.assertIn("command not allowed", out.get("error", ""))

    def test_api_allowlist_is_manifest_first_with_fallback(self):
        with patch("yolozu.integrations.layers.api._allowed_from_manifest", return_value={"doctor"}):
            allowed = api_layer._build_allowed_top_level()
        self.assertIn("doctor", allowed)
        self.assertIn("validate", allowed)

    def test_jobs_manager_submit_and_status(self):
        manager = JobManager(max_workers=1)
        job_id = manager.submit("dummy", lambda: {"ok": True, "tool": "dummy"})
        status = manager.status(job_id)
        self.assertIsNotNone(status)
        self.assertIn(status["status"], ("queued", "running", "completed"))

    def test_jobs_manager_persistence_reload(self):
        with tempfile.TemporaryDirectory() as td:
            manager = JobManager(max_workers=1, storage_dir=td)
            job_id = manager.submit("dummy", lambda: {"ok": True, "tool": "dummy"})
            for _ in range(100):
                status = manager.status(job_id)
                if status and status["status"] == "completed":
                    break
                time.sleep(0.01)
            reloaded = JobManager(max_workers=1, storage_dir=td)
            status2 = reloaded.status(job_id)
            self.assertIsNotNone(status2)
            self.assertEqual(status2["status"], "completed")

    def test_jobs_manager_stale_running_becomes_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            payload = {
                "job_id": "job_stale",
                "name": "stale",
                "status": "running",
                "created_at": time.time(),
                "started_at": time.time(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
            Path(td, "job_stale.json").write_text(json.dumps(payload), encoding="utf-8")
            manager = JobManager(max_workers=1, storage_dir=td)
            status = manager.status("job_stale")
            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "unknown")

    def test_runs_list_and_describe_shape(self):
        runs = runs_list(limit=2)
        self.assertTrue(runs["ok"])
        self.assertIn("runs", runs)

        with tempfile.TemporaryDirectory() as td:
            run_id = Path(td).name
            missing = runs_describe(run_id)
            self.assertIn("ok", missing)

    def test_jobs_api_shape(self):
        lst = jobs_list()
        self.assertTrue(lst["ok"])
        self.assertIn("exit_code", lst)
        unknown = jobs_status("job_unknown")
        self.assertFalse(unknown["ok"])
        self.assertIn("exit_code", unknown)

    def test_api_layer_includes_limits_metadata(self):
        completed = subprocess.CompletedProcess(args=["x"], returncode=0, stdout="ok", stderr="")
        with patch("yolozu.integrations.layers.api.subprocess.run", return_value=completed):
            out = run_cli_tool("doctor", ["doctor"])
        self.assertTrue(out["ok"])
        self.assertIn("limits", out)
        self.assertFalse(out["limits"]["stdout_truncated"])
        self.assertFalse(out["limits"]["stderr_truncated"])

    def test_api_layer_truncates_large_output(self):
        big = "a" * 210_000
        completed = subprocess.CompletedProcess(args=["x"], returncode=0, stdout=big, stderr="")
        with patch("yolozu.integrations.layers.api.subprocess.run", return_value=completed):
            out = run_cli_tool("doctor", ["doctor"])
        self.assertTrue(out["ok"])
        self.assertTrue(out["limits"]["stdout_truncated"])
        self.assertIn("...[truncated]", out["stdout"])

    def test_api_layer_timeout_shape(self):
        timeout_error = subprocess.TimeoutExpired(cmd=["x"], timeout=1, output="partial", stderr="err")
        with patch("yolozu.integrations.layers.api.subprocess.run", side_effect=timeout_error):
            out = run_cli_tool("doctor", ["doctor"])
        self.assertFalse(out["ok"])
        self.assertEqual(out["exit_code"], 124)
        self.assertIn("timeout", out["error"])
        self.assertIn("limits", out)


if __name__ == "__main__":
    unittest.main()
