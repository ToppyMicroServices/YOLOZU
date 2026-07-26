import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from yolozu.integrations import tool_runner
from yolozu.integrations.layers.jobs import JobManager

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime
    torch = None


class TestIntegrationTTTJobs(unittest.TestCase):
    def _make_dataset(self, root: Path, repo_root: Path) -> Path:
        dataset = root / "dataset"
        images = dataset / "images" / "val"
        labels = dataset / "labels" / "val"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        source = repo_root / "data" / "smoke" / "images" / "val" / "000000000009.jpg"
        shutil.copyfile(source, images / "000001.jpg")
        (labels / "000001.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        return dataset

    def _make_compatible_checkpoint(self, root: Path) -> tuple[Path, Path]:
        from yolozu.adapter import RTDETRPoseAdapter

        config = root / "tiny_rtdetr.json"
        config.write_text(
            json.dumps(
                {
                    "dataset": {
                        "root": ".",
                        "split": "val",
                        "format": "yolo",
                    },
                    "model": {
                        "num_classes": 1,
                        "hidden_dim": 64,
                        "num_queries": 10,
                        "stem_channels": 8,
                        "backbone_channels": [16, 32, 64],
                        "stage_blocks": [1, 1, 1],
                        "num_encoder_layers": 1,
                        "num_decoder_layers": 1,
                        "nhead": 4,
                        "encoder_dim_feedforward": 128,
                        "decoder_dim_feedforward": 128,
                    },
                    "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
                }
            ),
            encoding="utf-8",
        )
        adapter = RTDETRPoseAdapter(
            config_path=str(config),
            device="cpu",
            image_size=(32, 32),
        )
        checkpoint = root / "compatible.pt"
        torch.save(adapter.get_model().state_dict(), checkpoint)
        return config, checkpoint

    def _wait_for_terminal(self, job_id: str) -> dict:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            payload = tool_runner.jobs_status(job_id)
            job = payload.get("job") or {}
            if job.get("status") in {"completed", "failed", "cancelled"}:
                return payload
            time.sleep(0.02)
        self.fail(f"job did not reach a terminal state: {job_id}")
        return {}

    def test_missing_checkpoint_fails_before_queueing(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root, repo_root)
            dataset_rel = str(dataset.relative_to(repo_root))
            manager = JobManager(max_workers=1, storage_dir=root / "jobs")
            try:
                with patch.object(
                    tool_runner,
                    "_job_manager",
                    return_value=manager,
                ):
                    before = len(manager.list())
                    result = tool_runner.ttt_job(
                        dataset_rel,
                        f"{root.name}/missing.pt",
                    )
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["exit_code"], 2)
                    self.assertEqual(result["stage"], "preflight")
                    self.assertIs(result["queued"], False)
                    self.assertIn("fully compatible", result["error"])
                    self.assertEqual(len(manager.list()), before)
            finally:
                manager._executor.shutdown(wait=True)

    @unittest.skipIf(torch is None, "torch is required for live TTT job integration")
    def test_ttt_and_ctta_jobs_complete_with_exit_and_reports(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root, repo_root)
            config, checkpoint = self._make_compatible_checkpoint(root)

            def rel(path: Path) -> str:
                return str(path.relative_to(repo_root))

            manager = JobManager(max_workers=1, storage_dir=root / "jobs")
            try:
                with patch.object(
                    tool_runner,
                    "_job_manager",
                    return_value=manager,
                ):
                    cases = (
                        (
                            "ttt",
                            tool_runner.ttt_job,
                            "tent",
                            "sample",
                        ),
                        (
                            "ctta",
                            tool_runner.ctta_job,
                            "cotta",
                            "stream",
                        ),
                    )
                    for lane, submit, method, reset in cases:
                        output = root / f"{lane}_predictions.json"
                        report = root / f"{lane}_report.json"
                        queued = submit(
                            f"  {rel(dataset)}  ",
                            f"  {rel(checkpoint)}  ",
                            f"  {rel(output)}  ",
                            config=f"  {rel(config)}  ",
                            report=f"  {rel(report)}  ",
                            method=method,
                            reset=reset,
                            steps=1,
                            max_images=1,
                        )
                        self.assertTrue(queued["ok"], queued)
                        self.assertEqual(queued["status"], "queued")
                        self.assertEqual(queued["preflight"]["status"], "full")
                        terminal = self._wait_for_terminal(queued["job_id"])
                        job = terminal["job"]
                        self.assertEqual(job["status"], "completed", terminal)
                        result = job["result"]
                        self.assertTrue(result["ok"], result)
                        self.assertEqual(result["exit_code"], 0)
                        self.assertEqual(result["command"][3], "export")
                        self.assertNotIn("test", result["command"][3:])
                        self.assertTrue(output.is_file())
                        self.assertTrue(report.is_file())
                        self.assertEqual(
                            result["artifacts"]["predictions"],
                            rel(output),
                        )
                        self.assertEqual(
                            result["artifacts"]["ttt_report"],
                            rel(report),
                        )
                        report_payload = json.loads(
                            report.read_text(encoding="utf-8")
                        )
                        ttt = report_payload["ttt"]
                        self.assertEqual(ttt["method"], method)
                        method_report = ttt["report"]
                        if method_report.get("mode") == "sample":
                            method_report = method_report["per_sample"][0][
                                "report"
                            ]
                        self.assertEqual(
                            method_report["method_profile"]["efficacy"],
                            "not_established",
                        )
                        persisted = json.loads(
                            (
                                root
                                / "jobs"
                                / f"{queued['job_id']}.json"
                            ).read_text(encoding="utf-8")
                        )
                        self.assertEqual(persisted["status"], "completed")
                        self.assertEqual(
                            persisted["result"]["exit_code"],
                            0,
                        )
            finally:
                manager._executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
