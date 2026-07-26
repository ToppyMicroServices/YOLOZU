import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from yolozu.integrations import tool_runner


class TestIntegrationToolRunner(unittest.TestCase):
    def test_a1_validate_predictions_strict_default(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "validate_predictions", "summary": "ok"}
            out = tool_runner.validate_predictions("data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "validate_predictions",
            [
                "validate",
                "predictions",
                "data/smoke/predictions/predictions_dummy.json",
                "--json",
                "--strict",
            ],
            json_result_key="validation",
        )

    def test_a1b_validate_predictions_non_strict_requests_repair_mode(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {
                "ok": True,
                "tool": "validate_predictions",
                "summary": "ok",
            }
            out = tool_runner.validate_predictions(
                "reports/legacy_predictions.json",
                strict=False,
            )

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "validate_predictions",
            [
                "validate",
                "predictions",
                "reports/legacy_predictions.json",
                "--json",
            ],
            json_result_key="validation",
        )

    def test_a2_validate_dataset_builds_expected_args(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "validate_dataset", "summary": "ok"}
            out = tool_runner.validate_dataset("data/smoke", split="val", strict=True, mode="warn")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "validate_dataset",
            ["validate", "dataset", "data/smoke", "--mode", "warn", "--split", "val", "--strict"],
        )

    def test_a3_convert_dataset_manifest_mode_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "convert_dataset", "summary": "ok"}
            out = tool_runner.convert_dataset(
                from_format="ultralytics",
                output="reports/converted_dataset",
                data="data/smoke",
            )

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "convert_dataset",
            [
                "migrate",
                "dataset",
                "--from",
                "ultralytics",
                "--output",
                "reports/converted_dataset",
                "--mode",
                "manifest",
                "--data",
                "data/smoke",
                "--force",
            ],
        )

    def test_b4_predict_images_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "predict_images", "summary": "ok"}
            out = tool_runner.predict_images("data/smoke/images/val")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "predict_images",
            [
                "predict-images",
                "--backend",
                "dummy",
                "--input-dir",
                "data/smoke/images/val",
                "--output",
                "reports/mcp_predict_images.json",
                "--dry-run",
                "--strict",
                "--force",
            ],
            artifacts={"predictions": "reports/mcp_predict_images.json"},
        )

    def test_b5_parity_check_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "parity_check", "summary": "ok"}
            out = tool_runner.parity_check("reports/ref.json", "reports/cand.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "parity_check",
            [
                "parity",
                "--reference",
                "reports/ref.json",
                "--candidate",
                "reports/cand.json",
                "--iou-thresh",
                "0.5",
                "--score-atol",
                "1e-06",
                "--bbox-atol",
                "0.0001",
            ],
        )

    def test_b6_calibrate_predictions_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "calibrate_predictions", "summary": "ok"}
            out = tool_runner.calibrate_predictions("data/smoke", "reports/predictions.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "calibrate_predictions",
            [
                "calibrate",
                "--method",
                "fracal",
                "--dataset",
                "data/smoke",
                "--task",
                "auto",
                "--predictions",
                "reports/predictions.json",
                "--output",
                "reports/mcp_calibrated_predictions.json",
                "--output-report",
                "reports/mcp_calibration_report.json",
                "--force",
            ],
            artifacts={
                "predictions": "reports/mcp_calibrated_predictions.json",
                "report": "reports/mcp_calibration_report.json",
            },
        )

    def test_c7_eval_coco_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_coco", "summary": "ok"}
            out = tool_runner.eval_coco("data/smoke", "data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_coco",
            [
                "eval-coco",
                "--dataset",
                "data/smoke",
                "--predictions",
                "data/smoke/predictions/predictions_dummy.json",
                "--output",
                "reports/mcp_coco_eval.json",
                "--dry-run",
            ],
            artifacts={"report": "reports/mcp_coco_eval.json"},
        )

    def test_c7b_eval_coco_repair_is_explicit(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {
                "ok": True,
                "tool": "eval_coco",
                "summary": "ok",
            }
            out = tool_runner.eval_coco(
                "data/smoke",
                "reports/legacy_predictions.json",
                repair=True,
            )

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_coco",
            [
                "eval-coco",
                "--dataset",
                "data/smoke",
                "--predictions",
                "reports/legacy_predictions.json",
                "--output",
                "reports/mcp_coco_eval.json",
                "--dry-run",
                "--repair",
            ],
            artifacts={"report": "reports/mcp_coco_eval.json"},
        )

    def test_c8_eval_instance_seg_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_instance_seg", "summary": "ok"}
            out = tool_runner.eval_instance_seg("data/smoke", "reports/instance_seg_predictions.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_instance_seg",
            [
                "eval-instance-seg",
                "--dataset",
                "data/smoke",
                "--predictions",
                "reports/instance_seg_predictions.json",
                "--output",
                "reports/mcp_instance_seg_eval.json",
            ],
            artifacts={"report": "reports/mcp_instance_seg_eval.json"},
        )

    def test_c9_eval_long_tail_defaults(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            run_cli.return_value = {"ok": True, "tool": "eval_long_tail", "summary": "ok"}
            out = tool_runner.eval_long_tail("data/smoke", "data/smoke/predictions/predictions_dummy.json")

        self.assertTrue(out["ok"])
        run_cli.assert_called_once_with(
            "eval_long_tail",
            [
                "eval-long-tail",
                "--dataset",
                "data/smoke",
                "--predictions",
                "data/smoke/predictions/predictions_dummy.json",
                "--output",
                "reports/mcp_long_tail_eval.json",
            ],
            artifacts={"report": "reports/mcp_long_tail_eval.json"},
        )

    def test_d10_train_job_args(self):
        with patch("yolozu.integrations.tool_runner.submit_job") as submit_job:
            submit_job.return_value = {"ok": True, "tool": "jobs.submit", "job_id": "job_x"}
            out = tool_runner.train_job("configs/train.yaml", run_id="exp01", resume="runs/exp00")

        self.assertTrue(out["ok"])
        submit_job.assert_called_once_with("train", ["train", "configs/train.yaml", "--run-id", "exp01", "--resume", "runs/exp00"])

    def test_d11_export_predictions_job_args(self):
        with patch("yolozu.integrations.tool_runner.submit_job") as submit_job:
            submit_job.return_value = {"ok": True, "tool": "jobs.submit", "job_id": "job_x"}
            out = tool_runner.export_predictions_job("data/smoke", "reports/export_predictions.json", split="val", force=True)

        self.assertTrue(out["ok"])
        submit_job.assert_called_once_with(
            "export_predictions",
            [
                "export",
                "--backend",
                "labels",
                "--dataset",
                "data/smoke",
                "--output",
                "reports/export_predictions.json",
                "--split",
                "val",
                "--force",
            ],
            artifacts={"predictions": "reports/export_predictions.json"},
        )

    def test_d11_export_onnx_job_alias_args(self):
        with patch("yolozu.integrations.tool_runner.submit_job") as submit_job:
            submit_job.return_value = {"ok": True, "tool": "jobs.submit", "job_id": "job_x"}
            out = tool_runner.export_onnx_job("data/smoke", "reports/export_predictions.json", split="val", force=True)

        self.assertTrue(out["ok"])
        submit_job.assert_called_once_with(
            "export_predictions",
            [
                "export",
                "--backend",
                "labels",
                "--dataset",
                "data/smoke",
                "--output",
                "reports/export_predictions.json",
                "--split",
                "val",
                "--force",
            ],
            artifacts={"predictions": "reports/export_predictions.json"},
        )

    def test_d12_test_job_args(self):
        with patch("yolozu.integrations.tool_runner.submit_job") as submit_job:
            submit_job.return_value = {"ok": True, "tool": "jobs.submit", "job_id": "job_x"}
            out = tool_runner.test_job("configs/test.yaml", extra_args=["--max-images", "8"])

        self.assertTrue(out["ok"])
        submit_job.assert_called_once_with("test", ["test", "configs/test.yaml", "--max-images", "8"])

    def test_scenario_extra_args_reject_undeclared_flags_before_submit(self):
        with patch("yolozu.integrations.tool_runner.submit_job") as submit_job:
            out = tool_runner.test_job(
                "configs/test.yaml",
                extra_args=["-x=/tmp/outside.json"],
            )

        self.assertFalse(out["ok"])
        self.assertIn("undeclared flag", out["error"])
        submit_job.assert_not_called()

    def test_scenario_extra_args_reject_missing_values(self):
        with patch("yolozu.integrations.tool_runner.run_cli_tool") as run_cli:
            out = tool_runner.run_scenarios(
                "configs/test.yaml",
                extra_args=["--output"],
            )

        self.assertFalse(out["ok"])
        self.assertIn("requires one value", out["error"])
        run_cli.assert_not_called()

    def test_real_failed_subprocess_job_propagates_to_jobs_status(self):
        repo_root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        original_pythonpath = os.environ.get("PYTHONPATH")
        tool_runner._job_manager.cache_clear()
        manager = None
        try:
            with tempfile.TemporaryDirectory() as td:
                workspace = Path(td)
                (workspace / "invalid.json").write_text(
                    json.dumps(
                        {
                            "adapter": "not_a_real_adapter",
                            "dataset": "missing",
                        }
                    ),
                    encoding="utf-8",
                )
                os.chdir(workspace)
                os.environ["PYTHONPATH"] = str(repo_root)
                queued = tool_runner.test_job("invalid.json")
                self.assertTrue(queued["ok"], queued)
                manager = tool_runner._job_manager()

                response = {}
                for _ in range(200):
                    response = tool_runner.jobs_status(
                        queued["job_id"]
                    )
                    if (
                        (response.get("job") or {}).get("status")
                        == "failed"
                    ):
                        break
                    time.sleep(0.01)

                self.assertFalse(response["ok"], response)
                self.assertEqual(response["exit_code"], 1)
                self.assertEqual(
                    response["job"]["status"],
                    "failed",
                )
                self.assertFalse(
                    response["job"]["result"]["ok"],
                )
                self.assertNotEqual(
                    response["job"]["result"]["exit_code"],
                    0,
                )
                self.assertIn("cli failed", response["error"])
        finally:
            if manager is not None:
                manager._executor.shutdown(wait=True)
            tool_runner._job_manager.cache_clear()
            os.chdir(original_cwd)
            if original_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = original_pythonpath

    def test_e13_ttt_job_args(self):
        with patch(
            "yolozu.integrations.tool_runner._ttt_export_job"
        ) as export_job:
            export_job.return_value = {
                "ok": True,
                "tool": "jobs.submit",
                "job_id": "job_x",
            }
            out = tool_runner.ttt_job(
                "data/target",
                "weights/model.pt",
                "runs/ttt/predictions.json",
                config="configs/model.json",
                report="runs/ttt/report.json",
                method="tent",
                preset="safe",
                steps=2,
                reset="sample",
                max_images=4,
            )

        self.assertTrue(out["ok"])
        export_job.assert_called_once_with(
            job_name="ttt",
            dataset="data/target",
            checkpoint="weights/model.pt",
            output="runs/ttt/predictions.json",
            config="configs/model.json",
            split="val",
            report="runs/ttt/report.json",
            method="tent",
            preset="safe",
            steps=2,
            reset="sample",
            device="cpu",
            max_images=4,
            force=True,
            public=False,
        )

    def test_e14_ctta_job_args(self):
        with patch(
            "yolozu.integrations.tool_runner._ttt_export_job"
        ) as export_job:
            export_job.return_value = {
                "ok": True,
                "tool": "jobs.submit",
                "job_id": "job_x",
            }
            out = tool_runner.ctta_job(
                "data/stream",
                "weights/model.pt",
                method="cotta",
                preset="conservative",
                steps=1,
                reset="stream",
                max_images=4,
            )

        self.assertTrue(out["ok"])
        export_job.assert_called_once_with(
            job_name="ctta",
            dataset="data/stream",
            checkpoint="weights/model.pt",
            output=None,
            config="builtin:base",
            split="val",
            report=None,
            method="cotta",
            preset="conservative",
            steps=1,
            reset="stream",
            device="cpu",
            max_images=4,
            force=True,
            public=False,
        )


if __name__ == "__main__":
    unittest.main()
