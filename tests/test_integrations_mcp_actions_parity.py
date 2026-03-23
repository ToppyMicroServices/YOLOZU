import unittest
from unittest.mock import patch

from yolozu.integrations import tool_runner
from yolozu.integrations.tool_reference import _ast_literal, build_tool_surface_reference, collect_surface_parity_errors


class TestIntegrationsMcpActionsParity(unittest.TestCase):
    def test_ast_literal_falls_back_to_expr_marker(self):
        self.assertEqual(_ast_literal(object()), "<expr>")

    def test_surface_parity_has_no_drift(self):
        reference = build_tool_surface_reference()
        errors = collect_surface_parity_errors(reference)
        self.assertEqual(errors, [], "parity drift detected:\n- " + "\n- ".join(errors))

    def test_tool_runner_responses_keep_contract_keys(self):
        required = {"ok", "tool", "summary", "exit_code"}
        run_cli_payload = {
            "ok": True,
            "tool": "stub",
            "summary": "stub ok",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "artifacts": {},
        }
        submit_payload = {
            "ok": True,
            "tool": "jobs.submit",
            "summary": "job queued: job_x",
            "exit_code": 0,
            "job_id": "job_x",
            "status": "queued",
            "meta": {},
        }

        with (
            patch("yolozu.integrations.tool_runner.run_cli_tool", return_value=run_cli_payload),
            patch("yolozu.integrations.tool_runner.submit_job", return_value=submit_payload),
            patch("yolozu.integrations.tool_runner.collect_artifact_metadata", return_value={}),
        ):
            samples = [
                ("doctor", (), {}),
                ("validate_predictions", ("data/smoke/predictions/predictions_dummy.json",), {}),
                ("validate_dataset", ("data/smoke",), {}),
                ("eval_coco", ("data/smoke", "data/smoke/predictions/predictions_dummy.json"), {}),
                ("predict_images", ("data/smoke/images/val",), {}),
                ("parity_check", ("reports/ref.json", "reports/cand.json"), {}),
                ("calibrate_predictions", ("data/smoke", "reports/predictions.json"), {}),
                ("eval_instance_seg", ("data/smoke", "reports/instance_seg_predictions.json"), {}),
                ("eval_long_tail", ("data/smoke", "data/smoke/predictions/predictions_dummy.json"), {}),
                ("run_scenarios", ("configs/test.yaml",), {}),
                ("convert_dataset", (), {"from_format": "ultralytics", "output": "reports/converted_dataset", "data": "data/smoke"}),
                ("train_job", ("configs/train.yaml",), {}),
                ("export_predictions_job", ("data/smoke", "reports/export_predictions.json"), {}),
                ("test_job", ("configs/test.yaml",), {}),
                ("ttt_job", ("configs/test.yaml",), {}),
                ("ctta_job", ("configs/test.yaml",), {}),
                ("jobs_list", (), {}),
                ("jobs_status", ("job_unknown",), {}),
                ("jobs_cancel", ("job_unknown",), {}),
                ("runs_list", (), {}),
                ("runs_describe", ("run_unknown",), {}),
            ]

            for fn_name, args, kwargs in samples:
                out = getattr(tool_runner, fn_name)(*args, **kwargs)
                self.assertTrue(required.issubset(out.keys()), f"{fn_name} missing required keys: {sorted(required - set(out.keys()))}")


if __name__ == "__main__":
    unittest.main()
