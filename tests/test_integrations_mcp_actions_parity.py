import copy
import unittest
from unittest.mock import patch

from yolozu.integrations import tool_runner
from yolozu.integrations.tool_reference import build_tool_surface_reference, collect_surface_parity_errors

try:
    from yolozu.integrations import actions_api
except ImportError:  # pragma: no cover - optional Actions dependency
    actions_api = None


class TestIntegrationsMcpActionsParity(unittest.TestCase):
    def test_actions_ttt_route_redacts_unexpected_submission_errors(self):
        if actions_api is None:
            self.skipTest("Actions dependencies are not installed")
        request = actions_api.TTTJobRequest(
            dataset="data/smoke",
            checkpoint="checkpoints/model.pt",
        )
        with patch.object(
            actions_api,
            "ttt_job",
            side_effect=RuntimeError("private stack detail"),
        ):
            payload = actions_api.ttt_job_route(request)

        self.assertEqual(payload["error"], "internal job submission error")
        self.assertNotIn("private stack detail", str(payload))

    def test_surface_parity_has_no_drift(self):
        reference = build_tool_surface_reference()
        errors = collect_surface_parity_errors(reference)
        self.assertEqual(errors, [], "parity drift detected:\n- " + "\n- ".join(errors))

    def test_surface_cardinality_and_full_parameter_contracts(self):
        reference = build_tool_surface_reference()
        surfaces = reference["surfaces"]
        self.assertEqual(len(surfaces["guaranteed_ai_safe"]["tool_ids"]), 4)
        self.assertEqual(len(surfaces["mcp_live"]["tool_ids"]), 26)
        self.assertEqual(len(surfaces["actions_public"]["tool_ids"]), 21)
        self.assertEqual(len(reference["mcp_live_tools"]), 26)
        self.assertEqual(len(reference["tools"]), 21)

        for tool in reference["tools"]:
            with self.subTest(tool=tool["canonical_name"]):
                self.assertEqual(
                    tool["mcp"]["parameter_schema"],
                    tool["actions"]["parameter_schema"],
                )
                self.assertEqual(
                    tool["mcp"]["params"],
                    tool["tool_runner"]["params"],
                )

    def test_parity_checker_rejects_a_schema_contract_regression(self):
        reference = build_tool_surface_reference()
        tampered = copy.deepcopy(reference)
        tampered["tools"][0]["parity"]["mcp_vs_actions_schema"] = False

        errors = collect_surface_parity_errors(tampered)

        self.assertTrue(
            any("mcp_vs_actions_schema" in error for error in errors),
            errors,
        )

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
                ("ttt_job", ("data/smoke", "checkpoints/model.pt"), {}),
                ("ctta_job", ("data/smoke", "checkpoints/model.pt"), {}),
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
