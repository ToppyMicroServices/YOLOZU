import argparse
import unittest

from yolozu.inference import export_orchestrator


def _args(**overrides):
    ns = argparse.Namespace(
        tta=False,
        ttt=False,
        ttt_lite_non_torch=False,
        lora_r=0,
        torch_compile=False,
        torch_compile_backend="inductor",
        torch_compile_mode="reduce-overhead",
        torch_amp="off",
        torch_channels_last=False,
        torch_inference_mode=True,
        infer_batch_size=1,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


class TestExportOrchestratorTTTLite(unittest.TestCase):
    def test_ensure_wrapper_preserves_explicit_schema_version(self):
        payload = {
            "schema_version": 1,
            "predictions": [{"schema_version": 2, "image": "a.jpg", "detections": []}],
            "meta": {"adapter": "test"},
        }
        wrapped = export_orchestrator.ensure_wrapper(payload)
        self.assertEqual(wrapped["schema_version"], 1)
        self.assertEqual(wrapped["predictions"][0]["schema_version"], 2)
        self.assertEqual(wrapped["meta"]["adapter"], "test")

    def test_ensure_wrapper_versions_new_wrapper(self):
        wrapped = export_orchestrator.ensure_wrapper(
            [{"schema_version": 2, "image": "a.jpg", "detections": []}]
        )
        self.assertEqual(wrapped["schema_version"], 1)

    def test_ensure_wrapper_does_not_read_version_from_legacy_image_mapping(self):
        wrapped = export_orchestrator.ensure_wrapper({"schema_version": []})
        self.assertEqual(wrapped["schema_version"], 1)
        self.assertEqual(
            wrapped["predictions"],
            [{"image": "schema_version", "detections": []}],
        )

    def test_validate_torch_only_flags_allows_ttt_lite_non_torch(self):
        args = _args(ttt=True, ttt_lite_non_torch=True)
        export_orchestrator.validate_torch_only_flags(args=args, backend="onnxrt")

    def test_validate_torch_only_flags_rejects_ttt_without_lite(self):
        args = _args(ttt=True, ttt_lite_non_torch=False)
        with self.assertRaises(SystemExit):
            export_orchestrator.validate_torch_only_flags(args=args, backend="onnxrt")

    def test_validate_compile_report_accepts_proven_first_execution(self):
        report = {
            "requested": {
                "enabled": True,
                "backend": "eager",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
                "allow_fallback": False,
            },
            "actual": {
                "status": "compiled",
                "backend": "eager",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
            },
            "evidence": {
                "compile_api_available": True,
                "setup_completed": True,
                "first_execution_completed": True,
                "fallback_execution_completed": False,
                "counter_source": "torch._dynamo.utils.counters",
                "counter_delta": {"stats.unique_graphs": 1},
                "graph_count": 1,
                "graph_break_count": None,
                "captured_call_count": 1,
            },
            "failure": None,
        }
        export_orchestrator.validate_compile_report(
            report,
            enabled=True,
            backend="eager",
            mode="default",
            fullgraph=False,
            dynamic=False,
            allow_fallback=False,
        )

    def test_validate_compile_report_rejects_pending_lazy_compile(self):
        report = {
            "requested": {
                "enabled": True,
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": None,
                "allow_fallback": False,
            },
            "actual": {
                "status": "pending_first_execution",
                "backend": None,
                "mode": None,
                "fullgraph": None,
                "dynamic": None,
            },
            "evidence": {
                "compile_api_available": True,
                "setup_completed": True,
                "first_execution_completed": False,
                "fallback_execution_completed": False,
                "counter_source": None,
                "counter_delta": None,
                "graph_count": None,
                "graph_break_count": None,
                "captured_call_count": None,
            },
            "failure": None,
        }
        with self.assertRaisesRegex(ValueError, "not established"):
            export_orchestrator.validate_compile_report(
                report,
                enabled=True,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=None,
                allow_fallback=False,
            )

    def test_validate_compile_report_accepts_explicit_fallback(self):
        report = {
            "requested": {
                "enabled": True,
                "backend": "missing",
                "mode": "default",
                "fullgraph": True,
                "dynamic": None,
                "allow_fallback": True,
            },
            "actual": {
                "status": "fallback",
                "backend": "eager",
                "mode": None,
                "fullgraph": False,
                "dynamic": None,
            },
            "evidence": {
                "compile_api_available": True,
                "setup_completed": False,
                "first_execution_completed": False,
                "fallback_execution_completed": True,
                "counter_source": None,
                "counter_delta": None,
                "graph_count": None,
                "graph_break_count": None,
                "captured_call_count": None,
            },
            "failure": {
                "phase": "setup",
                "type": "ValueError",
                "message": "invalid backend",
            },
        }
        export_orchestrator.validate_compile_report(
            report,
            enabled=True,
            backend="missing",
            mode="default",
            fullgraph=True,
            dynamic=None,
            allow_fallback=True,
        )


if __name__ == "__main__":
    unittest.main()
