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
    def test_validate_torch_only_flags_allows_ttt_lite_non_torch(self):
        args = _args(ttt=True, ttt_lite_non_torch=True)
        export_orchestrator.validate_torch_only_flags(args=args, backend="onnxrt")

    def test_validate_torch_only_flags_rejects_ttt_without_lite(self):
        args = _args(ttt=True, ttt_lite_non_torch=False)
        with self.assertRaises(SystemExit):
            export_orchestrator.validate_torch_only_flags(args=args, backend="onnxrt")


if __name__ == "__main__":
    unittest.main()
