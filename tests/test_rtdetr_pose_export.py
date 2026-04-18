import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


class TestRtDetrPoseExport(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_export_onnx_uses_tensor_outputs_from_model_dict(self) -> None:
        import torch

        from rtdetr_pose.export import export_onnx

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                return {
                    "custom_logits": torch.zeros((1, 2), dtype=torch.float32),
                    "bbox": torch.zeros((1, 4), dtype=torch.float32),
                    "log_z": torch.tensor(1.0, dtype=torch.float32),
                    "non_tensor_meta": {"ignored": True},
                }

        captured: dict[str, object] = {}

        def _fake_export(model, dummy_input, output_path, **kwargs):
            captured["dynamic_axes"] = kwargs.get("dynamic_axes")
            captured["output_names"] = kwargs.get("output_names")
            Path(output_path).write_bytes(b"onnx")

        fake_onnx = types.SimpleNamespace(__version__="test")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out_path = Path(td) / "model.onnx"
            with unittest.mock.patch.dict(sys.modules, {"onnx": fake_onnx}):
                with unittest.mock.patch("torch.onnx.export", side_effect=_fake_export):
                    export_onnx(FakeModel(), torch.zeros((1, 3, 8, 8)), str(out_path))

        dyn_axes = captured.get("dynamic_axes")
        self.assertIsInstance(dyn_axes, dict)
        self.assertEqual(captured.get("output_names"), ["custom_logits", "bbox", "log_z"])
        self.assertIn("custom_logits", dyn_axes)
        self.assertIn("bbox", dyn_axes)
        self.assertNotIn("log_z", dyn_axes)
        self.assertNotIn("non_tensor_meta", dyn_axes)


if __name__ == "__main__":
    unittest.main()
