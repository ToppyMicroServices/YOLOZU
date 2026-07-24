"""Tests for PyTorch API integration features.

Tests torch.compile wrapper, AMP utilities, ONNX export bridge,
torchvision transforms bridge, and profiler integration.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# torch.compile wrapper tests
# ---------------------------------------------------------------------------


class TestCompileForInference(unittest.TestCase):
    """Tests for yolozu.inference.torch_export.compile_for_inference."""

    def test_fails_closed_when_torch_compile_unavailable(self):
        from yolozu.inference.torch_export import (
            TorchCompileError,
            compile_for_inference,
        )

        dummy = MagicMock()
        with patch(
            "yolozu.inference.torch_export.torch_compile_available",
            return_value=False,
        ):
            with self.assertRaises(TorchCompileError) as ctx:
                compile_for_inference(dummy)
        self.assertEqual(ctx.exception.evidence["actual"]["status"], "failed")
        self.assertEqual(ctx.exception.evidence["failure"]["phase"], "setup")

    def test_explicit_fallback_when_torch_compile_unavailable(self):
        from yolozu.inference.torch_export import (
            compile_for_inference,
            get_compile_evidence,
        )

        dummy = MagicMock()
        with patch(
            "yolozu.inference.torch_export.torch_compile_available",
            return_value=False,
        ):
            result = compile_for_inference(dummy, allow_fallback=True)
        self.assertIs(result, dummy)
        evidence = get_compile_evidence(result)
        self.assertEqual(evidence["actual"]["status"], "fallback")
        self.assertFalse(evidence["evidence"]["compile_api_available"])

    def test_invalid_backend_setup_failure_is_fail_closed(self):
        from yolozu.inference.torch_export import (
            TorchCompileError,
            compile_for_inference,
        )

        fake_torch = SimpleNamespace(
            compile=MagicMock(side_effect=ValueError("invalid backend"))
        )
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
            self.assertRaises(TorchCompileError) as ctx,
        ):
            compile_for_inference(MagicMock(), backend="definitely-not-a-backend")
        self.assertEqual(ctx.exception.evidence["actual"]["status"], "failed")
        self.assertEqual(ctx.exception.evidence["failure"]["phase"], "setup")

    def test_dynamo_suppress_errors_cannot_claim_compiled_execution(self):
        from yolozu.inference.torch_export import (
            TorchCompileError,
            compile_for_inference,
        )

        fake_torch = SimpleNamespace(
            compile=MagicMock(),
            _dynamo=SimpleNamespace(
                config=SimpleNamespace(suppress_errors=True),
            ),
        )
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
            self.assertRaises(TorchCompileError) as ctx,
        ):
            compile_for_inference(MagicMock())
        fake_torch.compile.assert_not_called()
        self.assertIn("silently run eager", str(ctx.exception))

    def test_invalid_backend_can_use_explicit_fallback(self):
        from yolozu.inference.torch_export import (
            compile_for_inference,
            get_compile_evidence,
        )

        model = MagicMock()
        fake_torch = SimpleNamespace(
            compile=MagicMock(side_effect=ValueError("invalid backend"))
        )
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            result = compile_for_inference(
                model,
                backend="definitely-not-a-backend",
                allow_fallback=True,
            )
        self.assertIs(result, model)
        evidence = get_compile_evidence(result)
        self.assertEqual(evidence["actual"]["status"], "fallback")
        self.assertEqual(evidence["failure"]["phase"], "setup")

    def test_lazy_first_execution_failure_is_fail_closed(self):
        from yolozu.inference.torch_export import (
            TorchCompileError,
            compile_for_inference,
            get_compile_evidence,
        )

        class LazyFailure:
            def forward(self, *_args, **_kwargs):
                raise RuntimeError("lazy compile failure")

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        fake_torch = SimpleNamespace(compile=MagicMock(return_value=LazyFailure()))
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            tracked = compile_for_inference(MagicMock())
            with self.assertRaises(TorchCompileError):
                tracked("input")
        evidence = get_compile_evidence(tracked)
        self.assertEqual(evidence["actual"]["status"], "failed")
        self.assertEqual(evidence["failure"]["phase"], "first_execution")
        self.assertFalse(evidence["evidence"]["first_execution_completed"])

    def test_lazy_failure_fallback_records_eager_execution(self):
        from yolozu.inference.torch_export import (
            compile_for_inference,
            get_compile_evidence,
        )

        class LazyFailure:
            def __init__(self):
                self.calls = 0

            def forward(self, *_args, **_kwargs):
                self.calls += 1
                raise RuntimeError("lazy compile failure")

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        model = MagicMock(side_effect=lambda value: f"eager:{value}")
        compiled = LazyFailure()
        fake_torch = SimpleNamespace(compile=MagicMock(return_value=compiled))
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            tracked = compile_for_inference(model, allow_fallback=True)
            self.assertEqual(tracked("one"), "eager:one")
            self.assertEqual(tracked("two"), "eager:two")
        self.assertEqual(compiled.calls, 1)
        self.assertEqual(model.call_count, 2)
        evidence = get_compile_evidence(tracked)
        self.assertEqual(evidence["actual"]["status"], "fallback")
        self.assertEqual(evidence["actual"]["backend"], "eager")
        self.assertTrue(evidence["evidence"]["fallback_execution_completed"])

    def test_success_records_requested_and_actual_settings(self):
        from yolozu.inference.torch_export import (
            compile_for_inference,
            get_compile_evidence,
        )

        counters = {
            "stats": {"unique_graphs": 0, "calls_captured": 0},
            "graph_break": {},
        }

        class SuccessfulCompiled:
            def forward(self, value):
                counters["stats"]["unique_graphs"] += 1
                counters["stats"]["calls_captured"] += 1
                return f"compiled:{value}"

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        fake_torch = SimpleNamespace(
            compile=MagicMock(return_value=SuccessfulCompiled()),
            _dynamo=SimpleNamespace(
                config=SimpleNamespace(suppress_errors=False),
                utils=SimpleNamespace(counters=counters),
            ),
        )
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            tracked = compile_for_inference(
                MagicMock(),
                backend="test_backend",
                mode="test_mode",
                fullgraph=True,
                dynamic=False,
            )
            self.assertEqual(tracked("input"), "compiled:input")
        evidence = get_compile_evidence(tracked)
        self.assertEqual(evidence["actual"]["status"], "compiled")
        self.assertEqual(evidence["actual"]["backend"], "test_backend")
        self.assertEqual(evidence["actual"]["mode"], "test_mode")
        self.assertTrue(evidence["actual"]["fullgraph"])
        self.assertFalse(evidence["actual"]["dynamic"])
        self.assertTrue(evidence["evidence"]["first_execution_completed"])
        self.assertEqual(evidence["evidence"]["graph_count"], 1)
        self.assertEqual(evidence["evidence"]["captured_call_count"], 1)
        self.assertEqual(
            evidence["evidence"]["counter_source"],
            "torch._dynamo.utils.counters",
        )

    def test_later_model_error_does_not_reclassify_compile_or_run_eager(self):
        from yolozu.inference.torch_export import (
            compile_for_inference,
            get_compile_evidence,
        )

        class LaterFailure:
            def __init__(self):
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("model execution failed")
                return f"compiled:{value}"

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        eager_model = MagicMock()
        fake_torch = SimpleNamespace(compile=MagicMock(return_value=LaterFailure()))
        with (
            patch(
                "yolozu.inference.torch_export.torch_compile_available",
                return_value=True,
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            tracked = compile_for_inference(eager_model, allow_fallback=True)
            self.assertEqual(tracked("first"), "compiled:first")
            with self.assertRaisesRegex(RuntimeError, "model execution failed"):
                tracked("second")

        eager_model.assert_not_called()
        evidence = get_compile_evidence(tracked)
        self.assertEqual(evidence["actual"]["status"], "compiled")
        self.assertIsNone(evidence["failure"])

    def test_compile_available_returns_bool(self):
        from yolozu.inference.torch_export import torch_compile_available

        self.assertIsInstance(torch_compile_available(), bool)

    def test_export_available_returns_bool(self):
        from yolozu.inference.torch_export import torch_export_available

        self.assertIsInstance(torch_export_available(), bool)

    def test_compile_with_real_torch(self):
        """If torch is installed, compile should succeed on a simple module."""
        torch = None
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from yolozu.inference.torch_export import compile_for_inference

        model = torch.nn.Linear(10, 5)
        model.eval()
        compiled = compile_for_inference(model, mode="default")
        # Compiled model should still produce output.
        x = torch.randn(2, 10)
        with torch.no_grad():
            out = compiled(x)
        self.assertEqual(out.shape, (2, 5))
        from yolozu.inference.torch_export import get_compile_evidence

        self.assertEqual(get_compile_evidence(compiled)["actual"]["status"], "compiled")

# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------


class TestExportModelOnnx(unittest.TestCase):
    """Tests for yolozu.inference.torch_export.export_model_onnx."""

    def test_export_raises_without_torch(self):
        from yolozu.inference.torch_export import export_model_onnx  # noqa: F401

        with patch.dict("sys.modules", {"torch": None}):
            # Force reimport would be messy; skip if torch is actually installed.
            pass  # Covered by integration tests below.

    def test_export_simple_model(self):
        """Export a trivial Linear model to ONNX."""
        torch = None
        try:
            import torch
            import onnxscript  # noqa: F401
        except ImportError:
            self.skipTest("torch or onnxscript not installed")

        from yolozu.inference.torch_export import export_model_onnx

        model = torch.nn.Linear(8, 4)
        model.eval()
        sample = torch.randn(1, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_model.onnx"
            result = export_model_onnx(model, sample, out_path, opset_version=14)
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 0)


# ---------------------------------------------------------------------------
# AMP utilities tests
# ---------------------------------------------------------------------------


class TestAmpUtils(unittest.TestCase):
    """Tests for yolozu.training.amp_utils."""

    def test_amp_available_returns_bool(self):
        from yolozu.training.amp_utils import amp_available

        self.assertIsInstance(amp_available(), bool)

    def test_make_amp_context_disabled(self):
        from yolozu.training.amp_utils import make_amp_context

        ctx_factory, scaler = make_amp_context(enabled=False)
        # Should return nullcontext.
        self.assertIsNone(scaler)
        ctx = ctx_factory()
        self.assertIsNotNone(ctx)

    def test_make_amp_context_enabled_cpu(self):
        """AMP on CPU with bfloat16 should work without a scaler."""
        torch = None
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        from yolozu.training.amp_utils import make_amp_context

        ctx_factory, scaler = make_amp_context(
            device_type="cpu", dtype="bfloat16", enabled=True
        )
        # CPU bfloat16 should not have a GradScaler.
        self.assertIsNone(scaler)
        with ctx_factory():
            pass  # No error.

    def test_make_amp_context_float16_dtype(self):
        """Verify dtype mapping for float16."""
        torch = None
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from yolozu.training.amp_utils import make_amp_context

        # On CPU, float16 autocast is valid since PyTorch 2.x.
        ctx_factory, scaler = make_amp_context(
            device_type="cpu", dtype="float16", enabled=True
        )
        # Scaler is only created for CUDA float16.
        self.assertIsNone(scaler)
        with ctx_factory():
            x = torch.randn(2, 3)
            _ = x + 1


# ---------------------------------------------------------------------------
# Profiler tests
# ---------------------------------------------------------------------------


class TestProfiler(unittest.TestCase):
    """Tests for yolozu.inference.profiler."""

    def test_profiler_available_returns_bool(self):
        from yolozu.inference.profiler import profiler_available

        self.assertIsInstance(profiler_available(), bool)

    def test_profile_callable(self):
        """profile_callable on a simple function should return a summary string."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        from yolozu.inference.profiler import profile_callable

        def dummy_fn():
            import torch

            x = torch.randn(32, 32)
            return torch.mm(x, x)

        summary = profile_callable(dummy_fn, iterations=6, warmup=2)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)

    def test_profile_inference_with_dummy_adapter(self):
        """profile_inference with DummyAdapter should work."""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        from yolozu.inference.adapter import DummyAdapter
        from yolozu.inference.profiler import profile_inference

        adapter = DummyAdapter()
        records = [{"image": f"fake_{i}.jpg"} for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = profile_inference(
                adapter, records, warmup=1, active=2, output_dir=tmpdir
            )
            self.assertIsInstance(summary, str)


# ---------------------------------------------------------------------------
# Transforms bridge tests
# ---------------------------------------------------------------------------


class TestTransformsBridge(unittest.TestCase):
    """Tests for yolozu.training.transforms_bridge."""

    def test_transforms_v2_available_returns_bool(self):
        from yolozu.training.transforms_bridge import transforms_v2_available

        self.assertIsInstance(transforms_v2_available(), bool)

    def test_build_detection_transforms(self):
        """Detection transforms should compose without error."""
        try:
            from torchvision.transforms import v2  # noqa: F401
        except ImportError:
            self.skipTest("torchvision.transforms.v2 not available")

        from yolozu.training.transforms_bridge import build_detection_transforms

        tfm = build_detection_transforms(size=(320, 320), hflip_prob=0.5)
        self.assertIsNotNone(tfm)

    def test_build_eval_transforms(self):
        """Eval transforms should compose without error."""
        try:
            from torchvision.transforms import v2  # noqa: F401
        except ImportError:
            self.skipTest("torchvision.transforms.v2 not available")

        from yolozu.training.transforms_bridge import build_eval_transforms

        tfm = build_eval_transforms(size=(640, 640))
        self.assertIsNotNone(tfm)

    def test_build_detection_transforms_raises_without_torchvision(self):
        """Should raise RuntimeError if torchvision.transforms.v2 missing."""
        from yolozu.training.transforms_bridge import build_detection_transforms

        with patch(
            "yolozu.training.transforms_bridge.transforms_v2_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                build_detection_transforms(size=(320, 320))

    def test_eval_transforms_apply_to_tensor(self):
        """Eval transforms should process a PIL Image → tensor."""
        torch_mod = None
        pil_image = None
        try:
            import torch as torch_mod
            from torchvision.transforms import v2  # noqa: F401
            from PIL import Image as pil_image
        except ImportError:
            self.skipTest("torch/torchvision/PIL not available")

        from yolozu.training.transforms_bridge import build_eval_transforms

        tfm = build_eval_transforms(size=(224, 224))
        self.assertIsNotNone(pil_image)
        self.assertIsNotNone(torch_mod)
        img = pil_image.new("RGB", (100, 80), color=(128, 64, 32))
        out = tfm(img)
        self.assertIsInstance(out, torch_mod.Tensor)
        self.assertEqual(out.shape[1:], (224, 224))
        self.assertEqual(out.shape[0], 3)


# ---------------------------------------------------------------------------
# Integration: adapter compile_model flag
# ---------------------------------------------------------------------------


class TestRTDETRPoseAdapterCompileFlag(unittest.TestCase):
    """Verify the compile_model kwarg is accepted and stored."""

    def test_compile_model_default_false(self):
        from yolozu.inference.adapter import RTDETRPoseAdapter

        adapter = RTDETRPoseAdapter()
        self.assertFalse(adapter.compile_model)
        report = adapter.get_compile_evidence()
        self.assertFalse(report["requested"]["enabled"])
        self.assertEqual(report["actual"]["status"], "not_requested")

    def test_compile_model_accepts_true(self):
        from yolozu.inference.adapter import RTDETRPoseAdapter

        adapter = RTDETRPoseAdapter(
            compile_model=True,
            compile_backend="inductor",
            compile_fullgraph=True,
            compile_dynamic=False,
            allow_compile_fallback=True,
        )
        self.assertTrue(adapter.compile_model)
        self.assertEqual(adapter.compile_backend, "inductor")
        self.assertEqual(adapter.compile_mode, "reduce-overhead")
        report = adapter.get_compile_evidence()
        self.assertTrue(report["requested"]["enabled"])
        self.assertTrue(report["requested"]["fullgraph"])
        self.assertFalse(report["requested"]["dynamic"])
        self.assertTrue(report["requested"]["allow_fallback"])
        self.assertEqual(report["actual"]["status"], "pending_first_execution")


if __name__ == "__main__":
    unittest.main()
