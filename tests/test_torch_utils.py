"""Tests for yolozu.training.torch_utils – PyTorch utility wrappers."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestAutoDevice(unittest.TestCase):
    """Tests for auto_device()."""

    def test_returns_torch_device(self):
        from yolozu.training.torch_utils import auto_device
        dev = auto_device()
        self.assertIsInstance(dev, torch.device)

    def test_cpu_always_available(self):
        from yolozu.training.torch_utils import auto_device
        dev = auto_device(prefer="cpu")
        self.assertEqual(dev.type, "cpu")

    def test_prefer_invalid_falls_back(self):
        from yolozu.training.torch_utils import auto_device
        # prefer a non-existent device → should fall back gracefully
        dev = auto_device(prefer="xpu:99")
        self.assertIn(dev.type, ("cpu", "cuda", "mps"))


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestSeedEverything(unittest.TestCase):
    """Tests for seed_everything()."""

    def test_returns_seed(self):
        from yolozu.training.torch_utils import seed_everything
        result = seed_everything(123)
        self.assertEqual(result, 123)

    def test_reproducible_random(self):
        import random
        from yolozu.training.torch_utils import seed_everything

        seed_everything(42)
        a = random.random()
        t1 = torch.randn(5)

        seed_everything(42)
        b = random.random()
        t2 = torch.randn(5)

        self.assertEqual(a, b)
        self.assertTrue(torch.equal(t1, t2))

    def test_default_seed(self):
        from yolozu.training.torch_utils import seed_everything
        result = seed_everything()
        self.assertEqual(result, 42)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestAmpInferenceContext(unittest.TestCase):
    """Tests for amp_inference_context()."""

    def test_context_manager_basic(self):
        from yolozu.training.torch_utils import amp_inference_context
        x = torch.randn(4, 4)
        with amp_inference_context("cpu", dtype=torch.bfloat16):
            y = x @ x.T
        # result should be computed (might be bf16 inside context)
        self.assertEqual(y.shape, (4, 4))

    def test_disabled(self):
        from yolozu.training.torch_utils import amp_inference_context
        x = torch.randn(4, 4)
        with amp_inference_context("cpu", enabled=False):
            y = x + 1.0
        self.assertEqual(y.dtype, torch.float32)

    def test_default_cpu_dtype(self):
        """Default dtype for CPU should be bfloat16."""
        from yolozu.training.torch_utils import amp_inference_context
        x = torch.randn(3, 3)
        with amp_inference_context("cpu"):
            y = x @ x.T
            # Inside autocast, matmul should produce bfloat16
            self.assertEqual(y.dtype, torch.bfloat16)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestCompileModel(unittest.TestCase):
    """Tests for compile_model()."""

    def test_compiles_simple_module(self):
        from yolozu.training.torch_utils import compile_model

        model = torch.nn.Linear(10, 5)
        compiled = compile_model(model, backend="eager")
        # Should return a compiled object (or wrapped callable)
        x = torch.randn(3, 10)
        out = compiled(x)
        self.assertEqual(out.shape, (3, 5))

    def test_fullgraph_option(self):
        from yolozu.training.torch_utils import compile_model

        model = torch.nn.Linear(10, 5)
        compiled = compile_model(model, backend="eager", fullgraph=True)
        x = torch.randn(2, 10)
        out = compiled(x)
        self.assertEqual(out.shape, (2, 5))


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestModelInfo(unittest.TestCase):
    """Tests for model_info()."""

    def test_linear_model(self):
        from yolozu.training.torch_utils import model_info

        model = torch.nn.Linear(10, 5)
        info = model_info(model)

        self.assertEqual(info["total_params"], 10 * 5 + 5)  # weights + bias
        self.assertEqual(info["trainable_params"], 55)
        self.assertEqual(info["frozen_params"], 0)
        self.assertIn("torch.float32", info["dtype_breakdown"])
        self.assertEqual(info["device"], "cpu")

    def test_frozen_params(self):
        from yolozu.training.torch_utils import model_info

        model = torch.nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad_(False)

        info = model_info(model)
        self.assertEqual(info["trainable_params"], 0)
        self.assertEqual(info["frozen_params"], 55)

    def test_sequential_model(self):
        from yolozu.training.torch_utils import model_info

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        info = model_info(model)
        # Linear(10,20): 10*20+20 = 220, Linear(20,5): 20*5+5 = 105
        self.assertEqual(info["total_params"], 220 + 105)
        self.assertEqual(info["num_buffers"], 0)

    def test_model_with_buffers(self):
        from yolozu.training.torch_utils import model_info

        model = torch.nn.BatchNorm1d(10)
        info = model_info(model)
        # BN has running_mean and running_var as buffers (10 each)
        # plus num_batches_tracked (1)
        self.assertEqual(info["num_buffers"], 3)
        self.assertEqual(info["total_buffer_elements"], 10 + 10 + 1)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestProfileCallable(unittest.TestCase):
    """Tests for profile_callable()."""

    def test_basic_profiling(self):
        from yolozu.training.torch_utils import profile_callable

        def matmul_fn():
            a = torch.randn(32, 32)
            return a @ a.T

        result = profile_callable(matmul_fn, warmup=1, active=2)
        self.assertIn("table", result)
        self.assertIn("events", result)
        self.assertIn("total_calls", result)
        self.assertIsInstance(result["table"], str)
        self.assertIsInstance(result["events"], list)
        self.assertGreater(result["total_calls"], 0)

    def test_with_args(self):
        from yolozu.training.torch_utils import profile_callable

        def add_fn(a, b):
            return a + b

        x = torch.randn(10)
        y = torch.randn(10)
        result = profile_callable(add_fn, args=(x, y), warmup=1, active=1)
        self.assertIn("table", result)

    def test_cpu_only_activities(self):
        from yolozu.training.torch_utils import profile_callable

        def fn():
            return torch.ones(5).sum()

        result = profile_callable(fn, warmup=1, active=1, activities=["cpu"])
        self.assertIn("events", result)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestConfigureMatmulPrecision(unittest.TestCase):
    """Tests for configure_matmul_precision()."""

    def test_returns_summary(self):
        from yolozu.training.torch_utils import configure_matmul_precision

        result = configure_matmul_precision("high")
        self.assertIn("matmul_precision", result)
        self.assertIn("set_precision_supported", result)
        self.assertIn("tf32_cuda_matmul", result)
        self.assertIn("tf32_cudnn", result)
        self.assertEqual(result["matmul_precision"], "high")

    def test_invalid_precision_raises(self):
        from yolozu.training.torch_utils import configure_matmul_precision

        with self.assertRaises(ValueError):
            configure_matmul_precision("low")

    def test_allow_tf32_argument(self):
        from yolozu.training.torch_utils import configure_matmul_precision

        result_true = configure_matmul_precision("high", allow_tf32=True)
        self.assertIn("tf32_cuda_matmul", result_true)
        self.assertIn("tf32_cudnn", result_true)

        result_false = configure_matmul_precision("medium", allow_tf32=False)
        self.assertEqual(result_false["matmul_precision"], "medium")


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestBackwardCompatShim(unittest.TestCase):
    """Ensure the backward-compat shim at yolozu.torch_utils works."""

    def test_shim_import(self):
        from yolozu.torch_utils import auto_device, seed_everything, model_info
        self.assertTrue(callable(auto_device))
        self.assertTrue(callable(seed_everything))
        self.assertTrue(callable(model_info))

    def test_shim_identity(self):
        from yolozu.torch_utils import compile_model as shim_fn
        from yolozu.training.torch_utils import compile_model as canonical_fn
        self.assertIs(shim_fn, canonical_fn)


if __name__ == "__main__":
    unittest.main()
