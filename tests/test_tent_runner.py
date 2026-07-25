import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from yolozu.tta import TentConfig, TentRunner
from yolozu.tta.tent import _deterministic_weak_photometric_view


@unittest.skipIf(torch is None, "torch not installed")
class TestTentRunner(unittest.TestCase):
    def test_tent_runner_step(self):
        model = nn.Sequential(nn.Linear(4, 3))
        runner = TentRunner(model, config=TentConfig(lr=1e-3))
        batch = torch.randn(2, 4)
        out = runner.adapt_step(batch)
        self.assertIn("loss_entropy", out)
        log = runner.maybe_log()
        self.assertIsInstance(log, dict)
        self.assertGreater(log.get("updated_param_count", 0), 0)

    def test_aux_target_is_recomputed_from_current_batch_without_persistence(self):
        class RecordingDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 3)
                self.pose = nn.Linear(4, 6)
                self.calls = []

            def forward(self, x):
                self.calls.append(
                    {
                        "training": bool(self.training),
                        "input": x.detach().clone(),
                    }
                )
                return {"logits": self.proj(x), "rot6d": self.pose(x)}

        model = RecordingDetector()
        runner = TentRunner(
            model,
            config=TentConfig(lr=1e-3, aux_pose_weight=1.0),
        )
        first = torch.full((2, 4), 0.2)
        second = torch.full((2, 4), 0.8)

        first_metrics = runner.adapt_step(first)
        second_metrics = runner.adapt_step(second)

        self.assertNotIn("_teacher_outputs", vars(runner))
        self.assertGreater(first_metrics["loss_aux_consistency"], 0.0)
        self.assertGreater(second_metrics["loss_aux_consistency"], 0.0)
        self.assertEqual(first_metrics["aux_target_current_batch"], 1.0)
        self.assertEqual(second_metrics["aux_target_current_batch"], 1.0)
        self.assertEqual(len(model.calls), 4)
        self.assertFalse(model.calls[0]["training"])
        self.assertTrue(model.calls[1]["training"])
        self.assertFalse(model.calls[2]["training"])
        self.assertTrue(model.calls[3]["training"])
        self.assertTrue(torch.equal(model.calls[0]["input"], first))
        self.assertTrue(torch.equal(model.calls[2]["input"], second))
        self.assertFalse(torch.equal(model.calls[2]["input"], first))
        self.assertEqual(model.calls[1]["input"].shape, first.shape)
        self.assertEqual(model.calls[3]["input"].shape, second.shape)

    def test_weak_view_is_deterministic_shape_preserving_and_non_mutating(self):
        batch = torch.tensor([[[[0.0, 0.5], [1.0, -0.5]]]], dtype=torch.float32)
        original = batch.clone()
        first = _deterministic_weak_photometric_view(batch)
        second = _deterministic_weak_photometric_view(batch)

        self.assertEqual(first.shape, batch.shape)
        self.assertEqual(first.dtype, batch.dtype)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(batch, original))
        self.assertGreater(float((first - batch).abs().mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
