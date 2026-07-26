"""Tests for multi-task TTA extensions (tent aux losses, presets, config)."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from yolozu.tta.config import TTTConfig
from yolozu.tta.presets import (
    PRESETS,
    _choose_default_preset_id,
)


class TestTTTConfigMultitask(unittest.TestCase):
    def test_default_aux_weights_zero(self):
        cfg = TTTConfig()
        self.assertEqual(cfg.aux_pose_weight, 0.0)
        self.assertEqual(cfg.aux_keypoints_weight, 0.0)
        self.assertEqual(cfg.aux_depth_weight, 0.0)
        self.assertEqual(cfg.aux_seg_weight, 0.0)
        self.assertEqual(cfg.aux_temperature, 1.0)
        self.assertIsNone(cfg.sdft_task)

    def test_custom_aux_weights(self):
        cfg = TTTConfig(aux_pose_weight=0.5, aux_seg_weight=0.3, sdft_task="pose")
        self.assertEqual(cfg.aux_pose_weight, 0.5)
        self.assertEqual(cfg.aux_seg_weight, 0.3)
        self.assertEqual(cfg.sdft_task, "pose")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresetsMultitask(unittest.TestCase):
    def test_pose_safe_exists(self):
        p = PRESETS["pose_safe"]
        self.assertEqual(p.method, "tent")
        self.assertEqual(p.steps, 3)
        self.assertEqual(p.lr, 5e-5)

    def test_keypoints_safe_exists(self):
        p = PRESETS["keypoints_safe"]
        self.assertEqual(p.method, "tent")
        self.assertEqual(p.steps, 2)

    def test_depth_safe_exists(self):
        p = PRESETS["depth_safe"]
        self.assertEqual(p.method, "tent")
        self.assertEqual(p.update_filter, "norm_only")

    def test_seg_safe_exists(self):
        p = PRESETS["seg_safe"]
        self.assertEqual(p.method, "tent")
        self.assertEqual(p.update_filter, "norm_only")

    def test_pose_mim_exists(self):
        p = PRESETS["pose_mim"]
        self.assertEqual(p.method, "mim")
        self.assertEqual(p.update_filter, "adapter_only")


class TestChooseDefaultPresetId(unittest.TestCase):
    def test_pose_tent(self):
        args = SimpleNamespace(
            ttt_method="tent", ttt_update_filter="all", ttt_sdft_task="pose"
        )
        self.assertEqual(_choose_default_preset_id(args), "pose_safe")

    def test_keypoints_tent(self):
        args = SimpleNamespace(
            ttt_method="tent", ttt_update_filter="all", ttt_sdft_task="keypoints"
        )
        self.assertEqual(_choose_default_preset_id(args), "keypoints_safe")

    def test_depth_tent(self):
        args = SimpleNamespace(
            ttt_method="tent", ttt_update_filter="all", ttt_sdft_task="depth"
        )
        self.assertEqual(_choose_default_preset_id(args), "depth_safe")

    def test_seg_tent(self):
        args = SimpleNamespace(
            ttt_method="tent", ttt_update_filter="all", ttt_sdft_task="seg"
        )
        self.assertEqual(_choose_default_preset_id(args), "seg_safe")

    def test_pose_mim(self):
        args = SimpleNamespace(
            ttt_method="mim", ttt_update_filter="all", ttt_sdft_task="pose"
        )
        self.assertEqual(_choose_default_preset_id(args), "pose_mim")

    def test_no_task_default(self):
        args = SimpleNamespace(
            ttt_method="tent", ttt_update_filter="all", ttt_sdft_task=""
        )
        self.assertEqual(_choose_default_preset_id(args), "safe")

    def test_no_task_attr(self):
        args = SimpleNamespace(ttt_method="tent", ttt_update_filter="all")
        self.assertEqual(_choose_default_preset_id(args), "safe")


# ---------------------------------------------------------------------------
# Tent aux consistency (requires torch)
# ---------------------------------------------------------------------------


if torch is not None and nn is not None:
    from yolozu.tta.tent import (
        TentConfig,
        TentRunner,
        _aux_consistency_loss,
        _extract_outputs,
    )

    class TestExtractOutputs(unittest.TestCase):
        def test_dict_passthrough(self):
            d = {"logits": torch.randn(2, 3), "rot6d": torch.randn(2, 6)}
            self.assertIs(_extract_outputs(d), d)

        def test_tensor_wraps(self):
            t = torch.randn(2, 5)
            result = _extract_outputs(t)
            self.assertIn("logits", result)

    class TestAuxConsistencyLoss(unittest.TestCase):
        def test_no_weights_returns_none(self):
            outputs = {"logits": torch.randn(2, 5), "rot6d": torch.randn(2, 6)}
            teacher = {"logits": torch.randn(2, 5), "rot6d": torch.randn(2, 6)}
            total, parts = _aux_consistency_loss(outputs, teacher)
            self.assertIsNone(total)
            self.assertEqual(len(parts), 0)

        def test_pose_weight_contributes(self):
            outputs = {
                "logits": torch.randn(2, 5),
                "rot6d": torch.randn(2, 3, 6),
                "log_z": torch.randn(2, 3, 1),
            }
            teacher = {k: torch.randn_like(v) for k, v in outputs.items()}
            total, parts = _aux_consistency_loss(outputs, teacher, pose_weight=1.0)
            self.assertIsNotNone(total)
            self.assertIn("aux_rot6d", parts)
            self.assertGreater(float(total), 0.0)

        def test_keypoints_weight(self):
            outputs = {
                "logits": torch.randn(2, 5),
                "keypoints": torch.randn(2, 5, 17, 2),
            }
            teacher = {k: torch.randn_like(v) for k, v in outputs.items()}
            total, parts = _aux_consistency_loss(outputs, teacher, keypoints_weight=1.0)
            self.assertIsNotNone(total)
            self.assertIn("aux_keypoints", parts)

        def test_depth_weight(self):
            outputs = {
                "logits": torch.randn(2, 5),
                "depth": torch.randn(2, 10),
            }
            teacher = {k: torch.randn_like(v) for k, v in outputs.items()}
            total, parts = _aux_consistency_loss(outputs, teacher, depth_weight=1.0)
            self.assertIsNotNone(total)
            self.assertIn("aux_depth", parts)

        def test_seg_weight(self):
            outputs = {
                "logits": torch.randn(2, 5),
                "seg_logits": torch.randn(2, 1, 8, 8),
            }
            teacher = {k: torch.randn_like(v) for k, v in outputs.items()}
            total, parts = _aux_consistency_loss(outputs, teacher, seg_weight=1.0)
            self.assertIsNotNone(total)
            self.assertIn("aux_seg_logits", parts)

        def test_missing_teacher_fails_closed_without_aux_loss(self):
            x = torch.randn(2, 3, 6, requires_grad=True)
            outputs = {"logits": torch.randn(2, 5), "rot6d": x}
            total, _ = _aux_consistency_loss(outputs, None, pose_weight=1.0)
            self.assertIsNone(total)

        def test_shape_mismatch_skipped(self):
            outputs = {"logits": torch.randn(2, 5), "rot6d": torch.randn(2, 3, 6)}
            teacher = {"logits": torch.randn(2, 5), "rot6d": torch.randn(3, 3, 6)}
            total, parts = _aux_consistency_loss(outputs, teacher, pose_weight=1.0)
            self.assertNotIn("aux_rot6d", parts)

    class TestTentRunnerMultitask(unittest.TestCase):
        def _make_model(self):
            """Simple model that returns multi-head outputs."""

            class MultiHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bn = nn.BatchNorm1d(4)
                    self.fc = nn.Linear(4, 5)
                    self.rot_head = nn.Linear(4, 6)

                def forward(self, x):
                    h = self.bn(x)
                    return {
                        "logits": self.fc(h),
                        "rot6d": self.rot_head(h),
                    }

            return MultiHead()

        def test_tent_runner_aux_pose(self):
            model = self._make_model()
            cfg = TentConfig(
                lr=1e-4,
                update_filter="all",
                aux_pose_weight=0.5,
            )
            runner = TentRunner(model, config=cfg)
            self.assertTrue(runner._has_aux)

            batch = torch.randn(2, 4)
            metrics = runner.adapt_step(batch)
            self.assertIn("loss_entropy", metrics)
            self.assertIn("aux_rot6d", metrics)

        def test_tent_runner_no_aux(self):
            model = self._make_model()
            cfg = TentConfig(lr=1e-4, update_filter="all")
            runner = TentRunner(model, config=cfg)
            self.assertFalse(runner._has_aux)

            batch = torch.randn(2, 4)
            metrics = runner.adapt_step(batch)
            self.assertIn("loss_entropy", metrics)
            self.assertNotIn("aux_rot6d", metrics)

        def test_tent_runner_has_no_persistent_teacher_state(self):
            model = self._make_model()
            cfg = TentConfig(lr=1e-4, update_filter="all", aux_pose_weight=1.0)
            runner = TentRunner(model, config=cfg)

            first = torch.full((2, 4), 0.2)
            second = torch.full((2, 4), 0.8)
            first_metrics = runner.adapt_step(first)
            second_metrics = runner.adapt_step(second)
            self.assertNotIn("_teacher_outputs", vars(runner))
            self.assertEqual(first_metrics["aux_target_current_batch"], 1.0)
            self.assertEqual(second_metrics["aux_target_current_batch"], 1.0)
            runner.reset()
            self.assertNotIn("_teacher_outputs", vars(runner))


if __name__ == "__main__":
    unittest.main()
