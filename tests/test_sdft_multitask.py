"""Tests for multi-task SDFT loss functions (pose, keypoints, depth, seg)."""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.sdft import (
    POSE_KEYS,
    KEYPOINTS_KEYS,
    DEPTH_KEYS,
    SEG_KEYS,
    SdftConfig,
    compute_sdft_loss,
    _rot6d_geodesic_loss,
    _keypoints_loss,
    _depth_loss,
    _seg_bce_loss,
    _get_key_weight,
    _compute_key_loss,
    make_pose_sdft_config,
    make_keypoints_sdft_config,
    make_depth_sdft_config,
    make_seg_sdft_config,
    make_full_sdft_config,
)


class TestConstants(unittest.TestCase):
    def test_pose_keys(self):
        self.assertEqual(POSE_KEYS, ("rot6d", "log_z", "offsets", "k_delta"))

    def test_keypoints_keys(self):
        self.assertEqual(KEYPOINTS_KEYS, ("keypoints",))

    def test_depth_keys(self):
        self.assertEqual(DEPTH_KEYS, ("depth", "depth_map"))

    def test_seg_keys(self):
        self.assertEqual(SEG_KEYS, ("seg_logits", "mask_logits"))


class TestConfigValidation(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_negative_rot6d_weight(self):
        cfg = SdftConfig(rot6d_weight=-1.0, keys=("rot6d",))
        with self.assertRaises(ValueError):
            compute_sdft_loss(
                {"rot6d": torch.randn(1, 3, 6)},
                {"rot6d": torch.randn(1, 3, 6)},
                cfg,
            )

    @unittest.skipIf(torch is None, "torch not installed")
    def test_negative_keypoints_weight(self):
        cfg = SdftConfig(keypoints_weight=-1.0, keys=("keypoints",))
        with self.assertRaises(ValueError):
            compute_sdft_loss(
                {"keypoints": torch.randn(1, 3, 17, 2)},
                {"keypoints": torch.randn(1, 3, 17, 2)},
                cfg,
            )

    @unittest.skipIf(torch is None, "torch not installed")
    def test_negative_depth_weight(self):
        cfg = SdftConfig(depth_weight=-1.0, keys=("depth",))
        with self.assertRaises(ValueError):
            compute_sdft_loss(
                {"depth": torch.randn(1, 10)},
                {"depth": torch.randn(1, 10)},
                cfg,
            )

    @unittest.skipIf(torch is None, "torch not installed")
    def test_negative_seg_weight(self):
        cfg = SdftConfig(seg_weight=-1.0, keys=("seg_logits",))
        with self.assertRaises(ValueError):
            compute_sdft_loss(
                {"seg_logits": torch.randn(1, 1, 8, 8)},
                {"seg_logits": torch.randn(1, 1, 8, 8)},
                cfg,
            )


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


class TestRot6dLoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_zero_when_equal(self):
        x = torch.randn(2, 3, 6)
        loss = _rot6d_geodesic_loss(x, x)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_positive_when_different(self):
        a = torch.randn(2, 3, 6)
        b = torch.randn(2, 3, 6)
        loss = _rot6d_geodesic_loss(a, b)
        self.assertGreater(float(loss), 0.0)


class TestKeypointsLoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_zero_when_equal(self):
        kp = torch.randn(2, 5, 17, 2)
        loss = _keypoints_loss(kp, kp)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_accepts_flat_shape(self):
        kp_4d = torch.randn(2, 5, 17, 2)
        kp_3d = kp_4d.reshape(2, 5, 34)
        loss = _keypoints_loss(kp_3d, kp_3d)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_positive_when_different(self):
        a = torch.randn(2, 5, 17, 2)
        b = torch.randn(2, 5, 17, 2)
        loss = _keypoints_loss(a, b)
        self.assertGreater(float(loss), 0.0)


class TestDepthLoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_zero_when_equal_si(self):
        d = torch.randn(2, 10)
        loss = _depth_loss(d, d, scale_invariant=True)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_zero_when_equal_l1(self):
        d = torch.randn(2, 10)
        loss = _depth_loss(d, d, scale_invariant=False)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_shift_invariance(self):
        """Scale-invariant loss should be invariant to a global constant shift."""
        d = torch.randn(2, 10)
        shifted = d + 5.0
        loss = _depth_loss(d, shifted, scale_invariant=True)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_l1_not_shift_invariant(self):
        d = torch.randn(2, 10)
        shifted = d + 5.0
        loss = _depth_loss(d, shifted, scale_invariant=False)
        self.assertGreater(float(loss), 4.0)


class TestSegBCELoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_low_when_similar(self):
        x = torch.randn(2, 1, 8, 8) * 5.0  # strong logits
        loss_same = _seg_bce_loss(x, x)
        # BCE with same logits vs sigmoid(x) should be small
        self.assertLess(float(loss_same), 0.5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_higher_when_opposite(self):
        x = torch.randn(2, 1, 8, 8) * 5.0
        loss_opp = _seg_bce_loss(x, -x)
        loss_same = _seg_bce_loss(x, x)
        self.assertGreater(float(loss_opp), float(loss_same))


# ---------------------------------------------------------------------------
# _get_key_weight
# ---------------------------------------------------------------------------


class TestGetKeyWeight(unittest.TestCase):
    def test_known_keys(self):
        cfg = SdftConfig(rot6d_weight=2.0, log_z_weight=0.5, seg_weight=3.0)
        self.assertEqual(_get_key_weight(cfg, "rot6d"), 2.0)
        self.assertEqual(_get_key_weight(cfg, "log_z"), 0.5)
        self.assertEqual(_get_key_weight(cfg, "seg_logits"), 3.0)
        self.assertEqual(_get_key_weight(cfg, "mask_logits"), 3.0)

    def test_unknown_key_falls_back(self):
        cfg = SdftConfig(other_l1_weight=0.42)
        self.assertAlmostEqual(_get_key_weight(cfg, "foo_bar"), 0.42)


# ---------------------------------------------------------------------------
# _compute_key_loss dispatch
# ---------------------------------------------------------------------------


class TestComputeKeyLoss(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_logits_uses_kl(self):
        cfg = SdftConfig()
        s = torch.randn(2, 3, 5)
        loss = _compute_key_loss("logits", s, s, cfg)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_rot6d_dispatch(self):
        cfg = SdftConfig()
        s = torch.randn(2, 3, 6)
        loss = _compute_key_loss("rot6d", s, s, cfg)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_keypoints_dispatch(self):
        cfg = SdftConfig()
        s = torch.randn(2, 5, 17, 2)
        loss = _compute_key_loss("keypoints", s, s, cfg)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_depth_dispatch(self):
        cfg = SdftConfig()
        s = torch.randn(2, 10)
        loss = _compute_key_loss("depth", s, s, cfg)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_seg_dispatch(self):
        cfg = SdftConfig()
        s = torch.randn(2, 1, 8, 8) * 5.0
        loss = _compute_key_loss("seg_logits", s, s, cfg)
        self.assertLess(float(loss), 0.5)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_unknown_key_uses_l1(self):
        cfg = SdftConfig()
        s = torch.randn(2, 4)
        t = torch.randn(2, 4)
        loss = _compute_key_loss("whatever", s, t, cfg)
        expected = torch.nn.functional.l1_loss(s, t)
        self.assertAlmostEqual(float(loss), float(expected), places=5)


# ---------------------------------------------------------------------------
# compute_sdft_loss integration with pose outputs
# ---------------------------------------------------------------------------


class TestComputeSdftLossPose(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_all_pose_keys(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "rot6d": torch.randn(2, 3, 6),
            "log_z": torch.randn(2, 3, 1),
            "offsets": torch.randn(2, 3, 2),
            "k_delta": torch.randn(2, 4),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_pose_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft_rot6d", parts)
        self.assertIn("loss_sdft_log_z", parts)
        self.assertIn("loss_sdft_offsets", parts)
        self.assertIn("loss_sdft_k_delta", parts)
        self.assertIn("loss_sdft_logits", parts)
        self.assertIn("loss_sdft_bbox", parts)
        self.assertTrue(float(total) > 0)
        self.assertTrue(total.requires_grad is False)  # inputs had no grad

    @unittest.skipIf(torch is None, "torch not installed")
    def test_pose_missing_optional_keys(self):
        """Pose config with keys set but model missing some — should still work."""
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "rot6d": torch.randn(2, 3, 6),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_pose_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft_rot6d", parts)
        self.assertNotIn("loss_sdft_log_z", parts)
        self.assertGreater(float(total), 0)


class TestComputeSdftLossKeypoints(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_keypoints_end_to_end(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "keypoints": torch.randn(2, 3, 17, 2),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_keypoints_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft_keypoints", parts)
        self.assertGreater(float(total), 0)


class TestComputeSdftLossDepth(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_depth_end_to_end(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "depth": torch.randn(2, 10),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_depth_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft_depth", parts)
        self.assertGreater(float(total), 0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_depth_no_scale_invariant(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "depth": torch.randn(2, 10),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_depth_sdft_config(depth_scale_invariant=False)
        total, _ = compute_sdft_loss(student, teacher, cfg)
        self.assertGreater(float(total), 0)


class TestComputeSdftLossSeg(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_seg_end_to_end(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "seg_logits": torch.randn(2, 1, 8, 8),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_seg_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertIn("loss_sdft_seg_logits", parts)
        self.assertGreater(float(total), 0)


class TestMakeFullSdftConfig(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_full_config_all_heads(self):
        student = {
            "logits": torch.randn(2, 3, 5),
            "bbox": torch.randn(2, 3, 4),
            "rot6d": torch.randn(2, 3, 6),
            "log_z": torch.randn(2, 3, 1),
            "offsets": torch.randn(2, 3, 2),
            "k_delta": torch.randn(2, 4),
            "keypoints": torch.randn(2, 3, 17, 2),
            "depth": torch.randn(2, 10),
            "seg_logits": torch.randn(2, 1, 8, 8),
        }
        teacher = {k: torch.randn_like(v) for k, v in student.items()}
        cfg = make_full_sdft_config()
        total, parts = compute_sdft_loss(student, teacher, cfg)
        self.assertEqual(len(cfg.keys), 9)
        self.assertIn("loss_sdft_rot6d", parts)
        self.assertIn("loss_sdft_keypoints", parts)
        self.assertIn("loss_sdft_depth", parts)
        self.assertIn("loss_sdft_seg_logits", parts)


# ---------------------------------------------------------------------------
# Convenience constructor defaults
# ---------------------------------------------------------------------------


class TestConvenienceDefaults(unittest.TestCase):
    def test_pose_config_keys(self):
        cfg = make_pose_sdft_config()
        self.assertEqual(cfg.keys, ("logits", "bbox", "rot6d", "log_z", "offsets", "k_delta"))
        self.assertEqual(cfg.log_z_weight, 0.5)
        self.assertEqual(cfg.offsets_weight, 0.5)
        self.assertEqual(cfg.k_delta_weight, 0.3)

    def test_keypoints_config_keys(self):
        cfg = make_keypoints_sdft_config()
        self.assertEqual(cfg.keys, ("logits", "bbox", "keypoints"))

    def test_depth_config_keys(self):
        cfg = make_depth_sdft_config()
        self.assertEqual(cfg.keys, ("logits", "bbox", "depth"))
        self.assertTrue(cfg.depth_scale_invariant)

    def test_seg_config_keys(self):
        cfg = make_seg_sdft_config()
        self.assertEqual(cfg.keys, ("logits", "bbox", "seg_logits"))

    def test_full_config_keys(self):
        cfg = make_full_sdft_config()
        self.assertIn("rot6d", cfg.keys)
        self.assertIn("keypoints", cfg.keys)
        self.assertIn("depth", cfg.keys)
        self.assertIn("seg_logits", cfg.keys)


if __name__ == "__main__":
    unittest.main()
