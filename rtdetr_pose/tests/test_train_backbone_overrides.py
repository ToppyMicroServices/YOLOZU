import argparse
import unittest

from rtdetr_pose.config import ModelConfig
from rtdetr_pose.train_backbone_overrides import apply_backbone_overrides


class TestTrainBackboneOverrides(unittest.TestCase):
    def test_apply_overrides_existing_model_cfg(self):
        cfg = ModelConfig()
        args = argparse.Namespace(
            backbone_name="cspdarknet_s",
            backbone_norm="gn",
            backbone_args='{"width_mult": 0.5, "depth_mult": 0.34}',
            num_classes=80,
            num_keypoints=0,
            enable_mim=False,
            depth_mode="none",
            depth_dropout=0.0,
            hidden_dim=256,
            num_queries=300,
            use_uncertainty=False,
        )

        updated, summary = apply_backbone_overrides(cfg, args=args)
        self.assertIsNotNone(updated)
        self.assertIsNotNone(summary)
        assert updated is not None
        self.assertEqual(updated.backbone_name, "cspdarknet_s")
        self.assertEqual(updated.backbone_norm, "gn")
        self.assertEqual(updated.backbone.get("name"), "cspdarknet_s")
        self.assertEqual(updated.backbone.get("norm"), "gn")
        self.assertAlmostEqual(float(updated.backbone["args"]["width_mult"]), 0.5, places=6)

    def test_create_model_cfg_when_missing(self):
        args = argparse.Namespace(
            backbone_name="tiny_cnn",
            backbone_norm=None,
            backbone_args='{"stage_channels": [32, 64, 128]}',
            num_classes=5,
            num_keypoints=4,
            enable_mim=True,
            depth_mode="sidecar",
            depth_dropout=0.1,
            hidden_dim=128,
            num_queries=100,
            use_uncertainty=True,
        )

        updated, summary = apply_backbone_overrides(None, args=args)
        self.assertIsNotNone(updated)
        self.assertIsNotNone(summary)
        assert updated is not None
        self.assertEqual(updated.num_classes, 5)
        self.assertEqual(updated.num_keypoints, 4)
        self.assertEqual(updated.backbone_name, "tiny_cnn")
        self.assertIn("stage_channels", updated.backbone_kwargs)


if __name__ == "__main__":
    unittest.main()
