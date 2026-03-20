"""Tests for masked reconstruction branch with entropy loss and geometric consistency."""
import unittest
import sys
from pathlib import Path

# Add rtdetr_pose to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestMIMReconstruction(unittest.TestCase):
    def test_render_teacher_forward(self):
        """Test RenderTeacher produces correct output shape."""
        from rtdetr_pose.model import RenderTeacher
        
        batch_size = 2
        hidden_dim = 256
        h, w = 32, 32
        in_channels = 2  # mask + normalized depth
        
        teacher = RenderTeacher(hidden_dim=hidden_dim, in_channels=in_channels)
        geom_input = torch.randn(batch_size, in_channels, h, w)
        
        output = teacher(geom_input)
        
        self.assertEqual(output.shape, (batch_size, hidden_dim, h, w))
    
    def test_decoder_mim_forward(self):
        """Test DecoderMIM produces correct output shape."""
        from rtdetr_pose.model import DecoderMIM
        
        batch_size = 2
        hidden_dim = 256
        h, w = 32, 32
        
        decoder = DecoderMIM(hidden_dim=hidden_dim)
        masked_features = torch.randn(batch_size, hidden_dim, h, w)
        
        output = decoder(masked_features)
        
        self.assertEqual(output.shape, (batch_size, hidden_dim, h, w))
    
    def test_model_with_mim_enabled(self):
        """Test RTDETRPose with MIM enabled."""
        from rtdetr_pose.model import RTDETRPose
        
        model = RTDETRPose(
            num_classes=80,
            hidden_dim=256,
            num_queries=100,
            enable_mim=True,
        )
        
        self.assertIsNotNone(model.render_teacher)
        self.assertIsNotNone(model.decoder_mim)
        self.assertTrue(model.enable_mim)
    
    def test_model_without_mim(self):
        """Test RTDETRPose with MIM disabled."""
        from rtdetr_pose.model import RTDETRPose
        
        model = RTDETRPose(
            num_classes=80,
            hidden_dim=256,
            num_queries=100,
            enable_mim=False,
        )
        
        self.assertIsNone(model.render_teacher)
        self.assertIsNone(model.decoder_mim)
        self.assertFalse(model.enable_mim)
    
    def test_forward_with_mim(self):
        """Test forward pass with MIM branch."""
        from rtdetr_pose.model import RTDETRPose
        
        batch_size = 2
        h, w = 128, 128
        
        model = RTDETRPose(
            num_classes=80,
            hidden_dim=64,
            num_queries=50,
            enable_mim=True,
            stem_channels=16,
            backbone_channels=(32, 64, 64),
        )
        model.eval()
        
        # Input image
        x = torch.randn(batch_size, 3, h, w)
        
        # Geometry input (mask + normalized depth)
        geom_input = torch.randn(batch_size, 2, h // 4, w // 4)
        
        # Feature mask
        feature_mask = torch.rand(h // 4, w // 4) < 0.6
        
        # Forward with MIM
        with torch.no_grad():
            outputs = model(x, geom_input=geom_input, feature_mask=feature_mask, return_mim=True)
        
        # Check standard outputs
        self.assertIn("logits", outputs)
        self.assertIn("bbox", outputs)
        self.assertIn("log_z", outputs)
        self.assertIn("rot6d", outputs)
        self.assertIn("offsets", outputs)
        self.assertIn("k_delta", outputs)
        
        # Check MIM outputs
        self.assertIn("mim", outputs)
        mim_out = outputs["mim"]
        self.assertIn("recon_feat", mim_out)
        self.assertIn("teacher_feat", mim_out)
        self.assertIn("neck_feat", mim_out)
        self.assertIn("mask", mim_out)
        self.assertIn("entropy", mim_out)
    
    def test_forward_without_mim(self):
        """Test forward pass without MIM branch."""
        from rtdetr_pose.model import RTDETRPose
        
        batch_size = 2
        h, w = 128, 128
        
        model = RTDETRPose(
            num_classes=80,
            hidden_dim=64,
            num_queries=50,
            enable_mim=False,
            stem_channels=16,
            backbone_channels=(32, 64, 64),
        )
        model.eval()
        
        x = torch.randn(batch_size, 3, h, w)
        
        with torch.no_grad():
            outputs = model(x, return_mim=False)
        
        # Check standard outputs
        self.assertIn("logits", outputs)
        self.assertIn("bbox", outputs)
        
        # MIM should not be in outputs
        self.assertNotIn("mim", outputs)
    
    def test_mim_reconstruction_loss(self):
        """Test MIM reconstruction loss function."""
        from rtdetr_pose.losses import mim_reconstruction_loss
        
        batch_size = 2
        hidden_dim = 256
        h, w = 32, 32
        
        recon_feat = torch.randn(batch_size, hidden_dim, h, w)
        teacher_feat = torch.randn(batch_size, hidden_dim, h, w)
        mask = torch.rand(h, w) < 0.6
        
        # With mask
        loss_masked = mim_reconstruction_loss(recon_feat, teacher_feat, mask=mask)
        self.assertEqual(loss_masked.shape, torch.Size([]))
        self.assertGreaterEqual(loss_masked.item(), 0.0)
        
        # Without mask
        loss_full = mim_reconstruction_loss(recon_feat, teacher_feat, mask=None)
        self.assertEqual(loss_full.shape, torch.Size([]))
        self.assertGreaterEqual(loss_full.item(), 0.0)
    
    def test_mim_reconstruction_loss_no_teacher(self):
        """Test MIM loss with no teacher returns zero."""
        from rtdetr_pose.losses import mim_reconstruction_loss
        
        recon_feat = torch.randn(2, 256, 32, 32)
        loss = mim_reconstruction_loss(recon_feat, None, mask=None)
        
        self.assertAlmostEqual(loss.item(), 0.0)
    
    def test_entropy_loss(self):
        """Test entropy loss function."""
        from rtdetr_pose.losses import entropy_loss
        
        batch_size = 2
        num_queries = 100
        num_classes = 80
        
        logits = torch.randn(batch_size, num_queries, num_classes)
        loss = entropy_loss(logits)
        
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_losses_with_mim(self):
        """Test Losses class with MIM outputs."""
        from rtdetr_pose.losses import Losses
        
        losses_module = Losses(weights={"mim": 0.1, "entropy": 0.01})
        
        # Mock outputs with MIM
        outputs = {
            "logits": torch.randn(2, 100, 80),
            "bbox": torch.randn(2, 100, 4),
            "log_z": torch.randn(2, 100),
            "rot6d": torch.randn(2, 100, 6),
            "mim": {
                "recon_feat": torch.randn(2, 256, 32, 32),
                "teacher_feat": torch.randn(2, 256, 32, 32),
                "mask": torch.rand(32, 32) < 0.6,
                "entropy": torch.tensor(1.5),
            }
        }
        
        # Mock targets
        targets = {
            "labels": torch.randint(0, 80, (2, 100)),
            "bbox": torch.randn(2, 100, 4),
            "z_gt": torch.rand(2, 100) * 5.0 + 0.5,
            "R_gt": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 100, -1, -1),
        }
        
        result = losses_module(outputs, targets)
        
        self.assertIn("loss", result)
        self.assertIn("loss_mim", result)
        self.assertIn("loss_entropy", result)
        self.assertGreater(result["loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
