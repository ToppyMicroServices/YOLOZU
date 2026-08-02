import sys
from pathlib import Path
import unittest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestTrainRuntimeDistillationWeights(unittest.TestCase):
    def test_preweighted_sdft_total_is_not_scaled_again(self):
        from rtdetr_pose.train_runtime import combine_preweighted_distillation_losses
        from yolozu.sdft import SdftConfig, compute_sdft_loss

        student = {"bbox": torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])}
        teacher = {"bbox": torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])}
        supervised = torch.tensor(3.0)

        base, _ = compute_sdft_loss(student, teacher, SdftConfig(weight=1.0, keys=("bbox",)))
        half, _ = compute_sdft_loss(student, teacher, SdftConfig(weight=0.5, keys=("bbox",)))
        double, _ = compute_sdft_loss(student, teacher, SdftConfig(weight=2.0, keys=("bbox",)))

        self.assertTrue(torch.allclose(half, base * 0.5))
        self.assertTrue(torch.allclose(double, base * 2.0))
        self.assertTrue(
            torch.allclose(
                combine_preweighted_distillation_losses(supervised, sdft_total=half),
                supervised + (base * 0.5),
            )
        )
        self.assertTrue(
            torch.allclose(
                combine_preweighted_distillation_losses(supervised, sdft_total=double),
                supervised + (base * 2.0),
            )
        )

    def test_preweighted_sdft_and_derpp_totals_are_additive(self):
        from rtdetr_pose.train_runtime import combine_preweighted_distillation_losses

        total = combine_preweighted_distillation_losses(
            torch.tensor(3.0),
            sdft_total=torch.tensor(0.5),
            derpp_total=torch.tensor(0.25),
        )
        self.assertTrue(torch.allclose(total, torch.tensor(3.75)))


if __name__ == "__main__":
    unittest.main()
