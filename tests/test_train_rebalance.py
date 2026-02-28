import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rtdetr_pose"))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from rtdetr_pose.train_rebalance import build_weighted_sampler


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainRebalance(unittest.TestCase):
    def test_build_weighted_sampler_distributed(self):
        records = [
            {"labels": [{"class_id": 0}]},
            {"labels": [{"class_id": 0}]},
            {"labels": [{"class_id": 1}]},
            {"labels": [{"class_id": 2}]},
        ]
        sampler, report = build_weighted_sampler(
            records,
            num_classes=3,
            strategy="class_balanced",
            gamma=1.0,
            min_weight=0.25,
            max_weight=4.0,
            aggregate="max",
            seed=123,
            distributed=True,
            world_size=2,
            rank=0,
        )
        self.assertIsNotNone(sampler)
        self.assertIsNotNone(report)
        self.assertTrue(hasattr(sampler, "set_epoch"))
        idx0 = list(iter(sampler))
        sampler.set_epoch(1)
        idx1 = list(iter(sampler))
        self.assertEqual(len(idx0), len(idx1))
        self.assertTrue(all(0 <= int(i) < len(records) for i in idx0))


if __name__ == "__main__":
    unittest.main()
