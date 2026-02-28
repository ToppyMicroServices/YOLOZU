import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from rtdetr_pose.train_rebalance import build_weighted_sampler


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainRebalance(unittest.TestCase):
    def test_class_balanced_sampler_upweights_rare_class(self):
        records = []
        for _ in range(9):
            records.append({"labels": [{"class_id": 0}]})
        records.append({"labels": [{"class_id": 1}]})

        sampler, report = build_weighted_sampler(
            records,
            num_classes=3,
            strategy="class_balanced",
            gamma=1.0,
            min_weight=0.25,
            max_weight=4.0,
            aggregate="max",
            seed=7,
        )

        self.assertIsNotNone(sampler)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.classes_with_labels, 2)
        self.assertEqual(report.instances_total, 10)
        self.assertGreater(report.max_weight, report.min_weight)

    def test_none_strategy_returns_none(self):
        sampler, report = build_weighted_sampler(
            [{"labels": [{"class_id": 0}]}],
            num_classes=1,
            strategy="none",
            gamma=1.0,
            min_weight=0.25,
            max_weight=4.0,
            aggregate="max",
            seed=0,
        )
        self.assertIsNone(sampler)
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
