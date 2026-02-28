import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rtdetr_pose"))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from rtdetr_pose.backbone_interface import BaseBackbone
except Exception:  # pragma: no cover
    BaseBackbone = object


@unittest.skipIf(torch is None or BaseBackbone is object, "torch not installed")
class TestBackboneInterface(unittest.TestCase):
    class _DummyBackbone(BaseBackbone):
        @property
        def out_channels(self):
            return (8, 16, 32)

        def forward(self, x):
            return [x, x, x]

    def test_validate_contract_accepts_floor_or_ceil(self):
        bb = self._DummyBackbone()
        x = torch.zeros((1, 3, 255, 255), dtype=torch.float32)
        feats = [
            torch.zeros((1, 8, 32, 32), dtype=torch.float32),
            torch.zeros((1, 16, 16, 16), dtype=torch.float32),
            torch.zeros((1, 32, 8, 8), dtype=torch.float32),
        ]
        bb.validate_contract(x, feats)

    def test_validate_contract_rejects_bad_shape(self):
        bb = self._DummyBackbone()
        x = torch.zeros((1, 3, 255, 255), dtype=torch.float32)
        feats = [
            torch.zeros((1, 8, 40, 40), dtype=torch.float32),
            torch.zeros((1, 16, 16, 16), dtype=torch.float32),
            torch.zeros((1, 32, 8, 8), dtype=torch.float32),
        ]
        with self.assertRaises(ValueError):
            bb.validate_contract(x, feats)


if __name__ == "__main__":
    unittest.main()
