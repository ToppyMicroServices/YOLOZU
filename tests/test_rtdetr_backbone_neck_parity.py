from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rtdetr_pose"))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from rtdetr_pose.config import ModelConfig
from rtdetr_pose.factory import build_model


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    "name,expect_torchvision",
    [
        ("cspresnet", False),
        ("cspdarknet_s", False),
        ("tiny_cnn", False),
        ("resnet50", True),
        ("convnext_tiny", True),
    ],
)
def test_rtdetr_backbone_neck_encoder_contract(name, expect_torchvision):
    if expect_torchvision:
        pytest.importorskip("torchvision")

    cfg = ModelConfig(
        num_classes=3,
        hidden_dim=96,
        num_queries=20,
        num_encoder_layers=1,
        num_decoder_layers=1,
        nhead=4,
        backbone={"name": name, "norm": "bn", "args": {}},
        projector={"d_model": 96},
    )
    model = build_model(cfg).eval()
    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        feats = model.backbone(x)
        assert len(feats) == 3
        assert [int(f.shape[1]) for f in feats] == list(model.backbone.out_channels)
        assert [128 // int(f.shape[-1]) for f in feats] == [8, 16, 32]

        pos = []
        for feat in feats:
            height, width = int(feat.shape[-2]), int(feat.shape[-1])
            pos.append(model.position(height, width, feat.device, feat.dtype))

        memory, fused = model.encoder(feats, pos)
        out = model(x)

    assert len(fused) == 3
    assert [int(f.shape[1]) for f in fused] == [96, 96, 96]
    assert [int(f.shape[-2]) for f in fused] == [16, 8, 4]
    assert [int(f.shape[-1]) for f in fused] == [16, 8, 4]
    assert tuple(memory.shape) == (1, 16 * 16 + 8 * 8 + 4 * 4, 96)
    assert tuple(out["logits"].shape) == (1, 20, 4)
    assert tuple(out["bbox"].shape) == (1, 20, 4)
