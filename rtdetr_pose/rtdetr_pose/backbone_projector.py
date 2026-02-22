from __future__ import annotations

from typing import Iterable, Sequence

try:
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    Tensor = None  # type: ignore[assignment,misc]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]


class BackboneProjector(nn.Module):
    """Project backbone P3/P4/P5 channels to a unified d_model via 1x1 conv."""

    def __init__(self, in_channels: Sequence[int], d_model: int):
        super().__init__()
        channels = [int(c) for c in in_channels]
        if len(channels) != 3:
            raise ValueError("BackboneProjector expects three input channels (P3,P4,P5)")
        self.in_channels = tuple(channels)
        self.d_model = int(d_model)
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, self.d_model, kernel_size=1, stride=1, padding=0, bias=True) for c in channels]
        )

    def forward(self, features: Iterable[Tensor]) -> list[Tensor]:
        feats = list(features)
        if len(feats) != 3:
            raise ValueError(f"BackboneProjector expects 3 feature maps, got {len(feats)}")
        return [layer(feat) for layer, feat in zip(self.proj, feats)]
