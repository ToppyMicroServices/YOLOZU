from __future__ import annotations

from typing import Sequence

try:
    from torch import Tensor
except ImportError:  # pragma: no cover
    Tensor = None  # type: ignore[assignment,misc]

from ...backbone_interface import BaseBackbone
from .registry import register_backbone


class ResNet50Backbone(BaseBackbone):
    out_strides: Sequence[int] = (8, 16, 32)

    def __init__(self, **_: object):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as exc:
            raise RuntimeError("torchvision is required for resnet50 backbone") from exc

        net = tvm.resnet50(weights=None)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self._out_channels = (512, 1024, 2048)

    @property
    def out_channels(self) -> Sequence[int]:
        return self._out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        x_in = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        outputs = [p3, p4, p5]
        self.validate_contract(x=x_in, features=outputs)
        return outputs


class ConvNeXtTinyBackbone(BaseBackbone):
    out_strides: Sequence[int] = (8, 16, 32)

    def __init__(self, **_: object):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as exc:
            raise RuntimeError("torchvision is required for convnext_tiny backbone") from exc

        net = tvm.convnext_tiny(weights=None)
        self.features = net.features
        self._out_channels = (192, 384, 768)

    @property
    def out_channels(self) -> Sequence[int]:
        return self._out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        x_in = x
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        p3 = self.features[3](x)
        x = self.features[4](p3)
        p4 = self.features[5](x)
        x = self.features[6](p4)
        p5 = self.features[7](x)
        outputs = [p3, p4, p5]
        self.validate_contract(x=x_in, features=outputs)
        return outputs


@register_backbone("resnet50")
def _build_resnet50(**kwargs) -> BaseBackbone:
    return ResNet50Backbone(**kwargs)


@register_backbone("convnext_tiny")
def _build_convnext_tiny(**kwargs) -> BaseBackbone:
    return ConvNeXtTinyBackbone(**kwargs)
