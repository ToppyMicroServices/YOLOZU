from __future__ import annotations

from typing import Sequence

try:
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    Tensor = None  # type: ignore[assignment,misc]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]

from ...backbone_interface import BaseBackbone
from .blocks import CSPBlock, ConvNormAct, SPPF
from .registry import register_backbone


class _BaseCSPBackbone(BaseBackbone):
    def _forward_impl(self, x: Tensor) -> list[Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = self._forward_impl(x)
        self.validate_contract(x, outputs)
        return outputs


class CSPResNetBackbone(_BaseCSPBackbone):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] = (1, 2, 2),
        use_sppf: bool = True,
        activation: str = "silu",
        norm: str = "bn",
    ):
        super().__init__()
        c3, c4, c5 = (int(stage_channels[0]), int(stage_channels[1]), int(stage_channels[2]))
        b3, b4, b5 = (int(stage_blocks[0]), int(stage_blocks[1]), int(stage_blocks[2]))
        self._out_channels = (c3, c4, c5)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(stem_channels, stem_channels * 2, kernel_size=3, padding=1, activation=activation, norm=norm),
        )
        in_ch = stem_channels * 2
        self.stage3 = nn.Sequential(
            ConvNormAct(in_ch, c3, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c3, c3, num_blocks=b3, activation=activation, norm=norm),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c3, c4, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c4, c4, num_blocks=b4, activation=activation, norm=norm),
        )
        self.stage5 = nn.Sequential(
            ConvNormAct(c4, c5, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c5, c5, num_blocks=b5, activation=activation, norm=norm),
        )
        self.sppf = SPPF(c5, c5, activation=activation, norm=norm) if bool(use_sppf) else None

    @property
    def out_channels(self) -> Sequence[int]:
        return self._out_channels

    def _forward_impl(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        if self.sppf is not None:
            p5 = self.sppf(p5)
        return [p3, p4, p5]


class CSPDarknetBackbone(_BaseCSPBackbone):
    """CSPDarknet family backbone with P3/P4/P5 outputs."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        width_mult: float = 0.5,
        depth_mult: float = 0.5,
        activation: str = "silu",
        norm: str = "bn",
        use_sppf: bool = True,
    ):
        super().__init__()

        def _cw(v: int) -> int:
            return max(8, int(v * float(width_mult)))

        def _cd(v: int) -> int:
            return max(1, int(round(v * float(depth_mult))))

        stem_c = _cw(64)
        c3, c4, c5 = _cw(128), _cw(256), _cw(512)
        self._out_channels = (c3, c4, c5)

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_c // 2, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(stem_c // 2, stem_c, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stem_c, c3, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c3, c3, num_blocks=_cd(3), expansion=0.5, activation=activation, norm=norm),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c3, c4, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c4, c4, num_blocks=_cd(6), expansion=0.5, activation=activation, norm=norm),
        )
        self.stage5 = nn.Sequential(
            ConvNormAct(c4, c5, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            CSPBlock(c5, c5, num_blocks=_cd(6), expansion=0.5, activation=activation, norm=norm),
        )
        self.sppf = SPPF(c5, c5, activation=activation, norm=norm) if bool(use_sppf) else None

    @property
    def out_channels(self) -> Sequence[int]:
        return self._out_channels

    def _forward_impl(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        if self.sppf is not None:
            p5 = self.sppf(p5)
        return [p3, p4, p5]


class TinyCNNBackbone(_BaseCSPBackbone):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_blocks: Sequence[int] | None = None,
        use_sppf: bool | None = None,
        activation: str = "silu",
        norm: str = "bn",
    ):
        super().__init__()
        _ = stage_blocks
        _ = use_sppf
        c3, c4, c5 = int(stage_channels[0]), int(stage_channels[1]), int(stage_channels[2])
        self._out_channels = (c3, c4, c5)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stem_channels, c3, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(c3, c3, kernel_size=3, padding=1, activation=activation, norm=norm),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(c3, c4, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(c4, c4, kernel_size=3, padding=1, activation=activation, norm=norm),
        )
        self.stage5 = nn.Sequential(
            ConvNormAct(c4, c5, kernel_size=3, stride=2, padding=1, activation=activation, norm=norm),
            ConvNormAct(c5, c5, kernel_size=3, padding=1, activation=activation, norm=norm),
        )

    @property
    def out_channels(self) -> Sequence[int]:
        return self._out_channels

    def _forward_impl(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return [p3, p4, p5]


@register_backbone("cspresnet")
@register_backbone("csp_resnet")
def _build_cspresnet(**kwargs) -> BaseBackbone:
    return CSPResNetBackbone(**kwargs)


@register_backbone("cspdarknet_s")
def _build_cspdarknet_s(**kwargs) -> BaseBackbone:
    kwargs = dict(kwargs)
    kwargs.setdefault("width_mult", 0.5)
    kwargs.setdefault("depth_mult", 0.5)
    return CSPDarknetBackbone(**kwargs)


@register_backbone("tiny_cnn")
@register_backbone("tinycnn")
@register_backbone("simple_cnn")
def _build_tiny(**kwargs) -> BaseBackbone:
    return TinyCNNBackbone(**kwargs)
