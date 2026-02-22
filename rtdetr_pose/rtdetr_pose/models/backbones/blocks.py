from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    torch = None  # type: ignore[assignment]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]


def _normalize_activation_name(name: str) -> str:
    value = str(name or "silu").strip().lower().replace("-", "_")
    aliases = {
        "swish": "silu",
        "hard_swish": "hardswish",
        "leaky_relu": "leakyrelu",
    }
    return aliases.get(value, value)


def build_activation(name: str, *, inplace: bool = True) -> nn.Module:
    act = _normalize_activation_name(name)
    if act == "silu":
        return nn.SiLU(inplace=inplace)
    if act == "gelu":
        return nn.GELU()
    if act == "hardswish":
        return nn.Hardswish(inplace=inplace)
    if act == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    raise ValueError(f"unsupported activation: {name}")


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.view(1, -1, 1, 1)
        bias = bias.view(1, -1, 1, 1)
        return x * scale + bias


def build_norm(norm: str, num_channels: int) -> nn.Module:
    kind = str(norm or "bn").strip().lower()
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    if kind == "syncbn":
        return nn.SyncBatchNorm(num_channels)
    if kind == "frozenbn":
        return FrozenBatchNorm2d(num_channels)
    if kind == "gn":
        groups = 32
        while groups > 1 and num_channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"unsupported norm type: {norm}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        *,
        activation: str = "silu",
        norm: str = "bn",
        groups: int = 1,
        bias: Optional[bool] = None,
    ):
        super().__init__()
        if bias is None:
            bias = str(norm).lower() in {"frozenbn", "gn"}
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_activation(activation, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: float = 0.5,
        shortcut: bool = True,
        activation: str = "silu",
        norm: str = "bn",
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation, norm=norm)
        self.conv2 = ConvNormAct(hidden, out_channels, kernel_size=3, padding=1, activation=activation, norm=norm)
        self.shortcut = bool(shortcut and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        if self.shortcut:
            y = y + x
        return y


class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_blocks: int = 1,
        expansion: float = 0.5,
        activation: str = "silu",
        norm: str = "bn",
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation, norm=norm)
        self.conv2 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation, norm=norm)
        self.blocks = nn.Sequential(
            *[
                Bottleneck(hidden, hidden, expansion=1.0, activation=activation, norm=norm)
                for _ in range(int(num_blocks))
            ]
        )
        self.conv3 = ConvNormAct(hidden * 2, out_channels, kernel_size=1, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pool_size: int = 5, activation: str = "silu", norm: str = "bn"):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=1, activation=activation, norm=norm)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = ConvNormAct(out_channels * 4, out_channels, kernel_size=1, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
