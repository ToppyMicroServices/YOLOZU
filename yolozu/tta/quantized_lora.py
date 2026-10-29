"""Portable research INT4 storage with differentiable low-rank updates.

Weights use symmetric per-output-channel quantization and two signed four-bit
values per uint8 buffer element. Forward temporarily dequantizes them and uses
ordinary floating-point PyTorch operations. This is not an optimized INT4
kernel, NF4, or a reproduction of the original QLoRA recipe.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


RECIPE = "symmetric_int4_per_output_channel_v1"
ADAPTER_FORMAT = "yolozu_cnn_lora_v1"


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def pack_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a matrix/kernel into nibbles and return one float32 scale per row."""
    if weight.ndim < 2 or weight.numel() == 0:
        raise ValueError("INT4 weights must be nonempty with at least two dimensions")
    _finite(weight, "weight")
    with torch.no_grad():
        rows = weight.detach().float().reshape(weight.shape[0], -1)
        scales = rows.abs().amax(dim=1) / 7.0
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        values = torch.round(rows / scales[:, None]).clamp(-7, 7).to(torch.int8)
        nibbles = (values.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
        if nibbles.numel() % 2:
            nibbles = torch.cat((nibbles, nibbles.new_tensor([8])))
        packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed, scales


def unpack_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a packed weight for a floating-point reference operation."""
    count = math.prod(shape)
    if len(shape) < 2 or count <= 0 or any(size <= 0 for size in shape):
        raise ValueError("invalid INT4 weight shape")
    if packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != (count + 1) // 2:
        raise ValueError("packed weight does not match shape")
    if scales.ndim != 1 or scales.numel() != shape[0]:
        raise ValueError("scales do not match output channels")
    # No detached inputs participate here: gradients through the later floating
    # operation can reach adapters in earlier layers despite a frozen base.
    values = torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1)[:count]
    rows = values.to(dtype=dtype).reshape(shape[0], -1) - 8
    return (rows * scales.to(dtype=dtype)[:, None]).reshape(shape)


class _LowRankWeight(nn.Module):
    def __init__(self, base: nn.Linear | nn.Conv2d, *, quantize: bool, rank: int) -> None:
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        _finite(base.weight, "weight")
        if base.bias is not None:
            _finite(base.bias, "bias")
        self.quantized = bool(quantize)
        self.rank = int(rank)
        self.weight_shape = tuple(base.weight.shape)
        self.original_weight_bytes = base.weight.numel() * base.weight.element_size()
        if quantize:
            packed, scales = pack_int4(base.weight)
            self.register_buffer("packed_weight", packed)
            self.register_buffer("weight_scales", scales)
        else:
            self.register_buffer("base_weight", base.weight.detach())
        self.register_buffer("base_bias", None if base.bias is None else base.bias.detach())

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.quantized:
            return unpack_int4(self.packed_weight, self.weight_scales, self.weight_shape, dtype=dtype)
        return self.base_weight.to(dtype=dtype)

    def _initialize_adapters(self, base: nn.Module, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        self.lora_A = nn.Parameter(base.weight.new_empty(a_shape))
        self.lora_B = nn.Parameter(base.weight.new_zeros(b_shape))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)

    def _bias(self, dtype: torch.dtype) -> torch.Tensor | None:
        return None if self.base_bias is None else self.base_bias.to(dtype=dtype)


class QuantizedLoRALinear(_LowRankWeight):
    """Linear layer with optional packed frozen weights and a zero-initialized delta."""

    def __init__(self, base: nn.Linear, *, quantize: bool, rank: int = 0) -> None:
        super().__init__(base, quantize=quantize, rank=rank)
        self.in_features = base.in_features
        self.out_features = base.out_features
        if rank:
            self._initialize_adapters(base, (rank, base.in_features), (base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.dequantized_weight(x.dtype), self._bias(x.dtype))
        if self.rank:
            y = y + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return y


class QuantizedLoRAConv2d(_LowRankWeight):
    """Conv2d reference layer; LoRA uses the base kernel followed by a 1x1 kernel."""

    def __init__(self, base: nn.Conv2d, *, quantize: bool, rank: int = 0) -> None:
        if rank and base.groups != 1:
            raise ValueError("Conv2d LoRA adapters require groups=1; grouped bases may be quantized")
        super().__init__(base, quantize=quantize, rank=rank)
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding_mode = base.padding_mode
        self._reversed_padding_repeated_twice = base._reversed_padding_repeated_twice
        if rank:
            self._initialize_adapters(
                base,
                (rank, base.in_channels, *base.kernel_size),
                (base.out_channels, rank, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        y = F.conv2d(
            x, self.dequantized_weight(x.dtype), self._bias(x.dtype),
            self.stride, padding, self.dilation, self.groups,
        )
        if self.rank:
            delta = F.conv2d(x, self.lora_A, None, self.stride, padding, self.dilation, 1)
            y = y + F.conv2d(delta, self.lora_B)
        return y


def _is_adapter(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")


def _scope_matches(name: str, scopes: tuple[str, ...]) -> bool:
    return any(scope == "*" or name == scope or name.startswith(scope + ".") for scope in scopes)


def _validate_scopes(scopes: tuple[str, ...], name: str) -> None:
    if not isinstance(scopes, tuple) or any(not isinstance(s, str) or not s or s.endswith(".") for s in scopes):
        raise ValueError(f"{name} must be a tuple of nonempty module-name prefixes")


def _tensor_bytes(tensors: Any) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def prepare_model(
    model: nn.Module,
    *,
    quantize: bool,
    rank: int,
    adapter_include: tuple[str, ...],
    quant_include: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Freeze a model and replace selected Linear/Conv2d layers in place.

    Scopes match exact module names or descendants; ``('*',)`` selects all.
    ``quant_include=None`` quantizes all supported layers. A positive rank must
    select at least one adapter. Models already prepared here are rejected.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    _validate_scopes(adapter_include, "adapter_include")
    quant_scopes = ("*",) if quant_include is None else quant_include
    _validate_scopes(quant_scopes, "quant_include")
    if any(isinstance(module, _LowRankWeight) for module in model.modules()):
        raise ValueError("model already contains prepared quantized LoRA layers")
    if any(_is_adapter(name) for name, _ in model.named_parameters()):
        raise ValueError("model already contains LoRA adapter parameters")
    # Preflight the entire plan before modifying any module or requires_grad bit.
    planned: list[tuple[str, nn.Module, bool, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        q = bool(quantize and _scope_matches(name, quant_scopes))
        r = rank if _scope_matches(name, adapter_include) else 0
        if not q and not r:
            continue
        if not name:
            raise ValueError("wrap a standalone Linear/Conv2d in a parent module before preparation")
        if r and isinstance(module, nn.Conv2d) and module.groups != 1:
            raise ValueError(f"adapter layer {name} uses groups={module.groups}; only groups=1 is supported")
        _finite(module.weight, f"{name}.weight")
        if module.bias is not None:
            _finite(module.bias, f"{name}.bias")
        planned.append((name, module, q, r))
    if quantize and not any(q for _, _, q, _ in planned):
        raise ValueError("quantization scope selected no Linear or Conv2d layers")
    if rank and not any(r for _, _, _, r in planned):
        raise ValueError("adapter scope selected no Linear or Conv2d layers")

    before = _tensor_bytes(model.parameters()) + _tensor_bytes(model.buffers())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers: list[dict[str, Any]] = []
    for name, module, quantized, adapter_rank in planned:
        wrapper_type = QuantizedLoRALinear if isinstance(module, nn.Linear) else QuantizedLoRAConv2d
        wrapper = wrapper_type(module, quantize=quantized, rank=adapter_rank)
        wrapper.train(module.training)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, wrapper)
        layers.append({
            "name": name,
            "type": type(module).__name__,
            "weight_shape": list(module.weight.shape),
            "quantized": quantized,
            "bits": 4 if quantized else module.weight.element_size() * 8,
            "adapter_rank": adapter_rank,
            "original_weight_bytes": wrapper.original_weight_bytes,
            "packed_weight_bytes": wrapper.packed_weight.numel() if quantized else 0,
            "scale_bytes": wrapper.weight_scales.numel() * wrapper.weight_scales.element_size() if quantized else 0,
            "adapter_params": sum(p.numel() for p in wrapper.parameters()),
        })
    base_parameters = (p for name, p in model.named_parameters() if not _is_adapter(name))
    after = _tensor_bytes(base_parameters) + _tensor_bytes(model.buffers())
    adapters = [p for name, p in model.named_parameters() if _is_adapter(name)]
    return {
        "recipe": RECIPE if quantize else "floating_base_lora_v1",
        "arithmetic": "floating_point_after_transient_dequantization" if quantize else "floating_point",
        "quantized_layer_count": sum(layer["quantized"] for layer in layers),
        "adapter_layer_count": sum(layer["adapter_rank"] > 0 for layer in layers),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "base_storage_bytes_before": before,
        "base_storage_bytes_after": after,
        "adapter_storage_bytes": _tensor_bytes(adapters),
        "layers": layers,
    }


def _recipe_identity(model: nn.Module) -> list[dict[str, Any]]:
    identity = []
    for name, module in model.named_modules():
        if not isinstance(module, _LowRankWeight):
            continue
        item = {
            "name": name, "type": type(module).__name__, "shape": list(module.weight_shape),
            "rank": module.rank, "recipe": RECIPE if module.quantized else "floating_base_lora_v1",
        }
        if isinstance(module, QuantizedLoRAConv2d):
            item.update({
                "stride": list(module.stride),
                "padding": module.padding if isinstance(module.padding, str) else list(module.padding),
                "dilation": list(module.dilation),
                "groups": module.groups,
                "padding_mode": module.padding_mode,
            })
        identity.append(item)
    return identity


def base_fingerprint(model: nn.Module) -> str:
    """Hash frozen weights and buffers, including normalization state, excluding adapters."""
    digest = hashlib.sha256()
    digest.update(json.dumps(_recipe_identity(model), sort_keys=True).encode("utf-8"))
    architecture = [
        (name, type(module).__module__, type(module).__qualname__, module.extra_repr())
        for name, module in model.named_modules()
    ]
    digest.update(json.dumps(architecture).encode("utf-8"))
    for name, tensor in sorted(model.state_dict().items()):
        if _is_adapter(name):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_state_dict(model: nn.Module) -> dict[str, Any]:
    """Create a weights-only-loadable adapter envelope tied to the exact frozen base."""
    tensors = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if _is_adapter(name)}
    if not tensors:
        raise ValueError("model has no LoRA adapters")
    for name, tensor in tensors.items():
        _finite(tensor, name)
    return {
        "format": ADAPTER_FORMAT,
        "base_sha256": base_fingerprint(model),
        "recipe": _recipe_identity(model),
        "adapter_tensors": tensors,
    }


def load_adapter_state_dict(model: nn.Module, payload: dict[str, Any]) -> None:
    """Validate all adapter values and base identity before applying any changes."""
    if not isinstance(payload, dict) or payload.get("format") != ADAPTER_FORMAT:
        raise ValueError("unrecognized adapter format")
    if payload.get("recipe") != _recipe_identity(model):
        raise ValueError("adapter recipe does not match model")
    if payload.get("base_sha256") != base_fingerprint(model):
        raise ValueError("adapter frozen-base fingerprint mismatch")
    target = {name: p for name, p in model.named_parameters() if _is_adapter(name)}
    source = payload.get("adapter_tensors")
    if not target or not isinstance(source, dict) or set(source) != set(target):
        raise ValueError("adapter tensor names do not match model")
    for name, parameter in target.items():
        value = source[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError(f"adapter tensor shape/dtype mismatch: {name}")
        _finite(value, name)
    with torch.no_grad():
        for name, parameter in target.items():
            parameter.copy_(source[name].to(device=parameter.device))
