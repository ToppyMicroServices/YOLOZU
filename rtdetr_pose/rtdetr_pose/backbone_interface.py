from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from torch import Tensor, nn


class BaseBackbone(nn.Module, ABC):
    out_strides: Sequence[int] = (8, 16, 32)
    # Allow floor/ceil reduction differences across backbones with odd input sizes.
    stride_tolerance: int = 0

    @property
    @abstractmethod
    def out_channels(self) -> Sequence[int]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def validate_contract(self, x: Tensor, features: Sequence[Tensor]) -> None:
        if len(features) != 3:
            raise ValueError(f"backbone must return 3 features [P3,P4,P5], got {len(features)}")
        h0, w0 = int(x.shape[-2]), int(x.shape[-1])
        tol = max(0, int(getattr(self, "stride_tolerance", 0) or 0))
        for idx, (feat, stride) in enumerate(zip(features, self.out_strides), start=3):
            stride_i = max(1, int(stride))
            h_floor = max(h0 // stride_i, 1)
            w_floor = max(w0 // stride_i, 1)
            h_ceil = max((h0 + stride_i - 1) // stride_i, 1)
            w_ceil = max((w0 + stride_i - 1) // stride_i, 1)
            got_h, got_w = int(feat.shape[-2]), int(feat.shape[-1])
            h_min = min(h_floor, h_ceil) - tol
            h_max = max(h_floor, h_ceil) + tol
            w_min = min(w_floor, w_ceil) - tol
            w_max = max(w_floor, w_ceil) + tol
            h_ok = h_min <= got_h <= h_max
            w_ok = w_min <= got_w <= w_max
            if not h_ok or not w_ok:
                raise ValueError(
                    f"P{idx} shape mismatch: expected H in [{h_min},{h_max}] and W in [{w_min},{w_max}] "
                    f"for stride {stride_i} (floor=({h_floor},{w_floor}) ceil=({h_ceil},{w_ceil}) tol={tol}), "
                    f"got ({got_h},{got_w})"
                )
