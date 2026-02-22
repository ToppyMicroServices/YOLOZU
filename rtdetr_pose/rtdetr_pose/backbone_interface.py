from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

try:
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    class _NNModuleStub:  # type: ignore[no-redef]
        pass

    Tensor = None  # type: ignore[assignment,misc]
    nn = SimpleNamespace(Module=_NNModuleStub)  # type: ignore[assignment]


class BaseBackbone(nn.Module, ABC):
    out_strides: Sequence[int] = (8, 16, 32)

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
        for idx, (feat, stride) in enumerate(zip(features, self.out_strides), start=3):
            expected_h = max(h0 // int(stride), 1)
            expected_w = max(w0 // int(stride), 1)
            got_h, got_w = int(feat.shape[-2]), int(feat.shape[-1])
            if got_h != expected_h or got_w != expected_w:
                raise ValueError(
                    f"P{idx} shape mismatch: expected ({expected_h},{expected_w}) for stride {stride}, "
                    f"got ({got_h},{got_w})"
                )
