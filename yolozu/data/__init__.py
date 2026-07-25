"""Package resources + dataset adapters shipped with YOLOZU.

Keep this module import lightweight so resource-only callers
(`importlib.resources.files("yolozu.data")`) do not require optional
runtime dependencies from dataset adapters.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SynthGenShardDataset", "SynthGenStreamDataset", "SynthGenStreamPolicy", "collate_synthgen_batch"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"SynthGenShardDataset", "collate_synthgen_batch"}:
        from .synthgen_shard_dataset import SynthGenShardDataset, collate_synthgen_batch

        exports = {
            "SynthGenShardDataset": SynthGenShardDataset,
            "collate_synthgen_batch": collate_synthgen_batch,
        }
        value = exports[name]
        globals()[name] = value
        return value

    if name in {"SynthGenStreamDataset", "SynthGenStreamPolicy"}:
        from .synthgen_stream_dataset import SynthGenStreamDataset, SynthGenStreamPolicy

        exports = {
            "SynthGenStreamDataset": SynthGenStreamDataset,
            "SynthGenStreamPolicy": SynthGenStreamPolicy,
        }
        value = exports[name]
        globals()[name] = value
        return value

    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))
