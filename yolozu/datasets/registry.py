"""Unified dataset adapter registry.

Provides a declarative approach to dataset format discovery and loading.
Each adapter registers itself with a format name, a layout probe function,
and an iterator that yields samples in a common ``DatasetSample`` format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

__all__ = [
    "DatasetSample",
    "DatasetInfo",
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "probe_format",
    "iter_samples",
]


# ---------------------------------------------------------------------------
# Common sample record
# ---------------------------------------------------------------------------

@dataclass
class DatasetSample:
    """Format-agnostic sample record returned by every adapter.

    Attributes:
        image_path: Absolute path to the image file.
        split: Dataset split name (``train``, ``val``, ``test``, etc.).
        sample_id: Unique identifier within the dataset.
        mask_path: Optional semantic segmentation mask path.
        labels: Optional list of detection/pose labels (YOLO-normalised).
        annotation_path: Optional path to raw annotation file (e.g. VOC XML).
        extra: Adapter-specific metadata (class names, image size, etc.).
    """

    image_path: Path
    split: str
    sample_id: str
    mask_path: Path | None = None
    labels: list[dict[str, Any]] = field(default_factory=list)
    annotation_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a recognised dataset layout."""

    format_name: str
    root: Path
    splits: list[str]
    task: str  # "detection", "segmentation", "instance-seg", "pose", "multi"
    num_classes: int | None = None
    class_names: list[str] | None = None


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

ProbeFunc = Callable[[Path], DatasetInfo | None]
IterFunc = Callable[..., Iterator[DatasetSample]]


@runtime_checkable
class DatasetAdapter(Protocol):
    """Minimal protocol that dataset adapters implement."""

    format_name: str

    def probe(self, root: Path) -> DatasetInfo | None:
        """Return ``DatasetInfo`` if *root* matches this format, else ``None``."""
        raise NotImplementedError

    def iter_samples(
        self,
        root: Path,
        *,
        split: str = "train",
        **kwargs: Any,
    ) -> Iterator[DatasetSample]:
        """Yield ``DatasetSample`` records for the requested split."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, DatasetAdapter] = {}


def register_adapter(adapter: DatasetAdapter) -> DatasetAdapter:
    """Register *adapter* under its ``format_name``."""
    name = adapter.format_name
    if name in _ADAPTERS:
        raise ValueError(f"dataset adapter already registered: {name}")
    _ADAPTERS[name] = adapter
    return adapter


def get_adapter(name: str) -> DatasetAdapter:
    """Retrieve a registered adapter by format name."""
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset adapter: {name!r}  (registered: {sorted(_ADAPTERS)})")


def list_adapters() -> list[str]:
    """Return sorted list of registered adapter format names."""
    return sorted(_ADAPTERS)


def probe_format(root: str | Path) -> DatasetInfo | None:
    """Auto-detect dataset format at *root*.

    Iterates over all registered adapters and returns the first match.
    Returns ``None`` when no adapter recognises the layout.
    """
    root_path = Path(root)
    for adapter in _ADAPTERS.values():
        info = adapter.probe(root_path)
        if info is not None:
            return info
    return None


def iter_samples(
    root: str | Path,
    *,
    format_name: str | None = None,
    split: str = "train",
    **kwargs: Any,
) -> Iterator[DatasetSample]:
    """Yield ``DatasetSample`` from a dataset at *root*.

    When *format_name* is ``None`` the format is auto-detected via
    :func:`probe_format`.
    """
    root_path = Path(root)
    if format_name is not None:
        adapter = get_adapter(format_name)
    else:
        info = probe_format(root_path)
        if info is None:
            raise ValueError(f"cannot auto-detect dataset format at: {root_path}")
        adapter = get_adapter(info.format_name)
    yield from adapter.iter_samples(root_path, split=split, **kwargs)
