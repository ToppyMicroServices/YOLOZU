"""Stable third-party adapter SDK and conformance helpers.

Discovery reads package metadata only.  Plugin code is imported exclusively
after the caller opts in with ``allow_plugin_load=True``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from yolozu.api import PredictionsValidationError, validate_predictions
from yolozu.inference.adapter import DummyAdapter

__all__ = [
    "ADAPTER_ENTRY_POINT_GROUP",
    "ADAPTER_INTERFACE_VERSION",
    "AdapterSDKError",
    "AdapterPlugin",
    "AdapterDescriptor",
    "AdapterConformanceResult",
    "BuiltinDummyAdapterPlugin",
    "discover_adapters",
    "load_adapter_plugin",
    "check_adapter_plugin",
    "run_adapter_conformance",
    "test_adapter_plugin",
    "bundled_conformance_records",
]

ADAPTER_ENTRY_POINT_GROUP = "yolozu.adapters.v1"
ADAPTER_INTERFACE_VERSION = 1


class AdapterSDKError(ValueError):
    """Adapter discovery, loading, or conformance failed."""


@runtime_checkable
class AdapterPlugin(Protocol):
    """Stable entry-point interface contract for adapter providers."""

    adapter_id: str
    interface_version: int
    summary: str
    capabilities: Mapping[str, Any]

    def create(self, config: Mapping[str, Any]) -> Any:
        """Create an object implementing ``predict(records)``."""


@dataclass(frozen=True)
class AdapterDescriptor:
    name: str
    value: str
    group: str
    distribution: str | None
    distribution_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "group": self.group,
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
            "loaded": False,
        }


@dataclass(frozen=True)
class AdapterConformanceResult:
    adapter_id: str
    ok: bool
    record_count: int
    prediction_count: int
    detection_count: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ok": self.ok,
            "adapter_id": self.adapter_id,
            "record_count": self.record_count,
            "prediction_count": self.prediction_count,
            "detection_count": self.detection_count,
            "warnings": list(self.warnings),
        }


class BuiltinDummyAdapterPlugin:
    """Network-free reference plugin used to verify installed SDK wiring."""

    adapter_id = "dummy"
    interface_version = ADAPTER_INTERFACE_VERSION
    summary = "Built-in empty-detection adapter for interface-contract smoke tests."
    capabilities = {
        "tasks": ["object_detection"],
        "network_required": False,
        "model_inference": False,
    }

    def create(self, config: Mapping[str, Any]) -> DummyAdapter:
        if not isinstance(config, Mapping):
            raise AdapterSDKError("adapter config must be an object")
        return DummyAdapter()


def _entry_points() -> tuple[metadata.EntryPoint, ...]:
    available = metadata.entry_points()
    if hasattr(available, "select"):
        selected = available.select(group=ADAPTER_ENTRY_POINT_GROUP)
    else:  # pragma: no cover - compatibility for old importlib.metadata
        selected = available.get(ADAPTER_ENTRY_POINT_GROUP, ())
    return tuple(sorted(selected, key=lambda item: (item.name, item.value)))


def _descriptor(entry_point: metadata.EntryPoint) -> AdapterDescriptor:
    distribution = getattr(entry_point, "dist", None)
    return AdapterDescriptor(
        name=str(entry_point.name),
        value=str(entry_point.value),
        group=ADAPTER_ENTRY_POINT_GROUP,
        distribution=(str(distribution.name) if distribution is not None else None),
        distribution_version=(str(distribution.version) if distribution is not None else None),
    )


def discover_adapters() -> tuple[AdapterDescriptor, ...]:
    """List installed adapters without importing their modules."""

    return tuple(_descriptor(entry_point) for entry_point in _entry_points())


def _selected_entry_point(name: str) -> metadata.EntryPoint:
    token = str(name).strip()
    if not token:
        raise AdapterSDKError("adapter name is required")
    matches = [entry_point for entry_point in _entry_points() if entry_point.name == token]
    if not matches:
        raise AdapterSDKError(f"adapter entry point not found: {token}")
    if len(matches) > 1:
        providers = ", ".join(str(_descriptor(item).distribution) for item in matches)
        raise AdapterSDKError(f"duplicate adapter entry point {token!r}: {providers}")
    return matches[0]


def check_adapter_plugin(plugin: Any, *, expected_name: str | None = None) -> AdapterPlugin:
    """Validate loaded plugin metadata without creating its runtime adapter."""

    missing = [
        field
        for field in ("adapter_id", "interface_version", "summary", "capabilities", "create")
        if not hasattr(plugin, field)
    ]
    if missing:
        raise AdapterSDKError(f"adapter plugin is missing required fields: {', '.join(missing)}")
    adapter_id = getattr(plugin, "adapter_id")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise AdapterSDKError("adapter_id must be a nonempty string")
    if expected_name is not None and adapter_id != expected_name:
        raise AdapterSDKError(
            f"adapter_id {adapter_id!r} does not match entry-point name {expected_name!r}"
        )
    if getattr(plugin, "interface_version") != ADAPTER_INTERFACE_VERSION:
        raise AdapterSDKError(
            f"unsupported adapter interface_version: {getattr(plugin, 'interface_version')!r}"
        )
    if not isinstance(getattr(plugin, "summary"), str) or not getattr(plugin, "summary").strip():
        raise AdapterSDKError("summary must be a nonempty string")
    capabilities = getattr(plugin, "capabilities")
    if not isinstance(capabilities, Mapping):
        raise AdapterSDKError("capabilities must be an object")
    try:
        json.dumps(dict(capabilities), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AdapterSDKError(f"capabilities must be JSON-serializable: {exc}") from exc
    if not callable(getattr(plugin, "create")):
        raise AdapterSDKError("create must be callable")
    return plugin


def load_adapter_plugin(name: str, *, allow_plugin_load: bool = False) -> AdapterPlugin:
    """Explicitly import and validate one selected adapter plugin."""

    if not allow_plugin_load:
        raise AdapterSDKError(
            "plugin loading is disabled; set allow_plugin_load=True after reviewing the provider"
        )
    entry_point = _selected_entry_point(name)
    try:
        loaded = entry_point.load()
        plugin = loaded() if isinstance(loaded, type) else loaded
    except Exception as exc:
        raise AdapterSDKError(f"could not load adapter plugin {name!r}: {exc}") from exc
    return check_adapter_plugin(plugin, expected_name=entry_point.name)


def bundled_conformance_records() -> list[dict[str, Any]]:
    """Return one packaged, network-free image record for smoke conformance."""

    image = Path(__file__).resolve().parent / "data" / "conformance" / "adapter_sample.ppm"
    if not image.is_file():
        raise AdapterSDKError(f"packaged conformance image is missing: {image}")
    return [{"image": str(image), "image_hw": [8, 8]}]


def _validate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise AdapterSDKError("records must be a nonempty sequence of objects")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AdapterSDKError(f"records[{index}] must be an object")
        image = record.get("image")
        if not isinstance(image, str) or not image:
            raise AdapterSDKError(f"records[{index}].image must be a nonempty string")
        normalized.append(copy.deepcopy(dict(record)))
    if not normalized:
        raise AdapterSDKError("records must not be empty")
    return normalized


def run_adapter_conformance(
    adapter: Any,
    *,
    adapter_id: str = "<direct>",
    records: Sequence[Mapping[str, Any]] | None = None,
) -> AdapterConformanceResult:
    """Execute ``predict`` once and validate the stable predictions interface contract."""

    if not callable(getattr(adapter, "predict", None)):
        raise AdapterSDKError("created adapter must implement predict(records)")
    normalized_records = _validate_records(records or bundled_conformance_records())
    try:
        raw_predictions = adapter.predict(copy.deepcopy(normalized_records))
    except Exception as exc:
        raise AdapterSDKError(f"adapter predict failed: {exc}") from exc
    if not isinstance(raw_predictions, list):
        raise AdapterSDKError("adapter predict result must be a list")
    expected_images = [record["image"] for record in normalized_records]
    observed_images = [
        item.get("image") if isinstance(item, Mapping) else None
        for item in raw_predictions
    ]
    if observed_images != expected_images:
        raise AdapterSDKError(
            "adapter predict result must preserve record order and exact image keys"
        )
    try:
        validated = validate_predictions(raw_predictions)
    except PredictionsValidationError as exc:
        raise AdapterSDKError(
            f"adapter output violates the predictions interface contract: {exc.message}"
        ) from exc
    detections = sum(len(entry.get("detections") or []) for entry in validated.entries)
    return AdapterConformanceResult(
        adapter_id=str(adapter_id),
        ok=True,
        record_count=len(normalized_records),
        prediction_count=len(validated.entries),
        detection_count=detections,
        warnings=validated.warnings,
    )


def test_adapter_plugin(
    name: str,
    *,
    config: Mapping[str, Any] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    allow_plugin_load: bool = False,
) -> AdapterConformanceResult:
    """Load one approved plugin, create its adapter, and run conformance."""

    plugin = load_adapter_plugin(name, allow_plugin_load=allow_plugin_load)
    try:
        adapter = plugin.create(dict(config or {}))
    except Exception as exc:
        raise AdapterSDKError(f"adapter plugin create failed: {exc}") from exc
    return run_adapter_conformance(adapter, adapter_id=plugin.adapter_id, records=records)
