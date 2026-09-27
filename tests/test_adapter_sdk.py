from types import SimpleNamespace
import unittest
from unittest.mock import patch

from yolozu.adapter_sdk import (
    AdapterSDKError,
    BuiltinDummyAdapterPlugin,
    discover_adapters,
    load_adapter_plugin,
    run_adapter_conformance,
)


class _EntryPoints(list):
    def select(self, *, group):
        return _EntryPoints(item for item in self if item.group == group)


class _EntryPoint:
    group = "yolozu.adapters.v1"

    def __init__(self, name="dummy", loaded=None):
        self.name = name
        self.value = "provider:Plugin"
        self.dist = SimpleNamespace(name="provider", version="1.0")
        self.loaded = loaded or BuiltinDummyAdapterPlugin
        self.load_count = 0

    def load(self):
        self.load_count += 1
        return self.loaded


class TestAdapterSDK(unittest.TestCase):
    def test_discovery_does_not_import_plugin_code(self):
        entry_point = _EntryPoint()
        with patch("yolozu.adapter_sdk.metadata.entry_points", return_value=_EntryPoints([entry_point])):
            descriptors = discover_adapters()

        self.assertEqual(descriptors[0].name, "dummy")
        self.assertEqual(descriptors[0].distribution, "provider")
        self.assertEqual(entry_point.load_count, 0)

    def test_loading_requires_explicit_opt_in(self):
        entry_point = _EntryPoint()
        with patch("yolozu.adapter_sdk.metadata.entry_points", return_value=_EntryPoints([entry_point])):
            with self.assertRaises(AdapterSDKError):
                load_adapter_plugin("dummy")

        self.assertEqual(entry_point.load_count, 0)

    def test_explicit_load_checks_entry_point_identity(self):
        entry_point = _EntryPoint()
        with patch("yolozu.adapter_sdk.metadata.entry_points", return_value=_EntryPoints([entry_point])):
            plugin = load_adapter_plugin("dummy", allow_plugin_load=True)

        self.assertEqual(plugin.adapter_id, "dummy")
        self.assertEqual(entry_point.load_count, 1)

    def test_bundled_dummy_passes_prediction_conformance(self):
        adapter = BuiltinDummyAdapterPlugin().create({})
        result = run_adapter_conformance(adapter, adapter_id="dummy")

        self.assertTrue(result.ok)
        self.assertEqual(result.record_count, 1)
        self.assertEqual(result.prediction_count, 1)
        self.assertEqual(result.detection_count, 0)


if __name__ == "__main__":
    unittest.main()
