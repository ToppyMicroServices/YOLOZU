from __future__ import annotations

import argparse
import builtins
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from yolozu.demos.depth import _require_transformers
from yolozu.demos.pose6d import _download_sample
from yolozu.tta import integration as tta_integration
from yolozu.tta.presets import _ttt_core_is_defaultish


class TestTTADemoParseGuards(unittest.TestCase):
    def test_ttt_core_is_defaultish_returns_false_on_invalid_numeric(self) -> None:
        args = argparse.Namespace(
            ttt_steps="bad",
            ttt_batch_size=1,
            ttt_lr=1e-4,
            ttt_update_filter="all",
            ttt_max_batches=1,
        )
        self.assertFalse(_ttt_core_is_defaultish(args))

    def test_snapshot_norm_buffers_handles_missing_named_buffers(self) -> None:
        if tta_integration.torch is None:
            self.skipTest("torch not installed")

        class BrokenModel:
            def named_buffers(self):
                raise TypeError("bad iteration")

        self.assertEqual(tta_integration._snapshot_norm_buffers(BrokenModel()), [])

    def test_download_sample_url_error_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.jpg"
            with mock.patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("offline")):
                self.assertFalse(_download_sample(target=target))

    def test_require_transformers_wraps_missing_dependency(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "transformers":
                raise ImportError("missing transformers")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(RuntimeError):
                _require_transformers()


if __name__ == "__main__":
    unittest.main()
