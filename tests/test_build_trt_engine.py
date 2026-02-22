import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_trt_engine


class _DummyInput:
    def __init__(self, name):
        self.name = name


class _DummyNetwork:
    def __init__(self, names):
        self._inputs = [_DummyInput(n) for n in names]

    @property
    def num_inputs(self):
        return len(self._inputs)

    def get_input(self, idx):
        return self._inputs[idx]


class TestBuildTrtEngine(unittest.TestCase):
    def setUp(self):
        build_trt_engine._TRTEXEC_HELP_CACHE.clear()

    def test_resolve_input_name_prefers_requested(self):
        net = _DummyNetwork(["images", "other"])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "images")

    def test_resolve_input_name_falls_back(self):
        net = _DummyNetwork(["data"])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "data")

    def test_resolve_input_name_no_inputs(self):
        net = _DummyNetwork([])
        name = build_trt_engine._resolve_input_name(net, "images")
        self.assertEqual(name, "images")

    def test_trtexec_workspace_flag_prefers_mem_pool(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = " --memPoolSize=workspace:4096 "
        arg = build_trt_engine._trtexec_workspace_arg("trtexec", 4096)
        self.assertEqual(arg, "--memPoolSize=workspace:4096")

    def test_trtexec_workspace_flag_prefers_legacy_workspace(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = " --workspace=4096 "
        arg = build_trt_engine._trtexec_workspace_arg("trtexec", 4096)
        self.assertEqual(arg, "--workspace=4096")

    def test_trtexec_timing_cache_flag_is_optional(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = ""
        arg = build_trt_engine._trtexec_timing_cache_arg("trtexec", Path("/tmp/timing.cache"))
        self.assertIsNone(arg)

    def test_trtexec_timing_cache_flag_supported(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = " --timingCacheFile=/tmp/timing.cache "
        arg = build_trt_engine._trtexec_timing_cache_arg("trtexec", Path("/tmp/timing.cache"))
        self.assertEqual(arg, "--timingCacheFile=/tmp/timing.cache")

    def test_build_command_omits_shape_flags_for_static_onnx(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = " --memPoolSize --timingCacheFile "

        orig = build_trt_engine._onnx_dynamic_input
        build_trt_engine._onnx_dynamic_input = lambda _path, _name: {"name": "images", "dims": [1, 3, 64, 64], "dynamic": False}
        try:
            args = Namespace(
                trtexec="trtexec",
                input_name="images",
                min_shape="1x3x64x64",
                opt_shape="1x3x64x64",
                max_shape="1x3x64x64",
                workspace=4096,
                precision="fp16",
                calib_cache=None,
                extra_args=None,
            )
            cmd = build_trt_engine._build_command(
                args,
                onnx_path=Path("model.onnx"),
                engine_path=Path("model.plan"),
                timing_cache=Path("timing.cache"),
            )
            cmd_str = " ".join(cmd)
            self.assertNotIn("--minShapes=", cmd_str)
            self.assertNotIn("--optShapes=", cmd_str)
            self.assertNotIn("--maxShapes=", cmd_str)
        finally:
            build_trt_engine._onnx_dynamic_input = orig

    def test_build_command_keeps_shape_flags_for_dynamic_onnx(self):
        build_trt_engine._TRTEXEC_HELP_CACHE["trtexec"] = " --memPoolSize --timingCacheFile "

        orig = build_trt_engine._onnx_dynamic_input
        build_trt_engine._onnx_dynamic_input = lambda _path, _name: {"name": "images", "dims": ["batch", 3, 64, 64], "dynamic": True}
        try:
            args = Namespace(
                trtexec="trtexec",
                input_name="images",
                min_shape="1x3x64x64",
                opt_shape="1x3x64x64",
                max_shape="1x3x64x64",
                workspace=4096,
                precision="fp16",
                calib_cache=None,
                extra_args=None,
            )
            cmd = build_trt_engine._build_command(
                args,
                onnx_path=Path("model.onnx"),
                engine_path=Path("model.plan"),
                timing_cache=Path("timing.cache"),
            )
            cmd_str = " ".join(cmd)
            self.assertIn("--minShapes=images:1x3x64x64", cmd_str)
            self.assertIn("--optShapes=images:1x3x64x64", cmd_str)
            self.assertIn("--maxShapes=images:1x3x64x64", cmd_str)
        finally:
            build_trt_engine._onnx_dynamic_input = orig


if __name__ == "__main__":
    unittest.main()
