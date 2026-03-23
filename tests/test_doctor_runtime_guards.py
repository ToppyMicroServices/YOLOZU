import unittest
from unittest.mock import patch


_ORIGINAL_IMPORT = __import__


class TestDoctorRuntimeGuards(unittest.TestCase):
    def test_gather_gpu_info_handles_optional_import_errors(self) -> None:
        from yolozu.core.doctor import _gather_gpu_info

        def _fake_import(name, *args, **kwargs):
            if name in {"torch", "onnxruntime"}:
                raise OSError("missing runtime")
            return _ORIGINAL_IMPORT(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            gpu = _gather_gpu_info()
        self.assertIsNone(gpu["torch"])
        self.assertIsNone(gpu["onnxruntime"])

    def test_gather_runtime_capabilities_handles_probe_errors(self) -> None:
        from yolozu.core.doctor import _gather_runtime_capabilities

        class FakeCuda:
            @staticmethod
            def is_available():
                raise RuntimeError("boom")

        class FakeBackends:
            cudnn = type("Cudnn", (), {"version": staticmethod(lambda: 1), "is_available": staticmethod(lambda: False)})

        class FakeTorch:
            __version__ = "0"
            cuda = FakeCuda()
            backends = FakeBackends()
            version = type("Version", (), {"cuda": None})()

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                return FakeTorch()
            if name in {"onnxruntime", "tensorrt", "cv2"}:
                raise OSError("missing")
            return _ORIGINAL_IMPORT(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            runtime = _gather_runtime_capabilities(tools={"trtexec": False}, gpu={})
        self.assertFalse(runtime["torch"]["cuda_available"])
        self.assertFalse(runtime["onnxruntime"]["installed"])


if __name__ == "__main__":
    unittest.main()