import importlib.util
import unittest
from pathlib import Path


def _load_module(name: str, relpath: str):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_predictions_parity_trt = _load_module("check_predictions_parity_trt", "tools/check_predictions_parity_trt.py")
export_predictions_onnxrt = _load_module("export_predictions_onnxrt", "tools/export_predictions_onnxrt.py")
export_predictions_trt = _load_module("export_predictions_trt", "tools/export_predictions_trt.py")


class TestYoloRuntimePublicAliases(unittest.TestCase):
    def test_onnxrt_accepts_yolo_runtime_raw_postprocess_alias(self):
        args = export_predictions_onnxrt._parse_args(
            [
                "--dataset",
                "data/smoke",
                "--onnx",
                "models/dummy.onnx",
                "--raw-output",
                "output0",
                "--raw-postprocess",
                "yolo_runtime",
                "--dry-run",
            ]
        )
        self.assertEqual(args.raw_postprocess, "yolo_runtime")

    def test_trt_accepts_yolo_runtime_raw_postprocess_alias(self):
        args = export_predictions_trt._parse_args(
            [
                "--dataset",
                "data/smoke",
                "--engine",
                "models/dummy.plan",
                "--raw-output",
                "output0",
                "--raw-postprocess",
                "yolo_runtime",
                "--dry-run",
            ]
        )
        self.assertEqual(args.raw_postprocess, "yolo_runtime")

    def test_parity_trt_accepts_yolo_runtime_raw_postprocess_alias(self):
        args = check_predictions_parity_trt._parse_args(
            [
                "--reference",
                "reports/pred_ref.json",
                "--engine",
                "models/dummy.plan",
                "--dataset",
                "data/smoke",
                "--raw-output",
                "output0",
                "--raw-postprocess",
                "yolo_runtime",
                "--dry-run",
            ]
        )
        self.assertEqual(args.raw_postprocess, "yolo_runtime")


if __name__ == "__main__":
    unittest.main()
