import importlib.util
import unittest
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBackendShapeFormatContracts(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.onnxrt = _load_module(repo_root / "tools" / "export_predictions_onnxrt.py", "export_predictions_onnxrt_contracts")
        self.trt = _load_module(repo_root / "tools" / "export_predictions_trt.py", "export_predictions_trt_contracts")
        self.opencv_yolo = _load_module(repo_root / "tools" / "export_predictions_opencv_dnn.py", "export_predictions_opencv_dnn_contracts")
        self.opencv_rtdetr = _load_module(
            repo_root / "tools" / "export_predictions_opencv_dnn_rtdetr.py",
            "export_predictions_opencv_dnn_rtdetr_contracts",
        )

    def test_onnx_input_size_resolution_handles_static_and_non_square(self):
        size, warning = self.onnxrt._resolve_onnx_input_size([1, 3, 64, 64], default_size=640)
        self.assertEqual(size, 64)
        self.assertIn("ONNX-fixed input size 64x64", warning or "")

        size_dyn, warning_dyn = self.onnxrt._resolve_onnx_input_size([1, 3, "h", "w"], default_size=640)
        self.assertEqual(size_dyn, 640)
        self.assertIsNone(warning_dyn)

        with self.assertRaises(SystemExit) as ctx:
            self.onnxrt._resolve_onnx_input_size([1, 3, 64, 96], default_size=640)
        self.assertIn("non-square", str(ctx.exception))

    def test_onnx_decode_contract_rejects_mismatched_flags(self):
        with self.assertRaises(ValueError) as box_ctx:
            self.onnxrt._validate_decode_contract(
                exporter_name="skeleton",
                boxes_format="cxcywh",
                boxes_scale="abs",
                raw_output=None,
            )
        self.assertIn("only --boxes-format xyxy", str(box_ctx.exception))

        with self.assertRaises(ValueError) as scale_ctx:
            self.onnxrt._validate_decode_contract(
                exporter_name="skeleton",
                boxes_format="xyxy",
                boxes_scale="norm",
                raw_output="output0",
            )
        self.assertIn("--raw-output expects --boxes-scale abs", str(scale_ctx.exception))

    def test_trt_input_size_resolution_catches_static_shape_mismatch(self):
        with self.assertRaises(SystemExit) as mismatch_ctx:
            self.trt._resolve_trt_input_size(
                requested_imgsz=640,
                eng_h=64,
                eng_w=64,
                eng_dynamic=False,
            )
        self.assertIn("engine expects input 64x64 but imgsz=640", str(mismatch_ctx.exception))

        with self.assertRaises(SystemExit) as nonsquare_ctx:
            self.trt._resolve_trt_input_size(
                requested_imgsz=640,
                eng_h=64,
                eng_w=96,
                eng_dynamic=False,
            )
        self.assertIn("non-square", str(nonsquare_ctx.exception))

        size_dyn, warn_dyn = self.trt._resolve_trt_input_size(
            requested_imgsz=None,
            eng_h=None,
            eng_w=None,
            eng_dynamic=True,
        )
        self.assertEqual(size_dyn, 640)
        self.assertIn("dynamic", warn_dyn or "")

    def test_trt_decode_contract_rejects_mismatched_flags(self):
        with self.assertRaises(ValueError) as box_ctx:
            self.trt._validate_decode_contract(
                exporter_name="exporter",
                boxes_format="cxcywh",
                boxes_scale="abs",
                raw_output=None,
            )
        self.assertIn("only --boxes-format xyxy", str(box_ctx.exception))

        with self.assertRaises(ValueError) as scale_ctx:
            self.trt._validate_decode_contract(
                exporter_name="exporter",
                boxes_format="xyxy",
                boxes_scale="norm",
                raw_output="raw",
            )
        self.assertIn("--raw-output expects --boxes-scale abs", str(scale_ctx.exception))

    def test_yolo_decode_shape_errors_remain_actionable(self):
        np = self.opencv_yolo.np
        if np is None:
            self.skipTest("numpy not available")
        with self.assertRaises(ValueError) as y84:
            self.opencv_yolo._normalize_raw_yolo84(np.zeros((1, 7, 7), dtype=np.float32))
        self.assertIn("unsupported raw output shape for yolo_84", str(y84.exception))

        with self.assertRaises(ValueError) as y85:
            self.opencv_yolo._normalize_raw_yolo85_obj(np.zeros((1, 5, 5), dtype=np.float32))
        self.assertIn("unsupported raw output shape for yolo_85_obj", str(y85.exception))

    def test_rtdetr_decode_background_zero_branch(self):
        np = self.opencv_rtdetr.np
        if np is None:
            self.skipTest("numpy not available")

        logits = np.array(
            [
                [9.0, 1.0, 1.0],
                [8.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )
        class_ids, scores = self.opencv_rtdetr._decode_logits(
            logits,
            activation="softmax",
            background_class="zero",
        )
        self.assertEqual(int(class_ids[0]), 1)
        self.assertEqual(int(class_ids[1]), 2)
        self.assertTrue(float(scores[1]) > 0.0)


if __name__ == "__main__":
    unittest.main()
