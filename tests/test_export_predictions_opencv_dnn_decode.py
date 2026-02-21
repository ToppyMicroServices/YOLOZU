import importlib.util
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "export_predictions_opencv_dnn.py"
    spec = importlib.util.spec_from_file_location("export_predictions_opencv_dnn", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOpenCVDNNDecoding(unittest.TestCase):
    def test_normalize_yolo85_obj_shapes(self):
        mod = _load_module()
        np = mod.np
        if np is None:
            self.skipTest("numpy not available")

        raw = np.zeros((1, 3, 85), dtype=np.float32)
        out = mod._normalize_raw_yolo85_obj(raw)
        self.assertEqual(out.shape, (3, 85))

        raw2 = np.zeros((85, 3), dtype=np.float32)
        out2 = mod._normalize_raw_yolo85_obj(raw2)
        self.assertEqual(out2.shape, (3, 85))

    def test_decode_yolo85_obj_multiplies_objectness(self):
        mod = _load_module()
        np = mod.np
        if np is None:
            self.skipTest("numpy not available")

        # Two boxes, same location, different classes.
        # scores = obj * class_prob
        raw = np.zeros((1, 2, 85), dtype=np.float32)
        # Box 0: class 3 with high obj
        raw[0, 0, 0:4] = np.array([320.0, 320.0, 100.0, 100.0], dtype=np.float32)
        raw[0, 0, 4] = 0.5  # obj
        raw[0, 0, 5 + 3] = 0.8  # class prob
        # Box 1: class 7 with lower combined score
        raw[0, 1, 0:4] = np.array([320.0, 320.0, 100.0, 100.0], dtype=np.float32)
        raw[0, 1, 4] = 0.9
        raw[0, 1, 5 + 7] = 0.1

        boxes, scores, class_ids = mod._decode_yolo85_obj(
            raw,
            boxes_scale="abs",
            input_size=640,
            min_score=0.0,
            iou_thresh=0.5,
            max_det=10,
            agnostic=True,
        )
        self.assertEqual(len(scores), 1, "expected NMS to keep one box")
        self.assertEqual(int(class_ids[0]), 3)
        self.assertAlmostEqual(float(scores[0]), 0.5 * 0.8, places=6)


if __name__ == "__main__":
    unittest.main()

