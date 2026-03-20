import importlib.util
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "export_predictions_opencv_dnn_rtdetr.py"
    spec = importlib.util.spec_from_file_location("export_predictions_opencv_dnn_rtdetr", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOpenCVDNNRTDETRDecoding(unittest.TestCase):
    def test_normalize_boxes_accepts_transposed(self):
        mod = _load_module()
        np = mod.np
        if np is None:
            self.skipTest("numpy not available")

        raw = np.zeros((1, 4, 10), dtype=np.float32)
        out = mod._normalize_boxes_2d(raw)
        self.assertEqual(out.shape, (10, 4))

    def test_decode_logits_background_last_is_ignored(self):
        mod = _load_module()
        np = mod.np
        if np is None:
            self.skipTest("numpy not available")

        # Two queries, 3 classes (last is background). Make background highest.
        logits = np.array(
            [
                [0.0, 0.0, 10.0],
                [0.0, 5.0, 9.0],
            ],
            dtype=np.float32,
        )
        class_ids, scores = mod._decode_logits(
            logits,
            activation="softmax",
            background_class="last",
        )
        self.assertEqual(class_ids.shape, (2,))
        # background should be ignored, so first query picks class 0 or 1 (tie -> 0).
        self.assertEqual(int(class_ids[0]), 0)
        # second query should pick class 1 (since background ignored).
        self.assertEqual(int(class_ids[1]), 1)
        self.assertGreater(float(scores[1]), float(scores[0]))


if __name__ == "__main__":
    unittest.main()
