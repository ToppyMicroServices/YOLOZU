import importlib.util
import sys
from pathlib import Path
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolozu.adapter import RTDETRPoseAdapter


class TestRTDETRPoseAdapter(unittest.TestCase):
    def test_torch_accel_options_are_stored(self):
        adapter = RTDETRPoseAdapter(
            amp="bf16",
            channels_last=True,
            use_inference_mode=False,
        )
        self.assertEqual(adapter.amp, "bf16")
        self.assertTrue(adapter.channels_last)
        self.assertFalse(adapter.use_inference_mode)

    def test_requires_torch_when_used(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed; this test covers the no-torch path")

        adapter = RTDETRPoseAdapter()
        with self.assertRaises(RuntimeError) as ctx:
            adapter.predict([{"image": "does-not-exist.jpg", "labels": []}])
        self.assertIn("requires 'torch'", str(ctx.exception))

    def test_preprocess_shape_range_and_determinism(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "toy.jpg"
            arr = (np.arange(7 * 5 * 3, dtype=np.uint8).reshape(5, 7, 3) % 255)
            Image.fromarray(arr, mode="RGB").save(img_path)

            adapter = RTDETRPoseAdapter(image_size=(32, 32))
            adapter._ensure_backend()
            preprocess = adapter._backend["preprocess"]

            record = {"image": str(img_path)}
            x1, meta1, _ = preprocess(record)
            x2, meta2, _ = preprocess(record)

            self.assertEqual(tuple(x1.shape), (1, 3, 32, 32))
            self.assertEqual(tuple(x2.shape), (1, 3, 32, 32))
            self.assertTrue(bool((x1 >= 0.0).all().item()))
            self.assertTrue(bool((x1 <= 1.0).all().item()))
            self.assertTrue(bool((x1 == x2).all().item()))
            self.assertEqual(meta1.get("method"), "resize")
            self.assertEqual(meta1.get("normalize"), "0_1")
            self.assertEqual(meta1.get("input_size"), {"width": 32, "height": 32})
            self.assertEqual(meta1.get("model_input_size"), {"width": 32, "height": 32})
            self.assertEqual(meta1.get("resize", {}).get("algorithm"), "bilinear")
            self.assertEqual(meta1.get("pad"), {"left": 0, "top": 0, "right": 0, "bottom": 0})
            self.assertFalse(bool(meta1.get("letterbox")))
            self.assertEqual(meta1.get("color_order"), "RGB")
            self.assertEqual(meta1.get("dtype"), "float32")
            self.assertEqual(meta1.get("exif_orientation"), "normalized")
            self.assertEqual(meta1, meta2)

    def test_preprocess_scales_intrinsics(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "toy.jpg"
            arr = np.zeros((10, 20, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(img_path)

            adapter = RTDETRPoseAdapter(image_size=(40, 20))
            adapter._ensure_backend()
            preprocess = adapter._backend["preprocess"]

            record = {
                "image": str(img_path),
                "intrinsics": {"fx": 100.0, "fy": 200.0, "cx": 10.0, "cy": 5.0},
            }
            _, meta, intr = preprocess(record)
            self.assertIsInstance(intr, dict)
            # orig (w,h)=(20,10) -> dst (w,h)=(40,20) => sx=2, sy=2
            self.assertAlmostEqual(float(intr["fx"]), 200.0, places=6)
            self.assertAlmostEqual(float(intr["fy"]), 400.0, places=6)
            self.assertAlmostEqual(float(intr["cx"]), 20.0, places=6)
            self.assertAlmostEqual(float(intr["cy"]), 10.0, places=6)
            self.assertEqual(meta.get("orig_size"), {"width": 20, "height": 10})
            self.assertEqual(meta.get("input_size"), {"width": 40, "height": 20})

    def test_predict_requires_image_key(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        adapter = RTDETRPoseAdapter(image_size=(32, 32), init_seed=2026)
        with self.assertRaises(ValueError) as ctx:
            adapter.predict([{"labels": []}])
        self.assertIn("records[0].image", str(ctx.exception))

    def test_predict_is_deterministic_with_init_seed(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        from PIL import Image
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "toy.jpg"
            arr = (np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3) % 255)
            Image.fromarray(arr, mode="RGB").save(img_path)
            records = [{"image": str(img_path)}]

            adapter_a = RTDETRPoseAdapter(
                image_size=(32, 32),
                score_threshold=0.0,
                max_detections=5,
                init_seed=2026,
            )
            adapter_b = RTDETRPoseAdapter(
                image_size=(32, 32),
                score_threshold=0.0,
                max_detections=5,
                init_seed=2026,
            )

            out_a = adapter_a.predict(records)
            out_b = adapter_b.predict(records)
            self.assertEqual(out_a, out_b)
            entry = out_a[0]
            self.assertIn("image_w", entry)
            self.assertIn("image_h", entry)
            self.assertIn("orig_w", entry)
            self.assertIn("orig_h", entry)
            self.assertIn("model_input_w", entry)
            self.assertIn("model_input_h", entry)
            self.assertIn("preproc", entry)


if __name__ == "__main__":
    unittest.main()
