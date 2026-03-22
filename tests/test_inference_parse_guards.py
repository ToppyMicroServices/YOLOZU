import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import build_trt_engine
from yolozu.inference import export_orchestrator, infer_constraints
from yolozu.inference.predict_images import _render_overlays


class TestInferenceParseGuards(unittest.TestCase):
    def test_infer_constraints_ignores_malformed_numeric_depth(self):
        entries = [
            {
                "image": "x.jpg",
                "image_size": {"width": "bad", "height": object()},
                "detections": [
                    {
                        "class_id": 1,
                        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                        "log_z": "nan-not-a-number",
                    }
                ],
            }
        ]

        out = infer_constraints(entries, constraints_cfg={"enabled": {}}, bbox_format="cxcywh_norm")
        det = out[0]["detections"][0]
        self.assertNotIn("t_xyz", det)
        self.assertNotIn("constraints", det)

    def test_output_config_hash_returns_none_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(export_orchestrator.output_config_hash(path))

    def test_build_trt_engine_run_capture_returns_none_on_spawn_error(self):
        with mock.patch("tools.build_trt_engine.subprocess.check_output", side_effect=OSError("boom")):
            self.assertIsNone(build_trt_engine._run_capture(["trtexec", "--version"]))

    def test_render_overlays_skips_bad_bbox_values(self):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"Pillow not available: {exc}")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_path = root / "sample.png"
            Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path)
            overlays_dir = root / "overlays"
            payload = {
                "predictions": [
                    {
                        "image": str(image_path),
                        "detections": [
                            {"bbox": {"cx": "bad", "cy": 0.5, "w": 0.2, "h": 0.2}},
                            {"bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                        ],
                    }
                ]
            }
            report = _render_overlays(payload=payload, overlays_dir=overlays_dir, max_images=1)
            self.assertEqual(report["count"], 1)
            self.assertTrue(any(overlays_dir.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
