import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from yolozu.predictions.predictions_parity import compare_predictions


_MIN_JPEG_1X1 = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xC0,
        0x00,
        0x11,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x03,
        0x01,
        0x11,
        0x00,
        0x02,
        0x11,
        0x01,
        0x03,
        0x11,
        0x01,
        0xFF,
        0xD9,
    ]
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPredictionsParityTool(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_parity_ok(self):
        tool = _load_module(self.repo_root / "tools" / "check_predictions_parity.py", "check_predictions_parity")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = tmp / "000001.jpg"
            img.write_bytes(_MIN_JPEG_1X1)

            ref = [
                {
                    "image": str(img),
                    "detections": [
                        {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                        {"class_id": 1, "score": 0.1, "bbox": {"cx": 0.25, "cy": 0.25, "w": 0.1, "h": 0.1}},
                    ],
                }
            ]
            cand = [
                {
                    "image": str(img),
                    "detections": [
                        {"class_id": 1, "score": 0.1, "bbox": {"cx": 0.25, "cy": 0.25, "w": 0.1, "h": 0.1}},
                        {"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}},
                    ],
                }
            ]
            ref_path = tmp / "ref.json"
            cand_path = tmp / "cand.json"
            ref_path.write_text(json.dumps(ref))
            cand_path.write_text(json.dumps(cand))

            buf = io.StringIO()
            with redirect_stdout(buf):
                tool.main(["--reference", str(ref_path), "--candidate", str(cand_path)])
            report = json.loads(buf.getvalue())
            self.assertTrue(report["ok"])

    def test_parity_auto_accepts_xyxy_abs_boxes(self):
        tool = _load_module(self.repo_root / "tools" / "check_predictions_parity.py", "check_predictions_parity")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = tmp / "000001.jpg"
            img.write_bytes(_MIN_JPEG_1X1)

            ref = [
                {
                    "image": str(img),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50, "y2": 80},
                        }
                    ],
                }
            ]
            cand = [
                {
                    "image": str(img),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50, "y2": 80},
                        }
                    ],
                }
            ]
            ref_path = tmp / "ref.json"
            cand_path = tmp / "cand.json"
            ref_path.write_text(json.dumps(ref), encoding="utf-8")
            cand_path.write_text(json.dumps(cand), encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                tool.main(["--reference", str(ref_path), "--candidate", str(cand_path), "--image-size", "100,100"])
            report = json.loads(buf.getvalue())
            self.assertTrue(report["ok"])
            self.assertEqual(report["bbox_format"], "auto")

    def test_compare_predictions_reports_invalid_bbox_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = tmp / "000001.jpg"
            img.write_bytes(_MIN_JPEG_1X1)

            ref = [
                {
                    "image": str(img),
                    "detections": [
                        {"class_id": 0, "score": 0.9, "bbox": {"format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50}},
                    ],
                }
            ]
            cand = [
                {
                    "image": str(img),
                    "detections": [
                        {"class_id": 0, "score": 0.9, "bbox": {"format": "xyxy_abs", "x1": 10, "y1": 20, "x2": 50}},
                    ],
                }
            ]
            ref_path = tmp / "ref.json"
            cand_path = tmp / "cand.json"
            ref_path.write_text(json.dumps(ref), encoding="utf-8")
            cand_path.write_text(json.dumps(cand), encoding="utf-8")

            report = compare_predictions(reference=ref_path, candidate=cand_path, image_size=(100, 100))
            self.assertFalse(report["ok"])
            failures = report["results"][0]["failures"]
            self.assertEqual(failures[0]["type"], "invalid_reference_bbox")

    def test_parity_fails_on_bbox_mismatch(self):
        tool = _load_module(self.repo_root / "tools" / "check_predictions_parity.py", "check_predictions_parity")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img = tmp / "000001.jpg"
            img.write_bytes(_MIN_JPEG_1X1)

            ref = [{"image": str(img), "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]
            cand = [{"image": str(img), "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.6, "cy": 0.5, "w": 0.2, "h": 0.2}}]}]

            ref_path = tmp / "ref.json"
            cand_path = tmp / "cand.json"
            ref_path.write_text(json.dumps(ref))
            cand_path.write_text(json.dumps(cand))

            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    tool.main(["--reference", str(ref_path), "--candidate", str(cand_path), "--bbox-atol", "1e-6"])


if __name__ == "__main__":
    unittest.main()
