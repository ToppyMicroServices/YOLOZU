import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MIN_JPEG_1X1 = bytes(
    [
        0xFF,
        0xD8,  # SOI
        0xFF,
        0xC0,  # SOF0
        0x00,
        0x11,  # length
        0x08,  # precision
        0x00,
        0x01,  # height = 1
        0x00,
        0x01,  # width = 1
        0x03,  # components
        0x01,
        0x11,
        0x00,  # comp 1
        0x02,
        0x11,
        0x01,  # comp 2
        0x03,
        0x11,
        0x01,  # comp 3
        0xFF,
        0xD9,  # EOI
    ]
)


class TestEvalSuiteExportSettings(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_eval_suite_propagates_meta_settings(self):
        tool = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "coco-yolo"
            images = dataset / "images" / "val2017"
            labels = dataset / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            img1 = images / "000001.jpg"
            img1.write_bytes(_MIN_JPEG_1X1)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            preds = [
                {
                    "image": str(img1),
                    "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                }
            ]
            payload = {
                "predictions": preds,
                "meta": {
                    "exporter": "onnxruntime",
                    "protocol_id": "yolo26",
                    "imgsz": 640,
                    "min_score": 0.123,
                    "topk": 456,
                },
            }

            pred_dir = tmp / "preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / "pred_yolo26n.json"
            pred_path.write_text(json.dumps(payload))

            out_path = tmp / "eval_suite.json"
            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--predictions-glob",
                        str(pred_path),
                        "--dry-run",
                        "--output",
                        str(out_path),
                    ]
                )

            suite = json.loads(out_path.read_text())
            self.assertEqual(len(suite.get("results", [])), 1)
            result = suite["results"][0]
            settings = result.get("export_settings") or {}
            self.assertEqual(settings.get("imgsz"), 640)
            self.assertAlmostEqual(settings.get("score_threshold"), 0.123, places=6)
            self.assertEqual(settings.get("max_detections"), 456)
            pp = settings.get("preprocess") or {}
            self.assertEqual(pp.get("method"), "letterbox")
            self.assertEqual(pp.get("input_color"), "RGB")

    def test_eval_suite_propagates_bundled_exporter_extra_settings(self):
        tool = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite_nested_extra")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "coco-yolo"
            images = dataset / "images" / "val"
            labels = dataset / "labels" / "val"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            image = images / "000001.jpg"
            image.write_bytes(_MIN_JPEG_1X1)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            payload = {
                "predictions": [
                    {
                        "image": "images/val/000001.jpg",
                        "detections": [
                            {
                                "class_id": 0,
                                "score": 0.9,
                                "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                            }
                        ],
                    }
                ],
                "meta": {
                    "adapter": "yolox",
                    "config": "/models/yolox_exp.py",
                    "extra": {
                        "exporter": "yolox",
                        "protocol_id": "nms_applied",
                        "export_settings": {
                            "imgsz": 640,
                            "score_threshold": 0.25,
                            "iou_threshold": 0.45,
                            "max_detections": 300,
                            "bbox_format": "cxcywh_norm",
                            "protocol": "nms_applied",
                            "preprocessing": {
                                "method": "letterbox",
                                "input_color": "BGR",
                                "normalize": "0_1",
                            },
                        },
                    },
                },
            }
            pred_path = tmp / "predictions.json"
            pred_path.write_text(json.dumps(payload), encoding="utf-8")

            out_path = tmp / "eval_suite.json"
            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--split",
                        "val",
                        "--predictions-glob",
                        str(pred_path),
                        "--dry-run",
                        "--strict",
                        "--output",
                        str(out_path),
                    ]
                )

            result = json.loads(out_path.read_text(encoding="utf-8"))["results"][0]
            self.assertEqual(result["predictions_meta_ref"]["exporter"], "yolox")
            self.assertEqual(result["predictions_meta_ref"]["protocol_id"], "nms_applied")
            settings = result["export_settings"]
            self.assertEqual(settings["imgsz"], 640)
            self.assertEqual(settings["score_threshold"], 0.25)
            self.assertEqual(settings["preprocess"], settings["preprocessing"])

    def test_eval_suite_protocol_hash_and_version_in_report(self):
        tool = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "coco-yolo"
            images = dataset / "images" / "val2017"
            labels = dataset / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            img1 = images / "000001.jpg"
            img1.write_bytes(_MIN_JPEG_1X1)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            preds = [
                {
                    "image": str(img1),
                    "preprocess": {
                        "method": "letterbox",
                        "normalize": "0_1",
                        "input_size": {"width": 640, "height": 640},
                        "input_color": "RGB",
                        "resize_interp": "linear",
                        "letterbox_fill": [114, 114, 114],
                    },
                    "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                }
            ]
            payload = {
                "predictions": preds,
                "meta": {
                    "exporter": "onnxruntime",
                    "protocol_id": "yolo26",
                    "imgsz": 640,
                    "min_score": 0.001,
                    "iou": 0.7,
                    "max_det": 300,
                },
            }

            pred_dir = tmp / "preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / "pred_yolo26n.json"
            pred_path.write_text(json.dumps(payload))

            out_path = tmp / "eval_suite_protocol.json"
            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--protocol",
                        "yolo26",
                        "--dataset",
                        str(dataset),
                        "--predictions-glob",
                        str(pred_path),
                        "--dry-run",
                        "--output",
                        str(out_path),
                    ]
                )

            suite = json.loads(out_path.read_text())
            self.assertEqual(suite.get("protocol_id"), "yolo26")
            self.assertEqual(suite.get("protocol_schema_version"), 1)
            self.assertIsInstance(suite.get("protocol_hash"), str)
            self.assertEqual(len(suite.get("protocol_hash")), 64)
            result = suite["results"][0]
            self.assertTrue(result.get("protocol_check", {}).get("ok"))
            self.assertEqual(result.get("protocol_check", {}).get("mismatches"), [])

    def test_eval_suite_protocol_mismatch_fails(self):
        tool = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "coco-yolo"
            images = dataset / "images" / "val2017"
            labels = dataset / "labels" / "val2017"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)

            img1 = images / "000001.jpg"
            img1.write_bytes(_MIN_JPEG_1X1)
            (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            payload = {
                "predictions": [
                    {
                        "image": str(img1),
                        "detections": [{"class_id": 0, "score": 0.9, "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}}],
                    }
                ],
                "meta": {
                    "exporter": "onnxruntime",
                    "protocol_id": "yolo26",
                    "imgsz": 640,
                    "min_score": 0.123,
                    "iou": 0.7,
                    "max_det": 300,
                },
            }

            pred_path = tmp / "pred_yolo26n.json"
            pred_path.write_text(json.dumps(payload))

            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(io.StringIO()):
                    tool.main(
                        [
                            "--protocol",
                            "yolo26",
                            "--dataset",
                            str(dataset),
                            "--predictions-glob",
                            str(pred_path),
                            "--dry-run",
                            "--output",
                            str(tmp / "eval_suite_protocol_mismatch.json"),
                        ]
                    )
            self.assertIn("fixed-condition protocol validation failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
