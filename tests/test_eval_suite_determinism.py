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


def _without_timestamp(payload: dict) -> dict:
    copied = dict(payload)
    copied.pop("timestamp", None)
    return copied


class TestEvalSuiteDeterminism(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_eval_suite_is_deterministic_for_fixed_fixture(self):
        tool = _load_module(self.repo_root / "tools" / "eval_suite.py", "eval_suite")

        def fake_evaluate_coco_map(gt, dt):
            detections = len(dt)
            images = len(gt.get("images", []))
            base = round(detections / 100.0, 6)
            return {
                "metrics": {
                    "map50_95": base,
                    "map50": round(base + 0.1, 6),
                    "map75": round(base - 0.05, 6),
                    "ar100": round(base + 0.2, 6),
                },
                "stats": [base, round(base + 0.1, 6), round(base - 0.05, 6), round(base + 0.2, 6)],
                "dry_run": False,
                "counts": {"images": images, "detections": detections},
            }

        tool.evaluate_coco_map = fake_evaluate_coco_map

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

            pred_path = tmp / "pred_yolo26n.json"
            pred_path.write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "image": str(img1),
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
                            "exporter": "onnxruntime",
                            "protocol_id": "yolo26",
                            "imgsz": 640,
                            "min_score": 0.001,
                            "iou": 0.7,
                            "max_det": 300,
                        },
                    }
                )
            )

            out1 = tmp / "eval_suite_1.json"
            out2 = tmp / "eval_suite_2.json"

            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--predictions-glob",
                        str(pred_path),
                        "--output",
                        str(out1),
                    ]
                )
            with redirect_stdout(io.StringIO()):
                tool.main(
                    [
                        "--dataset",
                        str(dataset),
                        "--predictions-glob",
                        str(pred_path),
                        "--output",
                        str(out2),
                    ]
                )

            suite1 = json.loads(out1.read_text())
            suite2 = json.loads(out2.read_text())
            self.assertEqual(_without_timestamp(suite1), _without_timestamp(suite2))

            result1 = suite1["results"][0]
            result2 = suite2["results"][0]
            self.assertEqual(result1.get("metrics"), result2.get("metrics"))


if __name__ == "__main__":
    unittest.main()