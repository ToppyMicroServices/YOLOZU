import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _write_stub_png(path: Path, *, width: int, height: int) -> None:
    # Minimal PNG header sufficient for yolozu.image_size.get_image_size (no CRC validation).
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_len = (13).to_bytes(4, "big", signed=False)
    ihdr = b"IHDR"
    w = int(width).to_bytes(4, "big", signed=False)
    h = int(height).to_bytes(4, "big", signed=False)
    # bitdepth=8, colortype=2, compression=0, filter=0, interlace=0
    rest = bytes([8, 2, 0, 0, 0])
    path.write_bytes(sig + ihdr_len + ihdr + w + h + rest)


class TestCocoKeypointsEval(unittest.TestCase):
    def test_build_gt_and_convert_predictions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        from yolozu.coco_keypoints_eval import build_coco_keypoints_ground_truth, predictions_to_coco_keypoints

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            img_path = root / "0001.png"
            _write_stub_png(img_path, width=64, height=32)

            records = [
                {
                    "image": str(img_path),
                    "labels": [
                        {
                            "class_id": 0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.4,
                            "h": 0.4,
                            "keypoints": [0.50, 0.50, 2, 0.60, 0.50, 1],
                        }
                    ],
                }
            ]

            gt, index = build_coco_keypoints_ground_truth(records, keypoints_format="xy_norm")
            self.assertEqual(int(index.keypoints_count), 2)
            self.assertEqual(len(gt.get("images") or []), 1)
            self.assertEqual(len(gt.get("annotations") or []), 1)

            ann = (gt.get("annotations") or [])[0]
            self.assertEqual(len(ann.get("keypoints") or []), 6)
            # kp0 x=0.50*64=32, y=0.50*32=16
            self.assertAlmostEqual(float(ann["keypoints"][0]), 32.0, places=6)
            self.assertAlmostEqual(float(ann["keypoints"][1]), 16.0, places=6)
            self.assertEqual(int(ann["keypoints"][2]), 2)
            self.assertEqual(int(ann["num_keypoints"]), 2)

            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt.get("images") or []}
            preds = [
                {
                    "image": str(img_path),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                            "keypoints": [0.50, 0.50, 1, 0.60, 0.50, 0.2],
                        }
                    ],
                }
            ]
            dt = predictions_to_coco_keypoints(
                preds,
                coco_index=index,
                image_sizes=image_sizes,
                keypoints_format="xy_norm",
            )
            self.assertEqual(len(dt), 1)
            self.assertEqual(len(dt[0].get("keypoints") or []), 6)
            self.assertAlmostEqual(float(dt[0]["keypoints"][5]), 0.2, places=6)

    def test_predictions_windows_style_image_key(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        from yolozu.coco_keypoints_eval import build_coco_keypoints_ground_truth, predictions_to_coco_keypoints

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            img_path = root / "0001.png"
            _write_stub_png(img_path, width=64, height=32)

            records = [
                {
                    "image": str(img_path),
                    "labels": [
                        {
                            "class_id": 0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.4,
                            "h": 0.4,
                            "keypoints": [0.50, 0.50, 2, 0.60, 0.50, 1],
                        }
                    ],
                }
            ]

            gt, index = build_coco_keypoints_ground_truth(records, keypoints_format="xy_norm")
            image_sizes = {img["id"]: (int(img["width"]), int(img["height"])) for img in gt.get("images") or []}
            preds = [
                {
                    "image": str(img_path).replace("/", "\\"),
                    "detections": [
                        {
                            "class_id": 0,
                            "score": 0.9,
                            "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4},
                            "keypoints": [0.50, 0.50, 1, 0.60, 0.50, 0.2],
                        }
                    ],
                }
            ]
            dt = predictions_to_coco_keypoints(
                preds,
                coco_index=index,
                image_sizes=image_sizes,
                keypoints_format="xy_norm",
            )
            self.assertEqual(len(dt), 1)

    def test_oks_requires_pycocotools(self) -> None:
        from yolozu.coco_keypoints_eval import evaluate_coco_oks_map

        missing_modules = {
            "pycocotools": None,
            "pycocotools.coco": None,
            "pycocotools.cocoeval": None,
        }
        with mock.patch.dict(sys.modules, missing_modules):
            with self.assertRaises(RuntimeError):
                evaluate_coco_oks_map({"images": [], "annotations": [], "categories": []}, [])


if __name__ == "__main__":
    unittest.main()
