import tempfile
import unittest
from pathlib import Path


class TestDatasetKeypoints(unittest.TestCase):
    def test_parse_yolo_pose_label_xyv(self):
        repo_root = Path(__file__).resolve().parents[1]

        Image = None
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Pillow not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            img = root / "images" / "train2017" / "000000000001.jpg"
            Image.new("RGB", (32, 24), color=(0, 0, 0)).save(img)

            lab = root / "labels" / "train2017" / "000000000001.txt"
            # class cx cy w h + 2 keypoints (x,y,v)*2
            lab.write_text("0 0.5 0.5 0.4 0.4 0.25 0.25 2 0.75 0.25 1\n", encoding="utf-8")

            from yolozu.dataset import build_manifest

            manifest = build_manifest(root, split="train2017")
            records = manifest["images"]
            self.assertEqual(len(records), 1)
            labels = records[0]["labels"]
            self.assertEqual(len(labels), 1)
            self.assertIn("keypoints", labels[0])
            kps = labels[0]["keypoints"]
            self.assertIsInstance(kps, list)
            self.assertEqual(len(kps), 2)
            self.assertAlmostEqual(float(kps[0]["x"]), 0.25)
            self.assertAlmostEqual(float(kps[0]["y"]), 0.25)
            self.assertAlmostEqual(float(kps[0]["v"]), 2.0)

    def test_parse_yolo_pose_label_xy(self):
        repo_root = Path(__file__).resolve().parents[1]

        Image = None
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Pillow not available: {exc}")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            (root / "images" / "train2017").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "train2017").mkdir(parents=True, exist_ok=True)

            img = root / "images" / "train2017" / "000000000001.jpg"
            Image.new("RGB", (32, 24), color=(0, 0, 0)).save(img)

            lab = root / "labels" / "train2017" / "000000000001.txt"
            # class cx cy w h + 2 keypoints (x,y)*2
            lab.write_text("0 0.5 0.5 0.4 0.4 0.25 0.25 0.75 0.25\n", encoding="utf-8")

            from yolozu.dataset import build_manifest

            manifest = build_manifest(root, split="train2017")
            records = manifest["images"]
            labels = records[0]["labels"]
            kps = labels[0]["keypoints"]
            self.assertEqual(len(kps), 2)
            self.assertNotIn("v", kps[0])


if __name__ == "__main__":
    unittest.main()
