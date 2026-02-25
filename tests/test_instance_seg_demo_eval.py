import unittest
from pathlib import Path
import tempfile
import json


class TestInstanceSegDemoEval(unittest.TestCase):
    def test_instance_seg_eval_with_relative_pred_root(self):
        # Regression test: pred_root may be a *relative* path (common in demos).
        # Evaluation must still be able to resolve predicted mask paths.
        from yolozu.demos.instance_seg import run_instance_seg_demo

        out = run_instance_seg_demo(
            run_dir=Path("runs") / "unit-test-instance-seg-demo",
            seed=0,
            num_images=4,
            image_size=64,
            max_instances=2,
        )
        payload = __import__("json").loads(Path(out).read_text(encoding="utf-8"))
        result = payload.get("result") or {}

        artifacts = payload.get("artifacts") or {}
        overlays_dir = Path(str(artifacts.get("overlays_dir") or ""))
        self.assertTrue(overlays_dir.is_dir(), "expected overlays_dir artifact")
        overlays = sorted(overlays_dir.glob("overlay_img_*.png"))
        self.assertGreaterEqual(len(overlays), 1, "expected at least one overlay png")

        # Should produce at least some true positives; if masks can't be loaded, mAP collapses to 0.
        self.assertGreater(float(result.get("map50", 0.0)), 0.0)

    def test_instance_seg_demo_coco_instances_polygons(self):
        from yolozu.demos.instance_seg import run_instance_seg_demo

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            td_path = Path(td)
            images_dir = td_path / "coco_images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create a tiny image.
            img_path = images_dir / "000000000001.jpg"
            try:
                from PIL import Image
            except Exception as exc:
                raise unittest.SkipTest("Pillow not installed") from exc
            img = Image.new("RGB", (64, 48), (240, 240, 240))
            img.save(img_path)

            # Fake COCO instances JSON with a single polygon annotation.
            instances_path = td_path / "instances_val.json"
            coco = {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "iscrowd": 0,
                        "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                    }
                ],
                "categories": [{"id": 3, "name": "thing"}],
            }
            instances_path.write_text(json.dumps(coco), encoding="utf-8")

            out = run_instance_seg_demo(
                run_dir=td_path / "run",
                seed=0,
                num_images=1,
                max_instances=1,
                background="coco-instances",
                coco_instances_json=instances_path,
                coco_images_dir=images_dir,
            )
            payload = json.loads(Path(out).read_text(encoding="utf-8"))
            result = payload.get("result") or {}
            self.assertGreater(float(result.get("map50", 0.0)), 0.0)

            artifacts = payload.get("artifacts") or {}
            overlays_dir = Path(str(artifacts.get("overlays_dir") or ""))
            overlays = sorted(overlays_dir.glob("overlay_img_*.png"))
            self.assertGreaterEqual(len(overlays), 1, "expected at least one overlay png")


if __name__ == "__main__":
    unittest.main()
