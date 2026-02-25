import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
