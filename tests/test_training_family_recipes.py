import importlib.util
from pathlib import Path
import unittest

import yaml

from yolozu.training.platform import get_training_backend_spec


class TestTrainingFamilyRecipes(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_rtdetr_stable_recipe_uses_detr_optimizer_policy(self) -> None:
        cfg_path = self.repo_root / "configs" / "examples" / "train_rtdetr_stable.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        self.assertEqual(cfg["optimizer"], "adamw")
        self.assertTrue(cfg["use_param_groups"])
        self.assertLess(float(cfg["backbone_lr_mult"]), float(cfg["head_lr_mult"]))
        self.assertGreater(float(cfg["clip_grad_norm"]), 0.0)
        self.assertGreater(int(cfg["lr_warmup_steps"]), 0)
        self.assertTrue(cfg["use_ema"])
        self.assertNotIn("nms", cfg)

        spec = get_training_backend_spec("reference-rtdetr-pose")
        self.assertEqual(spec.training_family, "rtdetr")
        self.assertIn("AdamW", spec.optimizer_policy or "")
        self.assertIn("NMS-free", spec.postprocess_policy or "")
        self.assertIn("gradient_clipping", spec.stability_policy)

    def test_yolox_recipe_keeps_yolo_letterbox_sgd_nms_boundary(self) -> None:
        exp_path = self.repo_root / "configs" / "examples" / "finetune_external" / "yolox_s_finetune_smoke.py"
        spec = importlib.util.spec_from_file_location("yolox_s_finetune_smoke", exp_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        exp = module.get_exp()

        self.assertNotIn("optimizer", vars(exp))
        self.assertNotIn("preprocess", vars(exp))
        self.assertEqual(exp.decode_postprocess, "nms")
        self.assertGreater(float(exp.nmsthre), 0.0)
        self.assertTrue(bool(exp.nesterov))

        lane = get_training_backend_spec("yolox")
        self.assertEqual(lane.training_family, "yolo")
        self.assertIn("SGD", lane.optimizer_policy or "")
        self.assertIn("letterbox", lane.preprocess_policy or "")
        self.assertIn("NMS-applied", lane.postprocess_policy or "")

    def test_training_capability_docs_describe_family_policies(self) -> None:
        docs_path = self.repo_root / "docs" / "training_capability_matrix.md"
        text = docs_path.read_text(encoding="utf-8")

        self.assertIn("Detector-family policy", text)
        self.assertIn("`reference-rtdetr-pose` | RT-DETR / DETR-family | AdamW", text)
        self.assertIn("lower backbone LR", text)
        self.assertIn("`yolox` | YOLO-family | SGD", text)
        self.assertIn("Letterbox resize/pad", text)
        self.assertIn("NMS-applied", text)


if __name__ == "__main__":
    unittest.main()
