import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from yolozu.tta.cnn_experiment import (
    ExperimentConfig, adapt, discover_images, evaluation_records,
    run_experiment, select_pseudo_target,
)

try:
    import torch
    from yolozu.tta.quantized_lora import base_fingerprint, prepare_model
except ImportError:
    torch = None


if torch is not None:
    class ToyDetector(torch.nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))

        def forward(self, images, targets=None):
            feature = self.features(images[0].unsqueeze(0)).mean()
            if self.training:
                return {"loss_classifier": (feature - targets[0]["labels"].float().mean()).square()}
            height, width = images[0].shape[-2:]
            return [{"boxes": torch.tensor([[0.25 * width, 0.25 * height, 0.75 * width, 0.75 * height]],
                                            device=feature.device),
                     "scores": (0.85 + 0.1 * feature.sigmoid()).reshape(1),
                     "labels": torch.tensor([1], dtype=torch.int64, device=feature.device)}]


class TestCNNQLoRATTT(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        images = root / "data" / "images" / "val2017"
        labels = root / "data" / "labels" / "val2017"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16), (40 + 20 * index, 100, 150)).save(images / f"{index}.png")
        # There is deliberately no support label file: adapting must never read it.
        for index in range(1, 4):
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        checkpoint = root / "weights.pt"
        checkpoint.write_bytes(b"test-only factory checkpoint")
        return ExperimentConfig(dataset=str(root / "data"), checkpoint=str(checkpoint),
                                output=str(root / "out"), support_images=1, eval_images=2,
                                adapter_include=("features.0",), min_size=64, max_size=64,
                                skip_eval=True, threads=1)

    def test_cli_help(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "run_cnn_qlora_ttt.py"
        result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--adapter-include", result.stdout)
        self.assertIn("--skip-eval", result.stdout)

    def test_duplicate_bytes_across_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            images = Path(config.dataset) / "images" / config.split
            (images / "1.png").write_bytes((images / "0.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "identical image bytes"):
                discover_images(config)
            self.assertFalse(Path(config.output).exists())

    def test_disjoint_paths_and_eval_labels_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            support, evaluation = discover_images(config)
            self.assertFalse(set(support) & set(evaluation))
            self.assertEqual(len(evaluation_records(evaluation, config)), 2)
            self.assertFalse((Path(config.dataset) / "labels" / config.split / "0.txt").exists())

    def test_bounds_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            for update in ({"steps": 0}, {"eval_images": 1000}, {"learning_rate": float("nan")},
                           {"brightness": 0}, {"split": "../val"}, {"device": "mps"}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    replace(config, **update).validate()

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_pseudo_target_excludes_invalid_low_confidence_and_degenerate(self):
        output = {"boxes": torch.tensor([[0., 0., 2., 2.], [1., 1., 0., 0.], [0., 0., 2., 2.], [0., 0., 2., 2.]]),
                  "scores": torch.tensor([.9, .99, float("nan"), .2]),
                  "labels": torch.tensor([1, 1, 1, 1])}
        target = select_pseudo_target(output, threshold=.7, topk=10)
        self.assertEqual(target["boxes"].shape, (1, 4))
        self.assertEqual(target["labels"].tolist(), [1])

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_adaptation_preserves_base_and_batchnorm_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            model = ToyDetector().eval()
            prepare_model(model, quantize=True, rank=2, adapter_include=("features.0",))
            before = base_fingerprint(model)
            result = adapt(model, [torch.full((3, 16, 16), .3)], config)
            self.assertEqual(result["status"], "updated")
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertEqual(before, base_fingerprint(model))
            self.assertEqual(int(model.features[1].num_batches_tracked), 0)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_no_confident_targets_abstains_without_update(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.fixture(directory), pseudo_threshold=1.0)
            model = ToyDetector().eval()
            prepare_model(model, quantize=False, rank=2, adapter_include=("features.0",))
            result = adapt(model, [torch.full((3, 16, 16), .3)], config)
            self.assertEqual(result["status"], "no_update")
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["pseudo_label_counts"], [0])

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_nonfinite_second_step_rolls_back_first_update(self):
        class FailingDetector(ToyDetector):
            training_calls = 0

            def forward(self, images, targets=None):
                result = super().forward(images, targets)
                if self.training:
                    self.training_calls += 1
                    if self.training_calls == 2:
                        result = {key: value * float("nan") for key, value in result.items()}
                return result

        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.fixture(directory), steps=2)
            model = FailingDetector().eval()
            prepare_model(model, quantize=True, rank=2, adapter_include=("features.0",))
            result = adapt(model, [torch.full((3, 16, 16), .3)], config)
            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(result["stop_reason"], "non_finite_loss")
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertFalse(result["adapter_changed"])
            self.assertTrue(result["base_unchanged"])

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_four_arms_actual_updates_storage_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            report = run_experiment(config, model_factory=ToyDetector, category_mapping={1: 0})
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["promotion_eligible"])
            self.assertEqual(report["efficacy_conclusion"], "not_established")
            self.assertEqual(report["evaluation"]["status"], "skipped")
            self.assertEqual(set(report["arms"]), {"float_fixed", "int4_fixed", "float_lora_ttt", "int4_lora_ttt"})
            for name, arm in report["arms"].items():
                self.assertTrue(arm["base_unchanged"])
                self.assertEqual(arm["compute_dtype"], "float32")
                self.assertIsNone(arm["memory"]["cuda_peak_allocated_bytes"])
                if arm["ttt"]:
                    self.assertTrue(arm["adapter_reload_prediction_parity"])
                    self.assertTrue(arm["adaptation"]["adapter_changed"])
                output = Path(config.output) / name
                self.assertTrue((output / "shifted_overlay.png").is_file())
                self.assertEqual(len(json.loads((output / "clean_predictions.json").read_text())["predictions"]), 2)
            quantized = report["arms"]["int4_fixed"]["preparation"]
            self.assertGreater(quantized["quantized_layer_count"], 0)
            self.assertLess(quantized["base_storage_bytes_after"], quantized["base_storage_bytes_before"])
            before = (Path(config.output) / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_experiment(config, model_factory=ToyDetector, category_mapping={1: 0})
            self.assertEqual(before, (Path(config.output) / "report.json").read_bytes())

    @unittest.skipIf(torch is None or importlib.util.find_spec("pycocotools") is None, "torch/pycocotools unavailable")
    def test_coco_metric_and_evaluation_label_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.fixture(directory), skip_eval=False)
            report = run_experiment(config, model_factory=ToyDetector, category_mapping={1: 0})
            self.assertEqual(report["evaluation"]["metric_semantics"], "pycocotools_coco_bbox")
            self.assertEqual(set(report["evaluation_label_sha256"]), {"1.txt", "2.txt"})
            self.assertIn("cnn_experiment.py", report["source_sha256"])
            self.assertIn("run_cnn_qlora_ttt.py", report["source_sha256"])
            for arm in report["arms"].values():
                for domain in ("clean", "shifted"):
                    self.assertAlmostEqual(arm[domain]["evaluation"]["metrics"]["map50_95"], 1.0)
            self.assertEqual(report["deltas"]["int4_lora_ttt"]["shifted"], 0.0)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_failure_report_does_not_claim_success(self):
        def broken(config):
            raise RuntimeError("deliberate checkpoint failure")

        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "deliberate checkpoint failure"):
                run_experiment(config, model_factory=broken, category_mapping={1: 0})
            report = json.loads((Path(config.output) / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["promotion_eligible"])
            self.assertEqual(report["arms"], {})


if __name__ == "__main__":
    unittest.main()
