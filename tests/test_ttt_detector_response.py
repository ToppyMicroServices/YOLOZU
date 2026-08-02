import unittest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from yolozu.adapter import ModelAdapter
from yolozu.response_selection import detector_response_loss, select_foreground_queries
from yolozu.tta.config import TTTConfig
from yolozu.tta.integration import run_ttt


@unittest.skipIf(torch is None, "torch not installed")
class TestDetectorResponseSelection(unittest.TestCase):
    def test_selects_foreground_over_threshold_and_no_object(self):
        logits = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 3.0], [2.0, 1.0, 0.0]]])
        selected = select_foreground_queries(logits, confidence_min=0.5, topk=1)
        self.assertEqual(selected.mask.tolist(), [[True, False, False]])

    def test_response_loss_uses_class_and_box_for_selected_queries(self):
        teacher = {
            "logits": torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 3.0]]]),
            "bbox": torch.zeros(1, 2, 4),
        }
        student = {
            "logits": torch.tensor([[[2.0, 1.0, -1.0], [2.0, 0.0, 1.0]]], requires_grad=True),
            "bbox": torch.ones(1, 2, 4, requires_grad=True),
        }
        loss, metrics = detector_response_loss(
            student,
            teacher,
            confidence_min=0.5,
            topk=5,
            class_weight=1.0,
            bbox_weight=1.0,
            entropy_weight=0.05,
        )
        loss.backward()
        self.assertEqual(metrics["response_selected_queries"], 1.0)
        self.assertGreater(metrics["loss_response_class"], 0.0)
        self.assertGreater(metrics["loss_response_bbox"], 0.0)
        self.assertIsNotNone(student["logits"].grad)
        self.assertIsNotNone(student["bbox"].grad)

    def test_response_loss_rejects_negative_component_weight(self):
        outputs = {
            "logits": torch.zeros(1, 2, 3, requires_grad=True),
            "bbox": torch.zeros(1, 2, 4, requires_grad=True),
        }
        with self.assertRaisesRegex(ValueError, "class_weight must be >= 0"):
            detector_response_loss(
                outputs,
                {key: value.detach().clone() for key, value in outputs.items()},
                confidence_min=0.2,
                topk=2,
                class_weight=-1.0,
                bbox_weight=1.0,
                entropy_weight=0.05,
            )

    def test_response_loss_abstains_below_minimum_selection(self):
        teacher = {
            "logits": torch.tensor([[[4.0, 0.0, -1.0], [0.0, 0.0, 3.0]]]),
            "bbox": torch.zeros(1, 2, 4),
        }
        student = {
            "logits": torch.tensor([[[2.0, 1.0, -1.0], [2.0, 0.0, 1.0]]], requires_grad=True),
            "bbox": torch.ones(1, 2, 4, requires_grad=True),
        }
        loss, metrics = detector_response_loss(
            student,
            teacher,
            confidence_min=0.5,
            topk=5,
            min_selected=2,
            class_weight=1.0,
            bbox_weight=1.0,
            entropy_weight=0.05,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(metrics["response_selected_queries"], 1.0)
        self.assertEqual(metrics["response_abstained"], 1.0)

    def test_ttt_report_declares_detector_native_semantics(self):
        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(4)
                self.cls = nn.Linear(4, 3)
                self.box = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.norm(x).unsqueeze(1).repeat(1, 2, 1)
                logits = self.cls(hidden)
                logits[..., 0] = logits[..., 0] + 4.0
                return {"logits": logits, "bbox": self.box(hidden)}

        class Adapter(ModelAdapter):
            def __init__(self):
                self.model = Detector()

            def get_model(self):
                return self.model

            def build_loader(self, records, *, batch_size=1):
                yield torch.rand(2, 4)

            def predict(self, records):
                return []

        report = run_ttt(
            Adapter(),
            [{"image": "sample.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="tent",
                update_filter="norm_only",
                detector_response=True,
                response_conf_min=0.2,
            ),
        )
        self.assertEqual(report.forward_calls, 2)
        self.assertGreater(report.step_metrics[0]["response_selected_queries"], 0)
        semantics = report.method_profile["loss"]["detector_logits"]
        self.assertEqual(semantics["foreground_selection"], "teacher_confidence_and_no_object_margin")
        self.assertEqual(semantics["no_object_semantics"], "final_class_excluded_from_foreground_distillation")
        self.assertEqual(
            report.method_profile["abstention"]["effect"],
            "skip_backward_and_optimizer_step_and_restore_norm_buffers_when_no_auxiliary_loss_is_active",
        )

    def test_ttt_abstention_skips_backward_optimizer_and_parameter_update(self):
        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.BatchNorm1d(4)
                self.cls = nn.Linear(4, 3)
                self.box = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.norm(x).unsqueeze(1).repeat(1, 2, 1)
                return {"logits": self.cls(hidden), "bbox": self.box(hidden)}

        class Adapter(ModelAdapter):
            def __init__(self):
                self.model = Detector()

            def get_model(self):
                return self.model

            def build_loader(self, records, *, batch_size=1):
                yield torch.rand(2, 4)

            def predict(self, records):
                return []

        adapter = Adapter()
        before = {name: value.detach().clone() for name, value in adapter.model.named_parameters()}
        before_buffers = {name: value.detach().clone() for name, value in adapter.model.named_buffers()}
        report = run_ttt(
            adapter,
            [{"image": "sample.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="tent",
                steps=2,
                update_filter="norm_only",
                detector_response=True,
                response_conf_min=1.0,
                response_min_selected=1,
            ),
        )
        self.assertEqual(report.backward_calls, 0)
        self.assertEqual(report.optimizer_steps, 0)
        self.assertEqual(report.abstained_steps, 2)
        self.assertEqual(report.abstention_ratio, 1.0)
        self.assertTrue(all(step["update_abstained"] == 1.0 for step in report.step_metrics))
        self.assertTrue(
            all(step["buffers_restored_on_abstention"] == 1.0 for step in report.step_metrics)
        )
        for name, value in adapter.model.named_parameters():
            self.assertTrue(torch.equal(value.detach(), before[name]), name)
        for name, value in adapter.model.named_buffers():
            self.assertTrue(torch.equal(value.detach(), before_buffers[name]), name)


if __name__ == "__main__":
    unittest.main()
