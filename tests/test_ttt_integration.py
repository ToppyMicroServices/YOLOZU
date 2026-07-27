import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from yolozu.adapter import DummyAdapter, ModelAdapter
from yolozu.tta.config import TTTConfig
from yolozu.tta import integration as integration_module
from yolozu.tta.integration import run_ttt


class TestTTTMemoryPortability(unittest.TestCase):
    def test_process_memory_gracefully_degrades_without_resource_module(self):
        class CpuParameter:
            class Device:
                type = "cpu"

            device = Device()

        class CpuModel:
            @staticmethod
            def parameters():
                return iter([CpuParameter()])

        with unittest.mock.patch.object(integration_module, "resource", None):
            self.assertIsNone(integration_module._process_peak_rss_bytes())
            self.assertEqual(
                integration_module._memory_start(CpuModel()),
                {
                    "backend": "unavailable",
                    "metric": None,
                    "reason": "resource_module_unavailable",
                },
            )


@unittest.skipIf(torch is None, "torch not installed")
class TestTTTIntegration(unittest.TestCase):
    def test_run_ttt_tent_rollback_restores_norm_buffers(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.fc = nn.Linear(4, 3)

            def forward(self, x):
                x = self.bn(x)
                return self.fc(x)

        class Adapter(ModelAdapter):
            def __init__(self, model):
                self._model = model

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.randn(2, 4)

        torch.manual_seed(0)
        base_model = Model()
        base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

        adapter_rollback = Adapter(Model())
        adapter_rollback.get_model().load_state_dict(base_state, strict=True)
        bn0 = adapter_rollback.get_model().bn
        base_buffers = {
            "running_mean": bn0.running_mean.detach().clone(),
            "running_var": bn0.running_var.detach().clone(),
            "num_batches_tracked": bn0.num_batches_tracked.detach().clone(),
        }
        run_ttt(
            adapter_rollback,
            [{"image": "a.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="tent",
                steps=1,
                lr=1e-3,
                update_filter="norm_only",
                rollback_on_stop=True,
                max_loss_ratio=0.9,
            ),
        )
        bn1 = adapter_rollback.get_model().bn
        self.assertTrue(torch.allclose(bn1.running_mean, base_buffers["running_mean"]))
        self.assertTrue(torch.allclose(bn1.running_var, base_buffers["running_var"]))
        self.assertTrue(torch.equal(bn1.num_batches_tracked, base_buffers["num_batches_tracked"]))

        adapter_no_rollback = Adapter(Model())
        adapter_no_rollback.get_model().load_state_dict(base_state, strict=True)
        run_ttt(
            adapter_no_rollback,
            [{"image": "a.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="tent",
                steps=1,
                lr=1e-3,
                update_filter="norm_only",
                rollback_on_stop=False,
                max_loss_ratio=0.9,
            ),
        )
        bn2 = adapter_no_rollback.get_model().bn
        self.assertFalse(torch.equal(bn2.num_batches_tracked, base_buffers["num_batches_tracked"]))

    def test_run_ttt_tent(self):
        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = nn.Sequential(nn.Linear(4, 3))

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.randn(2, 4)

        adapter = Adapter()
        records = [{"image": "a.jpg"}, {"image": "b.jpg"}]
        report = run_ttt(adapter, records, config=TTTConfig(enabled=True, method="tent", steps=2, lr=1e-3))
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "tent")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)
        self.assertEqual(report.forward_calls, 2)
        self.assertEqual(report.backward_calls, 2)
        self.assertEqual(report.optimizer_steps, 2)
        self.assertEqual(report.update_ratio, 1.0)
        self.assertEqual((report.memory or {}).get("metric"), "peak_rss_bytes")

    def test_run_ttt_mim(self):
        class ReconModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)

            def forward(self, x):
                return self.conv(x)

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = ReconModel()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(1, 3, 8, 8)

        adapter = Adapter()
        records = [{"image": "a.jpg"}]
        report = run_ttt(
            adapter,
            records,
            config=TTTConfig(
                enabled=True,
                method="mim",
                steps=2,
                lr=1e-3,
                mim_mask_prob=0.5,
                mim_patch_size=2,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "mim")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)
        self.assertEqual(report.forward_calls, 2)
        self.assertEqual(report.backward_calls, 2)
        self.assertEqual(report.optimizer_steps, 2)

    def test_run_ttt_mim_structured_branch(self):
        class StructuredMIMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.enable_mim = True
                self.norm = nn.BatchNorm2d(3)
                self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)

            def forward(self, x, geom_input=None, feature_mask=None, return_mim=False):
                feat = self.norm(self.conv(x))
                logits = feat.mean(dim=(2, 3)).unsqueeze(1).repeat(1, 2, 1)
                out = {"logits": logits}
                if return_mim:
                    teacher = feat.detach()
                    out["mim"] = {
                        "recon_feat": feat,
                        "teacher_feat": teacher,
                        "mask": feature_mask,
                        "entropy": logits.abs().mean(),
                    }
                return out

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = StructuredMIMModel()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(1, 3, 8, 8)

        adapter = Adapter()
        report = run_ttt(
            adapter,
            [{"image": "a.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="mim",
                steps=2,
                lr=1e-3,
                update_filter="norm_only",
                mim_mask_prob=0.5,
                mim_patch_size=2,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "mim")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertTrue(report.step_metrics)
        self.assertIn("mim_loss", report.step_metrics[0])
        self.assertIn("entropy_loss", report.step_metrics[0])

    def test_run_ttt_cotta(self):
        class CoTTAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
                self.norm = nn.BatchNorm2d(6)

            def forward(self, x):
                y = self.norm(self.conv(x))
                y = y.mean(dim=(2, 3))
                return {"logits": y}

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = CoTTAModel()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(1, 3, 8, 8)

        adapter = Adapter()
        records = [{"image": "a.jpg"}]
        report = run_ttt(
            adapter,
            records,
            config=TTTConfig(
                enabled=True,
                method="cotta",
                steps=2,
                lr=1e-3,
                update_filter="norm_only",
                cotta_ema_momentum=0.99,
                cotta_augmentations=("identity", "hflip"),
                cotta_restore_prob=0.05,
                cotta_restore_interval=1,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "cotta")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)
        self.assertTrue(report.step_metrics)
        self.assertIn("ema_momentum", report.step_metrics[0])
        self.assertIn("restored_count", report.step_metrics[0])
        self.assertIn("aug", report.step_metrics[0])
        self.assertEqual(report.forward_calls, 8)
        self.assertEqual(report.backward_calls, 2)
        self.assertEqual(report.optimizer_steps, 2)

    def test_run_ttt_unsupported_adapter_errors(self):
        adapter = DummyAdapter()
        records = [{"image": "a.jpg"}]
        with self.assertRaises(RuntimeError):
            run_ttt(adapter, records, config=TTTConfig(enabled=True, method="tent", steps=1))

    def test_run_ttt_eata(self):
        class EATAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv2d(3, 8, kernel_size=1, bias=False)
                self.norm = nn.BatchNorm2d(8)
                self.head = nn.Linear(8, 6)

            def forward(self, x):
                feat = self.norm(self.backbone(x)).mean(dim=(2, 3))
                logits = self.head(feat)
                return {"logits": logits.unsqueeze(1).repeat(1, 2, 1)}

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = EATAModel()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(2, 3, 8, 8)

        adapter = Adapter()
        report = run_ttt(
            adapter,
            [{"image": "a.jpg"}, {"image": "b.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="eata",
                steps=2,
                lr=1e-3,
                update_filter="norm_only",
                eata_conf_min=0.0,
                eata_entropy_min=0.0,
                eata_entropy_max=10.0,
                eata_min_valid_dets=0,
                eata_anchor_lambda=1e-3,
                eata_selected_ratio_min=0.0,
                eata_max_skip_streak=2,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "eata")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)
        self.assertTrue(report.step_metrics)
        self.assertIn("selected_count", report.step_metrics[0])
        self.assertIn("adapt_loss", report.step_metrics[0])
        self.assertIn("anchor_loss", report.step_metrics[0])
        self.assertEqual(report.forward_calls, 2)
        self.assertEqual(report.backward_calls, 2)
        self.assertEqual(report.optimizer_steps, 2)

    def test_run_ttt_sar_lora_only(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_block = nn.Conv2d(3, 8, kernel_size=1, bias=False)
                self.head = nn.Conv2d(8, 6, kernel_size=1, bias=False)

            def forward(self, x):
                y = self.head(self.lora_block(x)).mean(dim=(2, 3))
                return {"logits": y.unsqueeze(1).repeat(1, 2, 1)}

        class Adapter(ModelAdapter):
            def __init__(self):
                self._model = Model()

            def predict(self, records):
                return [{"image": r["image"], "detections": []} for r in records]

            def supports_ttt(self) -> bool:
                return True

            def get_model(self):
                return self._model

            def build_loader(self, records, *, batch_size: int = 1):
                yield torch.rand(2, 3, 8, 8)

        adapter = Adapter()
        report = run_ttt(
            adapter,
            [{"image": "a.jpg"}, {"image": "b.jpg"}],
            config=TTTConfig(
                enabled=True,
                method="sar",
                steps=2,
                lr=1e-3,
                update_filter="lora_only",
                sar_rho=0.05,
                sar_adaptive=False,
                sar_first_step_scale=1.0,
            ),
        )
        self.assertTrue(report.enabled)
        self.assertEqual(report.method, "sar")
        self.assertEqual(report.steps_run, 2)
        self.assertEqual(len(report.losses), 2)
        self.assertIsInstance(report.updated_param_count, int)
        self.assertGreater(report.updated_param_count or 0, 0)
        self.assertTrue(report.step_metrics)
        self.assertIn("loss_first", report.step_metrics[0])
        self.assertIn("loss_second", report.step_metrics[0])
        self.assertIn("sar_rho", report.step_metrics[0])
        self.assertEqual(report.forward_calls, 4)
        self.assertEqual(report.backward_calls, 4)
        self.assertEqual(report.optimizer_steps, 2)


if __name__ == "__main__":
    unittest.main()
