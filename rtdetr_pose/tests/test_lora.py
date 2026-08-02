import sys
import argparse
from pathlib import Path
import tempfile
import unittest


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestLoRA(unittest.TestCase):
    @staticmethod
    def _snapshot_named_parameters(model):
        return {name: param.detach().clone() for name, param in model.named_parameters()}

    @staticmethod
    def _assert_only_lora_parameters_changed(testcase, before, model):
        changed_lora = []
        for name, param in model.named_parameters():
            changed = not torch.equal(before[name], param.detach())
            if "lora_" in name:
                changed_lora.append(changed)
            else:
                testcase.assertFalse(changed, f"base parameter changed: {name}")
        testcase.assertTrue(any(changed_lora), "no LoRA parameter changed")

    def test_apply_lora_head_replaces_linears(self):
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.lora import LoRALinear, apply_lora

        model = RTDETRPose(num_classes=3, hidden_dim=16, num_queries=2, num_decoder_layers=1, nhead=2)
        replaced = apply_lora(model, r=4, target="head")
        self.assertGreaterEqual(replaced, 1)
        self.assertIsInstance(model.head.cls, LoRALinear)
        self.assertIsInstance(model.head.box, LoRALinear)
        self.assertIsInstance(model.head.log_z, LoRALinear)
        self.assertIsInstance(model.head.rot6d, LoRALinear)

        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertIn("logits", out)
        self.assertIn("bbox", out)

    def test_freeze_base_trains_only_lora(self):
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.lora import LoRALinear, apply_lora, count_trainable_params, mark_only_lora_as_trainable

        model = RTDETRPose(num_classes=3, hidden_dim=16, num_queries=2, num_decoder_layers=1, nhead=2)
        apply_lora(model, r=2, target="head")
        info = mark_only_lora_as_trainable(model)
        self.assertGreater(info["lora_params"], 0)
        self.assertEqual(info["bias_params"], 0)

        # Ensure trainable params match LoRA params only.
        self.assertEqual(count_trainable_params(model), info["lora_params"])

        # Ensure a representative base param is frozen.
        some_base = model.backbone.stem[0].conv.weight
        self.assertFalse(bool(some_base.requires_grad))

        # Ensure LoRA params are trainable.
        self.assertIsInstance(model.head.cls, LoRALinear)
        self.assertTrue(bool(model.head.cls.lora_A.requires_grad))
        self.assertTrue(bool(model.head.cls.lora_B.requires_grad))

    def test_apply_lora_all_conv1x1_replaces_some_convs(self):
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.lora import LoRAConv2d, apply_lora

        model = RTDETRPose(num_classes=3, hidden_dim=16, num_queries=2, num_decoder_layers=1, nhead=2)
        replaced = apply_lora(model, r=2, target="all_conv1x1")
        self.assertGreaterEqual(replaced, 1)
        self.assertTrue(any(isinstance(m, LoRAConv2d) for m in model.modules()))

        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertIn("logits", out)

    def test_train_save_reload_prediction_parity_and_additional_update(self):
        from rtdetr_pose.lora import apply_lora, mark_only_lora_as_trainable
        from rtdetr_pose.model import RTDETRPose
        from rtdetr_pose.train_utils import load_checkpoint_into, save_checkpoint_bundle

        torch.manual_seed(2026)
        model = RTDETRPose(
            num_classes=3,
            hidden_dim=16,
            num_queries=2,
            num_decoder_layers=1,
            nhead=2,
        ).eval()
        replaced = apply_lora(model, r=2, alpha=4.0, target="head")
        self.assertGreater(replaced, 0)
        mark_only_lora_as_trainable(model)

        image = torch.randn(1, 3, 64, 64)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=1e-2,
        )
        before_train = self._snapshot_named_parameters(model)
        outputs = model(image)
        loss = outputs["logits"].square().mean() + outputs["bbox"].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self._assert_only_lora_parameters_changed(self, before_train, model)

        with torch.no_grad():
            expected = {
                "logits": model(image)["logits"].detach().clone(),
                "bbox": model(image)["bbox"].detach().clone(),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "lora_checkpoint.pt"
            args = argparse.Namespace(
                lora_r=2,
                lora_alpha=4.0,
                lora_dropout=0.0,
                lora_target="head",
                lora_freeze_base=True,
                lora_train_bias="none",
            )
            save_checkpoint_bundle(
                checkpoint,
                model=model,
                optim=optimizer,
                args=args,
                epoch=0,
                global_step=1,
                last_epoch_steps=1,
                last_epoch_avg=float(loss.detach()),
                last_loss_dict={"loss": loss.detach()},
                run_record={},
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["args"]["lora_r"], 2)
            self.assertEqual(payload["args"]["lora_target"], "head")

            torch.manual_seed(999)
            reloaded = RTDETRPose(
                num_classes=3,
                hidden_dim=16,
                num_queries=2,
                num_decoder_layers=1,
                nhead=2,
            ).eval()
            self.assertEqual(
                apply_lora(reloaded, r=2, alpha=4.0, target="head"),
                replaced,
            )
            mark_only_lora_as_trainable(reloaded)
            load_checkpoint_into(reloaded, None, checkpoint, restore_rng=False)

            with torch.no_grad():
                actual = reloaded(image)
            self.assertTrue(torch.equal(expected["logits"], actual["logits"]))
            self.assertTrue(torch.equal(expected["bbox"], actual["bbox"]))

            before_additional = self._snapshot_named_parameters(reloaded)
            optimizer2 = torch.optim.AdamW(
                [param for param in reloaded.parameters() if param.requires_grad],
                lr=1e-2,
            )
            outputs2 = reloaded(image)
            loss2 = outputs2["logits"].square().mean() + outputs2["bbox"].square().mean()
            optimizer2.zero_grad(set_to_none=True)
            loss2.backward()
            optimizer2.step()
            self._assert_only_lora_parameters_changed(self, before_additional, reloaded)

            with torch.no_grad():
                updated = reloaded(image)
            self.assertFalse(torch.equal(actual["logits"], updated["logits"]))
