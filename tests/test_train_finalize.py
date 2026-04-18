import json
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

torch = None
finalize_training = None
if importlib.util.find_spec("torch") is not None:
    import torch  # type: ignore[no-redef]
    from rtdetr_pose.train_finalize import finalize_training  # type: ignore[no-redef]


class TestTrainFinalize(unittest.TestCase):
    @unittest.skipIf(torch is None or finalize_training is None, "torch is not installed")
    def test_onnx_export_uses_cpu_even_when_runtime_device_is_non_cpu(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            onnx_out = root / "model.onnx"
            meta_out = root / "model.onnx.meta.json"
            summary_out = root / "training_summary.json"
            seen: dict[str, object] = {}

            args = SimpleNamespace(
                metrics_json=None,
                metrics_csv=None,
                checkpoint_out=None,
                checkpoint_bundle_out=None,
                best_checkpoint_out=None,
                ewc_state_out=None,
                si_state_out=None,
                onnx_out=str(onnx_out),
                onnx_meta_out=str(meta_out),
                training_summary_out=str(summary_out),
                onnx_opset=17,
                onnx_dynamic_hw=False,
                image_size=64,
                parity_json_out=None,
                parity_policy=None,
                epochs=1,
                max_steps=1,
            )
            model = torch.nn.Conv2d(3, 3, kernel_size=1)

            def _fake_export(mod, dummy, path, **kwargs):
                seen["model_device"] = str(next(mod.parameters()).device)
                seen["dummy_device"] = str(dummy.device)
                Path(path).write_bytes(b"onnx")

            fake_export = types.SimpleNamespace(export_onnx=_fake_export)

            with unittest.mock.patch.dict("sys.modules", {"rtdetr_pose.export": fake_export}):
                finalize_training(
                    args=args,
                    is_main=True,
                    ddp_enabled=False,
                    model=model,
                    optim=None,
                    sched=None,
                    scaler=None,
                    ema=None,
                    device=torch.device("meta"),
                    run_contract=None,
                    run_dir=root,
                    run_record={"run_id": "test"},
                    global_step=1,
                    last_loss_dict=None,
                    last_epoch_avg=None,
                    last_epoch_steps=1,
                    last_grad_norm=None,
                    last_data_time_s=None,
                    last_step_time_s=None,
                    last_throughput=None,
                    last_max_vram_mb=None,
                )

            self.assertEqual(seen.get("model_device"), "cpu")
            self.assertEqual(seen.get("dummy_device"), "cpu")
            payload = json.loads(meta_out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("export_device"), "cpu")
            self.assertEqual(str(next(model.parameters()).device), "cpu")
            summary = json.loads(summary_out.read_text(encoding="utf-8"))
            self.assertEqual(summary.get("format"), "yolozu_training_run_summary_v1")
            self.assertEqual(((summary.get("backend") or {}).get("backend_id")), "reference-rtdetr-pose")
            self.assertEqual((((summary.get("steps") or {}).get("export") or {}).get("artifact")), str(onnx_out))

    @unittest.skipIf(torch is None or finalize_training is None, "torch is not installed")
    def test_onnx_export_failure_writes_structured_meta(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            onnx_out = root / "model.onnx"
            meta_out = root / "model.onnx.meta.json"
            summary_out = root / "training_summary.json"

            args = SimpleNamespace(
                metrics_json=None,
                metrics_csv=None,
                checkpoint_out=None,
                checkpoint_bundle_out=None,
                best_checkpoint_out=None,
                ewc_state_out=None,
                si_state_out=None,
                onnx_out=str(onnx_out),
                onnx_meta_out=str(meta_out),
                training_summary_out=str(summary_out),
                onnx_opset=17,
                onnx_dynamic_hw=False,
                image_size=64,
                parity_json_out=None,
                parity_policy=None,
                epochs=1,
                max_steps=1,
            )
            model = torch.nn.Conv2d(3, 3, kernel_size=1)
            fake_export = types.SimpleNamespace(
                export_onnx=lambda *a, **k: (_ for _ in ()).throw(IndexError("bad axis"))
            )

            with unittest.mock.patch.dict("sys.modules", {"rtdetr_pose.export": fake_export}):
                finalize_training(
                    args=args,
                    is_main=True,
                    ddp_enabled=False,
                    model=model,
                    optim=None,
                    sched=None,
                    scaler=None,
                    ema=None,
                    device=torch.device("cpu"),
                    run_contract=None,
                    run_dir=root,
                    run_record={"run_id": "test"},
                    global_step=1,
                    last_loss_dict=None,
                    last_epoch_avg=None,
                    last_epoch_steps=1,
                    last_grad_norm=None,
                    last_data_time_s=None,
                    last_step_time_s=None,
                    last_throughput=None,
                    last_max_vram_mb=None,
                )

            self.assertTrue(meta_out.is_file())
            payload = json.loads(meta_out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("export_device"), "cpu")
            self.assertEqual(((payload.get("error") or {}).get("type")), "IndexError")
            self.assertIn("bad axis", str((payload.get("error") or {}).get("message")))
            self.assertIsNone(payload.get("onnx"))
            summary = json.loads(summary_out.read_text(encoding="utf-8"))
            self.assertEqual((((summary.get("steps") or {}).get("export") or {}).get("status")), "failed")


if __name__ == "__main__":
    unittest.main()
