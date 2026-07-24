import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _write_tiny_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": {"root": ".", "split": "val", "format": "yolo"},
                "model": {
                    "num_classes": 3,
                    "hidden_dim": 64,
                    "num_queries": 10,
                    "use_uncertainty": False,
                    "backbone_name": "tiny_cnn",
                    "stem_channels": 8,
                    "backbone_channels": [16, 32, 64],
                    "stage_blocks": [1, 1, 1],
                    "num_encoder_layers": 0,
                    "num_decoder_layers": 1,
                    "nhead": 8,
                },
                "loss": {"name": "default", "task_aligner": "none", "weights": {}},
                "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
            }
        ),
        encoding="utf-8",
    )


@unittest.skipIf(torch is None, "torch not installed")
class TestCheckpointEntrypointPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _run(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.repo_root / script), *args],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_public_adapter_cli_partial_opt_in_records_report(self):
        sys.path.insert(0, str(self.repo_root / "rtdetr_pose"))
        from rtdetr_pose.config import load_config
        from rtdetr_pose.factory import build_model

        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            model = build_model(load_config(str(config)).model)
            first_key, first_value = next(iter(model.state_dict().items()))
            checkpoint = root / "partial.pt"
            torch.save({first_key: first_value}, checkpoint)
            output = root / "predictions.json"

            proc = self._run(
                "tools/export_predictions.py",
                "--adapter",
                "rtdetr_pose",
                "--dataset",
                "data/smoke",
                "--split",
                "val",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--allow-partial-checkpoint",
                "--max-images",
                "1",
                "--image-size",
                "32",
                "--wrap",
                "--output",
                str(output),
            )

            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            report = payload["meta"]["inference"]["checkpoint_compatibility"]
            self.assertEqual(report["status"], "partial")
            self.assertTrue(report["allow_partial"])
            self.assertEqual(report["load"]["loaded_key_count"], 1)

    def test_public_adapter_cli_incompatible_load_removes_stale_predictions(self):
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            checkpoint = root / "incompatible.pt"
            torch.save({"wrong.weight": torch.zeros(1)}, checkpoint)
            output = root / "predictions.json"
            tta_log = root / "tta.json"
            ttt_log = root / "ttt.json"
            output.write_text('{"status":"stale"}', encoding="utf-8")
            tta_log.write_text('{"status":"stale"}', encoding="utf-8")
            ttt_log.write_text('{"status":"stale"}', encoding="utf-8")

            proc = self._run(
                "tools/export_predictions.py",
                "--adapter",
                "rtdetr_pose",
                "--dataset",
                "data/smoke",
                "--split",
                "val",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--max-images",
                "1",
                "--image-size",
                "32",
                "--wrap",
                "--tta",
                "--tta-log-out",
                str(tta_log),
                "--ttt",
                "--ttt-log-out",
                str(ttt_log),
                "--output",
                str(output),
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("checkpoint is incompatible", proc.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(tta_log.exists())
            self.assertFalse(ttt_log.exists())

    def test_tensorrt_export_incompatible_load_removes_stale_artifacts(self):
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            checkpoint = root / "incompatible.pt"
            torch.save({"wrong.weight": torch.zeros(1)}, checkpoint)
            onnx = root / "model.onnx"
            onnx_meta = root / "model.onnx.meta.json"
            engine = root / "model.plan"
            engine_meta = root / "model.plan.meta.json"
            for path in (onnx, onnx_meta, engine, engine_meta):
                path.write_bytes(b"stale")

            proc = self._run(
                "tools/export_trt.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--onnx",
                str(onnx),
                "--onnx-meta",
                str(onnx_meta),
                "--engine",
                str(engine),
                "--engine-meta",
                str(engine_meta),
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("checkpoint is incompatible", proc.stderr)
            for path in (onnx, onnx_meta, engine, engine_meta):
                self.assertFalse(path.exists(), str(path))

    def test_tensorrt_export_partial_opt_in_records_report(self):
        sys.path.insert(0, str(self.repo_root / "rtdetr_pose"))
        from rtdetr_pose.config import load_config
        from rtdetr_pose.factory import build_model

        script = self.repo_root / "tools" / "export_trt.py"
        spec = importlib.util.spec_from_file_location(
            "_test_export_trt_checkpoint",
            script,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            model = build_model(load_config(str(config)).model)
            first_key, first_value = next(iter(model.state_dict().items()))
            checkpoint = root / "partial.pt"
            torch.save({first_key: first_value}, checkpoint)
            onnx = root / "model.onnx"
            onnx_meta = root / "model.onnx.meta.json"

            def _fake_export(_model, _dummy, path, **_kwargs):
                Path(path).write_bytes(b"fake onnx")

            with patch(
                "rtdetr_pose.export.export_onnx",
                side_effect=_fake_export,
            ):
                result = module.main(
                    [
                        "--config",
                        str(config),
                        "--checkpoint",
                        str(checkpoint),
                        "--allow-partial-checkpoint",
                        "--onnx",
                        str(onnx),
                        "--onnx-meta",
                        str(onnx_meta),
                        "--skip-engine",
                    ]
                )

            self.assertEqual(result, 0)
            payload = json.loads(onnx_meta.read_text(encoding="utf-8"))
            report = payload["report"]["checkpoint_report"]
            self.assertEqual(report["status"], "partial")
            self.assertTrue(report["allow_partial"])
            self.assertEqual(report["load"]["loaded_key_count"], 1)

    def test_backend_suite_incompatible_load_removes_stale_report(self):
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            checkpoint = root / "incompatible.pt"
            torch.save({"wrong.weight": torch.zeros(1)}, checkpoint)
            output = root / "suite.json"
            output.write_text('{"status":"stale"}', encoding="utf-8")

            proc = self._run(
                "tools/rtdetr_pose_backend_suite.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--backends",
                "torch",
                "--device",
                "cpu",
                "--image-size",
                "32",
                "--samples",
                "1",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--output",
                str(output),
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("checkpoint is incompatible", proc.stderr)
            self.assertFalse(output.exists())

    def test_backend_suite_partial_opt_in_records_report(self):
        sys.path.insert(0, str(self.repo_root / "rtdetr_pose"))
        from rtdetr_pose.config import load_config
        from rtdetr_pose.factory import build_model

        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            _write_tiny_config(config)
            model = build_model(load_config(str(config)).model)
            first_key, first_value = next(iter(model.state_dict().items()))
            checkpoint = root / "partial.pt"
            torch.save({first_key: first_value}, checkpoint)
            output = root / "suite.json"

            proc = self._run(
                "tools/rtdetr_pose_backend_suite.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--allow-partial-checkpoint",
                "--backends",
                "torch",
                "--device",
                "cpu",
                "--image-size",
                "32",
                "--samples",
                "1",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--output",
                str(output),
            )

            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            report = payload["meta"]["checkpoint_compatibility"]
            self.assertEqual(report["status"], "partial")
            self.assertTrue(report["allow_partial"])
            self.assertEqual(report["load"]["loaded_key_count"], 1)

    def test_all_three_cli_families_expose_partial_opt_in(self):
        for script in (
            "tools/export_predictions.py",
            "tools/export_trt.py",
            "tools/rtdetr_pose_backend_suite.py",
        ):
            with self.subTest(script=script):
                proc = self._run(script, "--help")
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn("--allow-partial-checkpoint", proc.stdout)


class TestCheckpointEntrypointValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _run(
        self,
        script: str,
        *args: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.repo_root / script), *args],
            cwd=str(cwd or self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_tensorrt_export_rejects_checkpoint_with_skip_onnx(self):
        proc = self._run(
            "tools/export_trt.py",
            "--checkpoint",
            "unused.pt",
            "--skip-onnx",
            "--dry-run",
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--checkpoint cannot be used with --skip-onnx", proc.stderr)

    def test_backend_suite_rejects_unchecked_checkpoint_without_torch(self):
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            config = root / "tiny.json"
            checkpoint = root / "checkpoint.pt"
            output = root / "suite.json"
            _write_tiny_config(config)
            checkpoint.write_bytes(b"not loaded")
            output.write_text('{"status":"stale"}', encoding="utf-8")

            proc = self._run(
                "tools/rtdetr_pose_backend_suite.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--backends",
                "onnxrt",
                "--dry-run",
                "--output",
                str(output),
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "--checkpoint requires torch in --backends",
                proc.stderr,
            )
            self.assertFalse(output.exists())

    @unittest.skipIf(torch is None, "torch not installed")
    def test_relative_requested_logs_are_removed_from_repo_root(self):
        with tempfile.TemporaryDirectory(dir=str(self.repo_root)) as td:
            root = Path(td)
            relative_root = root.relative_to(self.repo_root)
            config = root / "tiny.json"
            checkpoint = root / "incompatible.pt"
            output = root / "predictions.json"
            tta_log = root / "tta.json"
            ttt_log = root / "ttt.json"
            _write_tiny_config(config)
            torch.save({"wrong.weight": torch.zeros(1)}, checkpoint)
            tta_log.write_text('{"status":"stale"}', encoding="utf-8")
            ttt_log.write_text('{"status":"stale"}', encoding="utf-8")

            proc = self._run(
                "tools/export_predictions.py",
                "--adapter",
                "rtdetr_pose",
                "--dataset",
                str(self.repo_root / "data" / "smoke"),
                "--split",
                "val",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--max-images",
                "1",
                "--image-size",
                "32",
                "--wrap",
                "--tta",
                "--tta-log-out",
                str(relative_root / "tta.json"),
                "--ttt",
                "--ttt-log-out",
                str(relative_root / "ttt.json"),
                "--output",
                str(output),
                cwd=root,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("checkpoint is incompatible", proc.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(tta_log.exists())
            self.assertFalse(ttt_log.exists())


if __name__ == "__main__":
    unittest.main()
