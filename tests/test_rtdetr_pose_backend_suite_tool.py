import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import onnxruntime as _onnxruntime  # type: ignore  # noqa: F401
    _HAS_ONNXRT = True
except Exception:  # pragma: no cover
    _HAS_ONNXRT = False

def _write_tiny_rtdetr_pose_config(path: Path) -> None:
    cfg = {
        "dataset": {"root": ".", "split": "val2017", "format": "yolo"},
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
    path.write_text(json.dumps(cfg))


@unittest.skipIf(torch is None, "torch not installed")
class TestRTDETRPoseBackendSuiteTool(unittest.TestCase):
    def test_torch_only_runs_and_writes_report(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "rtdetr_pose_backend_suite.py"
        self.assertTrue(script.is_file(), "missing tools/rtdetr_pose_backend_suite.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg_path = root / "tiny_rtdetr_pose.json"
            _write_tiny_rtdetr_pose_config(cfg_path)

            out_path = root / "suite.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(cfg_path),
                    "--backends",
                    "torch",
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--batch",
                    "1",
                    "--samples",
                    "1",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"rtdetr_pose_backend_suite.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            self.assertIn("parity", payload)
            self.assertIn("benchmark", payload)
            results = payload.get("benchmark", {}).get("results") or []
            self.assertTrue(any(r.get("name") == "torch" for r in results))

    @unittest.skipIf(not _HAS_ONNXRT, "onnx/onnxruntime not installed")
    def test_torch_and_onnxrt_parity_passes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "rtdetr_pose_backend_suite.py"
        self.assertTrue(script.is_file(), "missing tools/rtdetr_pose_backend_suite.py")

        sys.path.insert(0, str(repo_root / "rtdetr_pose"))
        from rtdetr_pose.config import load_config
        from rtdetr_pose.export import export_onnx
        from rtdetr_pose.factory import build_model

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            cfg_path = root / "tiny_rtdetr_pose.json"
            _write_tiny_rtdetr_pose_config(cfg_path)

            torch.manual_seed(0)
            model = build_model(load_config(str(cfg_path)).model).eval()
            ckpt_path = root / "tiny_checkpoint.pt"
            torch.save(model.state_dict(), ckpt_path)

            onnx_path = root / "tiny.onnx"
            dummy = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
            export_onnx(model, dummy, str(onnx_path), opset_version=17, input_name="images", dynamic_hw=False)

            out_path = root / "suite.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(ckpt_path),
                    "--onnx",
                    str(onnx_path),
                    "--backends",
                    "torch,onnxrt",
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--batch",
                    "1",
                    "--samples",
                    "1",
                    "--warmup",
                    "1",
                    "--iterations",
                    "2",
                    "--score-atol",
                    "1e-3",
                    "--bbox-atol",
                    "1e-3",
                    "--output",
                    str(out_path),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"rtdetr_pose_backend_suite.py failed:\n{proc.stdout}\n{proc.stderr}")

            payload = json.loads(out_path.read_text())
            candidates = payload.get("parity", {}).get("candidates") or {}
            onnxrt = candidates.get("onnxrt") or {}
            self.assertTrue(onnxrt.get("available"), f"expected onnxrt available, got: {onnxrt}")
            self.assertTrue(onnxrt.get("passed"), f"expected parity passed, got: {onnxrt}")


if __name__ == "__main__":
    unittest.main()
