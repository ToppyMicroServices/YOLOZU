import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import onnx  # type: ignore
except Exception:  # pragma: no cover
    onnx = None  # type: ignore


@unittest.skipIf(onnx is None, "onnx not installed")
class TestGenDummyDetsOnnxTool(unittest.TestCase):
    def test_default_ir_version_is_11(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "ci" / "gen_dummy_dets_onnx.py"
        self.assertTrue(script.is_file(), "missing tools/ci/gen_dummy_dets_onnx.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "dummy.onnx"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out",
                    str(out),
                    "--shape",
                    "1x3x64x64",
                    "--opset",
                    "17",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"gen_dummy_dets_onnx.py failed:\n{proc.stdout}\n{proc.stderr}")

            model = onnx.load(str(out))
            self.assertEqual(int(model.ir_version), 11)

    def test_ir_version_flag_overrides_model_ir(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "ci" / "gen_dummy_dets_onnx.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            out = Path(td) / "dummy_ir10.onnx"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out",
                    str(out),
                    "--shape",
                    "1x3x64x64",
                    "--opset",
                    "17",
                    "--ir-version",
                    "10",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"gen_dummy_dets_onnx.py failed:\n{proc.stdout}\n{proc.stderr}")

            model = onnx.load(str(out))
            self.assertEqual(int(model.ir_version), 10)


if __name__ == "__main__":
    unittest.main()
