import subprocess
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

from yolozu.core import doctor as doctor_mod
from rtdetr_pose import train_cli
from rtdetr_pose import train_minimal


class TestMacOSMpsSmoke(unittest.TestCase):
    def test_doctor_and_train_surface_mps_beta_support(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.test",
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
            ),
            version=types.SimpleNamespace(cuda=None),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(is_available=lambda: False, version=lambda: None),
                mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: True),
            ),
            float16="float16",
            bfloat16="bfloat16",
            amp=types.SimpleNamespace(autocast=lambda **kwargs: ("autocast", kwargs)),
        )
        fake_ort = types.SimpleNamespace(
            __version__="1.test",
            get_available_providers=lambda: ["CPUExecutionProvider", "CoreMLExecutionProvider"],
        )

        with mock.patch.dict("sys.modules", {"torch": fake_torch, "onnxruntime": fake_ort}, clear=False):
            runtime = doctor_mod._gather_runtime_capabilities(
                tools={"trtexec": False, "nvidia_smi": False},
                gpu={"nvidia_smi_list": []},
            )

        self.assertTrue(runtime["torch"]["mps_built"])
        self.assertTrue(runtime["torch"]["mps_available"])
        self.assertTrue(runtime["onnxruntime"]["coreml_provider"])

        parser = train_cli.build_parser()
        help_text = parser.format_help()
        self.assertIn("auto|cpu|cuda|cuda:0|mps", help_text)
        self.assertIn("best-effort beta", help_text)
        self.assertIn("falls back to fp32", help_text)

        resolved, warnings = train_minimal._resolve_device_string("auto", torch_module=fake_torch)
        self.assertEqual(resolved, "mps")
        self.assertEqual(warnings, [])

    def test_train_minimal_help_works_without_torch(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "rtdetr_pose" / "tools" / "train_minimal.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--device", proc.stdout)
        self.assertIn("--amp", proc.stdout)


if __name__ == "__main__":
    unittest.main()
