import subprocess
import re
import sys
import unittest
from pathlib import Path


class TestCLISurfaceReadiness(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.wrapper = self.repo_root / "tools" / "yolozu.py"

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )

    def test_canonical_and_compat_entrypoints_share_critical_top_level_commands(self):
        critical_commands = [
            "doctor",
            "export",
            "predict-images",
            "registry",
            "completion",
            "guide",
        ]
        entrypoints = [
            [sys.executable, "-m", "yolozu", "--help"],
            [sys.executable, "-m", "yolozu.cli", "--help"],
            [sys.executable, str(self.wrapper), "--help"],
        ]

        for cmd in entrypoints:
            proc = self._run(cmd)
            if proc.returncode != 0:
                self.fail(f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
            for token in critical_commands:
                self.assertIn(token, proc.stdout, f"missing {token} in {' '.join(cmd)}")

    def test_predict_images_critical_backend_surface_matches_wrapper(self):
        required_backends = {
            "dummy", "torch", "onnxrt", "trt", "executorch", "yolox",
            "opencv-dnn", "opencv-dnn-rtdetr", "opencv-dnn-yolo",
        }
        expected_tokens = [
            "--config",
            "--checkpoint",
            "--ttt",
            "--dump-io",
        ]
        entrypoints = [
            [sys.executable, "-m", "yolozu", "predict-images", "--help"],
            [sys.executable, "-m", "yolozu.cli", "predict-images", "--help"],
            [sys.executable, str(self.wrapper), "predict-images", "--help"],
        ]

        for cmd in entrypoints:
            proc = self._run(cmd)
            if proc.returncode != 0:
                self.fail(f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
            # The legacy wrapper also exposes the labels fixture backend.
            # Compare required choices without depending on help line wrapping
            # or rejecting additional, entrypoint-specific choices.
            choices = re.search(r"--backend\s+\{([^}]+)\}", proc.stdout)
            self.assertIsNotNone(choices, f"missing backend choices in {' '.join(cmd)}")
            actual = set(re.sub(r"\s+", "", choices.group(1)).split(","))
            self.assertTrue(required_backends <= actual, f"missing backends: {required_backends - actual}")
            for token in expected_tokens:
                self.assertIn(token, proc.stdout, f"missing {token} in {' '.join(cmd)}")


if __name__ == "__main__":
    unittest.main()
