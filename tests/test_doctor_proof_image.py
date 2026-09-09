import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yolozu.cli_entry import GUIDE_ROUTES
from yolozu.core.doctor import run_doctor_proof


class TestDoctorProofImage(unittest.TestCase):
    def test_proof_png_decodes_and_has_valid_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, status = run_doctor_proof(output_dir=tmp)
            self.assertEqual(status, 0, report)
            path = Path(tmp) / "toy_dataset/images/val2017/proof_0001.png"
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                self.assertEqual(image.size, (1, 1))
                self.assertEqual(image.mode, "RGB")

    def test_export_guide_writes_promised_overlay(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo_root)}
        with tempfile.TemporaryDirectory() as tmp:
            for command in GUIDE_ROUTES["export"]["commands"]:
                parts = shlex.split(command)
                self.assertEqual(parts[0], "yolozu")
                result = subprocess.run(
                    [sys.executable, "-m", "yolozu", *parts[1:]],
                    cwd=tmp, env=env, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, f"{command}\n{result.stdout}\n{result.stderr}")
            for output in GUIDE_ROUTES["export"]["outputs"]:
                self.assertTrue((Path(tmp) / output).is_file(), output)
            overlay = Path(tmp) / "reports/predict_overlays/000000_proof_0001.png"
            with Image.open(overlay) as image:
                image.load()
                self.assertEqual(image.size, (1, 1))
            html = (Path(tmp) / "reports/predict_images.html").read_text()
            self.assertIn("predict_overlays/000000_proof_0001.png", html)


if __name__ == "__main__":
    unittest.main()
