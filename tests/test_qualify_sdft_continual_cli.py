import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestQualifySDFTContinualCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "tools" / "qualify_sdft_continual.py"

    def test_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("three-seed", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)

    def test_refuses_existing_output_before_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.repo_root) as output_dir:
            proc = subprocess.run(
                [sys.executable, str(self.script), "--output-dir", output_dir],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing to replace existing output path", proc.stderr)

    def test_rejects_spec_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = Path(temp_dir) / "spec.json"
            spec.write_text(json.dumps({}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(self.script), "--spec", str(spec)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("spec must stay inside the repository", proc.stderr)

    def test_tracked_spec_has_three_seeds_and_real_coco(self) -> None:
        spec = json.loads(
            (self.repo_root / "configs/continual/sdft_coco128_blur_qualification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreaterEqual(len(spec["seeds"]), 3)
        self.assertEqual(spec["methods"], ["naive", "sdft"])
        self.assertEqual(spec["evaluation"]["backend"], "coco")
        self.assertTrue(spec["claim_boundary"]["independent_reproduction_required"])


if __name__ == "__main__":
    unittest.main()
