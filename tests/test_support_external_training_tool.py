import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestSupportExternalTrainingTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        self.assertTrue(script.is_file(), "missing tools/support_external_training.py")

        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"support_external_training --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("train-yolox", proc.stdout)
        self.assertIn("train-detectron2", proc.stdout)
        self.assertIn("train-mmdetection", proc.stdout)
        self.assertIn("train-mmpose", proc.stdout)
        self.assertIn("train-mmseg", proc.stdout)
        self.assertIn("train-tao", proc.stdout)
        self.assertIn("train-ultralytics", proc.stdout)
        self.assertIn("train-hf-detr", proc.stdout)

    def test_tao_dry_run_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "support_external_training.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            out = root / "train_tao.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "train-tao",
                    "--config",
                    "configs/examples/finetune_external/tao_finetune_smoke.yaml",
                    "--dataset",
                    "data/smoke",
                    "--split",
                    "val",
                    "--task-family",
                    "bbox",
                    "--dry-run",
                    "--output",
                    str(out),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"support_external_training train-tao failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["backend"]["backend_id"], "tao")
            self.assertIn("resume", payload["handoff_contracts"])
            self.assertIn("reports/resume_handoff.json", payload["run_output_contract"]["stable_artifacts"])


if __name__ == "__main__":
    unittest.main()
