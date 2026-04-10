import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestOrchestrateTrainTool(unittest.TestCase):
    def test_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "orchestrate_train.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"orchestrate_train --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--spec", proc.stdout)
        self.assertIn("--execute", proc.stdout)

    def test_plan_only_writes_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "orchestrate_train.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            spec = root / "spec.json"
            out = root / "orchestration.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {
                                "name": "yolox-smoke",
                                "backend": "yolox",
                                "config": "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                                "dataset": "data/smoke",
                                "split": "val",
                                "extra_args": ["--dry-run", "--output", str(root / "train_yolox.json")],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(script), "--spec", str(spec), "--output", str(out)],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"orchestrate_train plan failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "yolozu_training_orchestration_report_v1")
            self.assertEqual(len(payload["results"]), 1)
            self.assertTrue(payload["results"][0]["planned_only"])

    def test_execute_runs_external_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "orchestrate_train.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            spec = root / "spec.json"
            out = root / "orchestration.json"
            train_out = root / "train_yolox.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {
                                "name": "yolox-smoke",
                                "backend": "yolox",
                                "config": "configs/examples/finetune_external/yolox_s_finetune_smoke.py",
                                "dataset": "data/smoke",
                                "split": "val",
                                "extra_args": ["--dry-run", "--output", str(train_out)],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(script), "--spec", str(spec), "--output", str(out), "--execute"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"orchestrate_train execute failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(int(payload["counts"]["executed"]), 1)
            self.assertTrue(train_out.is_file())
            self.assertEqual(str(payload["results"][0]["summary_json"]), str(train_out.resolve()))
            self.assertTrue(bool(payload["results"][0].get("next_steps")))

    def test_execute_runs_detectron2_external_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "orchestrate_train.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            spec = root / "spec.json"
            out = root / "orchestration.json"
            train_out = root / "train_detectron2.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {
                                "name": "detectron2-keypoints-smoke",
                                "backend": "detectron2",
                                "config": "configs/examples/finetune_external/detectron2_finetune_smoke.yaml",
                                "dataset": "data/smoke",
                                "split": "val",
                                "extra_args": [
                                    "--dry-run",
                                    "--task-family",
                                    "keypoints",
                                    "--output",
                                    str(train_out),
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(script), "--spec", str(spec), "--output", str(out), "--execute"],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"orchestrate_train detectron2 execute failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["results"][0]["backend"], "detectron2")
            self.assertTrue(train_out.is_file())
            self.assertTrue(bool(payload["results"][0].get("next_steps")))


if __name__ == "__main__":
    unittest.main()
