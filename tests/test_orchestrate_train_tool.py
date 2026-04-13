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
            self.assertEqual(int(payload["counts"]["by_backend"]["yolox"]), 1)

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

    def test_execute_appends_registry_entry(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "orchestrate_train.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            spec = root / "spec.json"
            out = root / "orchestration.json"
            registry = root / "training_registry.jsonl"
            train_out = root / "train_mmpose.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {
                                "name": "mmpose-smoke",
                                "backend": "mmpose",
                                "config": "configs/examples/finetune_external/mmpose_finetune_smoke.py",
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
                [
                    sys.executable,
                    str(script),
                    "--spec",
                    str(spec),
                    "--output",
                    str(out),
                    "--execute",
                    "--registry-out",
                    str(registry),
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"orchestrate_train registry execute failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(str(payload["registry_out"]), str(registry.resolve()))
            self.assertEqual(int(payload["registry_summary"]["entries"]), 1)
            lines = [line for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["format"], "yolozu_training_registry_entry_v1")
            self.assertEqual(entry["backend_id"], "mmpose")

    def test_defaults_and_resume_are_forwarded(self) -> None:
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
                        "defaults": {
                            "dataset": "data/smoke",
                            "split": "val",
                            "resume_from": "models/resume.ckpt",
                        },
                        "experiments": [
                            {
                                "name": "tao-smoke",
                                "backend": "tao",
                                "config": "configs/examples/finetune_external/tao_finetune_smoke.yaml",
                                "extra_args": ["--dry-run", "--output", str(root / "train_tao.json")],
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
                self.fail(f"orchestrate_train defaults failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads(out.read_text(encoding="utf-8"))
            row = payload["results"][0]
            self.assertEqual(row["backend"], "tao")
            self.assertEqual(row["resume_from"], "models/resume.ckpt")
            self.assertIn("--resume-from", row["command"])


if __name__ == "__main__":
    unittest.main()
