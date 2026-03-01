import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestRunRealMultitaskFinetuneDemoTool(unittest.TestCase):
    def _load_module(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_real_multitask_finetune_demo.py"
        spec = importlib.util.spec_from_file_location("run_real_multitask_finetune_demo_tool", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_real_multitask_finetune_demo.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        self.assertIn("--prepare", proc.stdout)
        self.assertIn("--download-if-missing", proc.stdout)
        self.assertIn("--strict-provenance", proc.stdout)

    def test_prepare_flag_forwards_download_options(self):
        mod = self._load_module()
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = root / "data" / "real_multitask_fewshot"
            out_dir = root / "reports"
            prepare_commands: list[list[str]] = []

            def fake_run(cmd, *, cwd=None):
                prepare_commands.append(list(cmd))
                dataset_root.mkdir(parents=True, exist_ok=True)
                summary_path = dataset_root / "prepare_summary.json"
                summary = {"label_provenance": {"model_inference_used": False}}
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            def fake_train_task(**kwargs):
                run_dir = Path(str(kwargs["out_dir"])) / str(kwargs["task_name"])
                checkpoint = run_dir / "checkpoint_bundle.pt"
                return {
                    "task": str(kwargs["task_name"]),
                    "command": [],
                    "returncode": 0,
                    "val_map50_95": 0.0,
                    "run_dir": str(run_dir),
                    "checkpoint_bundle": str(checkpoint),
                    "ok": True,
                    "artifacts": {
                        "expected": {},
                        "present": {},
                        "missing": [],
                        "complete": True,
                    },
                }

            with patch.object(mod, "_run", side_effect=fake_run), patch.object(mod, "_train_task", side_effect=fake_train_task):
                rc = mod.main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--out",
                        str(out_dir),
                        "--prepare",
                        "--download-if-missing",
                        "--allow-auto-download",
                        "--accept-dataset-license",
                        "--download-num-images",
                        "8",
                        "--max-steps",
                        "1",
                        "--epochs",
                        "1",
                        "--force",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(prepare_commands, msg="expected prepare command to run")
            cmd = prepare_commands[0]
            self.assertIn("--download-if-missing", cmd)
            self.assertIn("--allow-auto-download", cmd)
            self.assertIn("--accept-dataset-license", cmd)
            self.assertIn("--download-num-images", cmd)

            report_path = out_dir / "multitask_finetune_demo_report.json"
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("ok"))
            evidence = payload.get("evidence") or {}
            self.assertEqual(int(evidence.get("tasks_total", -1)), 5)
            self.assertEqual(int(evidence.get("tasks_ok", -1)), 5)


if __name__ == "__main__":
    unittest.main()
