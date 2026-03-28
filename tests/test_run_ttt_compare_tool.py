import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunTTTCompareTool(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        images = dataset / "images" / "val"
        labels = dataset / "labels" / "val"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / "000001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return dataset

    def test_help_lists_boilerplate_and_skip_eval(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_ttt_compare.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"run_ttt_compare --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--boilerplate", proc.stdout)
        self.assertIn("--skip-eval", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)

    def test_dry_run_writes_plan_for_all_builtin_boilerplates(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_ttt_compare.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "dummy.ckpt"
            checkpoint.write_bytes(b"")
            for method in ("tent", "mim", "cotta", "eata", "sar"):
                run_dir = root / method
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--boilerplate",
                        method,
                        "--dataset",
                        str(dataset),
                        "--split",
                        "val",
                        "--checkpoint",
                        str(checkpoint),
                        "--run-dir",
                        str(run_dir),
                        "--max-images",
                        "1",
                        "--dry-run",
                        "--force",
                    ],
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                )
                if proc.returncode != 0:
                    self.fail(f"dry-run failed for {method}:\n{proc.stdout}\n{proc.stderr}")
                plan_path = run_dir / "plan.json"
                self.assertTrue(plan_path.is_file(), f"missing plan for {method}")
                payload = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(payload.get("boilerplate_name"), method)
                self.assertEqual(payload.get("method"), method)
                commands = payload.get("commands") or {}
                self.assertIn("baseline_export", commands)
                self.assertIn("adapted_export", commands)
                self.assertIn("--max-images", commands["baseline_export"])
                self.assertIn("--max-images", commands["adapted_export"])

    def test_mim_and_sar_boilerplates_expand_real_update_args(self):
        repo_root = Path(__file__).resolve().parents[1]
        for method in ("mim", "sar"):
            payload = json.loads((repo_root / "configs" / "examples" / "ttt_compare" / f"{method}.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload.get("preset"))
            extra = payload.get("extra_export_args")
            self.assertIsInstance(extra, list)
            self.assertIn("--ttt-update-filter", extra)
            idx = extra.index("--ttt-update-filter")
            self.assertLess(idx + 1, len(extra))
            self.assertEqual(extra[idx + 1], "norm_only")
            self.assertIn("--ttt-steps", extra)
            self.assertIn("--ttt-lr", extra)
        mim_payload = json.loads((repo_root / "configs" / "examples" / "ttt_compare" / "mim.json").read_text(encoding="utf-8"))
        common = mim_payload.get("common_export_args")
        self.assertIsInstance(common, list)
        self.assertEqual(
            common,
            ["--config", "configs/examples/ttt_compare/rtdetr_pose_mim_compare.json"],
        )

    def test_dry_run_mim_plan_includes_repo_backed_config_in_both_exports(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_ttt_compare.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "dummy.ckpt"
            checkpoint.write_bytes(b"")
            run_dir = root / "mim"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--boilerplate",
                    "mim",
                    "--dataset",
                    str(dataset),
                    "--split",
                    "val",
                    "--checkpoint",
                    str(checkpoint),
                    "--run-dir",
                    str(run_dir),
                    "--dry-run",
                    "--force",
                ],
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            if proc.returncode != 0:
                self.fail(f"mim dry-run failed:\n{proc.stdout}\n{proc.stderr}")
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            baseline = payload["commands"]["baseline_export"]
            adapted = payload["commands"]["adapted_export"]
            expected = "configs/examples/ttt_compare/rtdetr_pose_mim_compare.json"
            self.assertIn("--config", baseline)
            self.assertEqual(baseline[baseline.index("--config") + 1], expected)
            self.assertIn("--config", adapted)
            self.assertEqual(adapted[adapted.index("--config") + 1], expected)

    def test_shell_wrapper_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "ttt_compare.sh"
        proc = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            self.fail(f"ttt_compare.sh --help failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("--boilerplate", proc.stdout)
        self.assertIn("tent", proc.stdout)
        self.assertIn("dry-run", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
