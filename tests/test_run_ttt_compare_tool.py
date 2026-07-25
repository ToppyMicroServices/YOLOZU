import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.run_ttt_compare as run_ttt_compare

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TestRunTTTCompareTool(unittest.TestCase):
    @staticmethod
    def _preflight_stub(**kwargs):
        return {
            "config": str(kwargs["config_path"]),
            "model_class": "tests.CompatibleDetector",
            "checkpoint_compatibility": {
                "status": "compatible",
                "load": {"loaded": True},
            },
            "structured_mim_supported": True,
        }

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
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "dummy.ckpt"
            checkpoint.write_bytes(b"non-empty-checkpoint")
            for method in ("tent", "mim", "mim_probe", "cotta", "eata", "sar"):
                run_dir = root / method
                with mock.patch.object(
                    run_ttt_compare,
                    "_model_checkpoint_preflight",
                    side_effect=self._preflight_stub,
                ):
                    result = run_ttt_compare.main(
                        [
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
                        ]
                    )
                self.assertEqual(result, 0)
                plan_path = run_dir / "plan.json"
                self.assertTrue(plan_path.is_file(), f"missing plan for {method}")
                payload = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(payload.get("boilerplate_name"), method)
                expected_method = "mim" if method == "mim_probe" else method
                self.assertEqual(payload.get("method"), expected_method)
                commands = payload.get("commands") or {}
                self.assertIn("baseline_export", commands)
                self.assertIn("adapted_export", commands)
                self.assertIn("--max-images", commands["baseline_export"])
                self.assertIn("--max-images", commands["adapted_export"])
                self.assertEqual(
                    (payload.get("execution_status") or {}).get("state"),
                    "not_executed",
                )
                prerequisites = payload.get("prerequisites") or {}
                self.assertGreater(int(prerequisites.get("dataset_images") or 0), 0)
                self.assertEqual(
                    len(str(prerequisites.get("checkpoint_sha256") or "")), 64
                )
                if method in {"tent", "cotta", "eata", "sar"}:
                    self.assertEqual(
                        prerequisites["configs"][0]["path"],
                        "rtdetr_pose/configs/base.json",
                    )

    def test_mim_and_sar_boilerplates_expand_real_update_args(self):
        repo_root = Path(__file__).resolve().parents[1]
        for method in ("mim", "mim_probe", "sar"):
            payload = json.loads(
                (
                    repo_root
                    / "configs"
                    / "examples"
                    / "ttt_compare"
                    / f"{method}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNone(payload.get("preset"))
            extra = payload.get("extra_export_args")
            self.assertIsInstance(extra, list)
            self.assertIn("--ttt-update-filter", extra)
            idx = extra.index("--ttt-update-filter")
            self.assertLess(idx + 1, len(extra))
            self.assertEqual(extra[idx + 1], "norm_only")
            self.assertIn("--ttt-steps", extra)
            self.assertIn("--ttt-lr", extra)
        expected_configs = {
            "mim": "configs/examples/ttt_compare/rtdetr_pose_mim_compare.json",
            "mim_probe": "configs/examples/ttt_compare/yolo26n_mim_real_probe.json",
        }
        for method, expected in expected_configs.items():
            mim_payload = json.loads(
                (
                    repo_root
                    / "configs"
                    / "examples"
                    / "ttt_compare"
                    / f"{method}.json"
                ).read_text(encoding="utf-8")
            )
            common = mim_payload.get("common_export_args")
            self.assertIsInstance(common, list)
            self.assertEqual(common, ["--config", expected])

    def test_dry_run_mim_plan_includes_repo_backed_config_in_both_exports(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "dummy.ckpt"
            checkpoint.write_bytes(b"non-empty-checkpoint")
            run_dir = root / "mim"
            with mock.patch.object(
                run_ttt_compare,
                "_model_checkpoint_preflight",
                side_effect=self._preflight_stub,
            ):
                result = run_ttt_compare.main(
                    [
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
                    ]
                )
            self.assertEqual(result, 0)
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            baseline = payload["commands"]["baseline_export"]
            adapted = payload["commands"]["adapted_export"]
            expected = "configs/examples/ttt_compare/rtdetr_pose_mim_compare.json"
            self.assertIn("--config", baseline)
            self.assertEqual(baseline[baseline.index("--config") + 1], expected)
            self.assertIn("--config", adapted)
            self.assertEqual(adapted[adapted.index("--config") + 1], expected)

    def test_dry_run_mim_probe_plan_includes_real_probe_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "dummy.ckpt"
            checkpoint.write_bytes(b"non-empty-checkpoint")
            run_dir = root / "mim_probe"
            with mock.patch.object(
                run_ttt_compare,
                "_model_checkpoint_preflight",
                side_effect=self._preflight_stub,
            ):
                result = run_ttt_compare.main(
                    [
                        "--boilerplate",
                        "mim_probe",
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
                    ]
                )
            self.assertEqual(result, 0)
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            baseline = payload["commands"]["baseline_export"]
            adapted = payload["commands"]["adapted_export"]
            expected = "configs/examples/ttt_compare/yolo26n_mim_real_probe.json"
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

    def test_concise_aliases_run_full_prerequisite_validation(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            run_dir = root / "concise"
            with mock.patch.object(
                run_ttt_compare,
                "_model_checkpoint_preflight",
                side_effect=self._preflight_stub,
            ):
                result = run_ttt_compare.main(
                    [
                        "--method",
                        "tent",
                        "--data",
                        str(dataset),
                        "--weights",
                        str(checkpoint),
                        "--out",
                        str(run_dir),
                        "-n",
                        "1",
                        "--dry-run",
                        "--force",
                    ]
                )
            self.assertEqual(result, 0)
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["execution_status"]["state"], "not_executed")
            self.assertEqual(payload["prerequisites"]["dataset_images"], 1)

    def test_dry_run_rejects_empty_checkpoint_and_records_failed_stage(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_ttt_compare.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "empty.pt"
            checkpoint.write_bytes(b"")
            run_dir = root / "invalid"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "-b",
                    "tent",
                    "-d",
                    str(dataset),
                    "-c",
                    str(checkpoint),
                    "-r",
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
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            status = payload["execution_status"]
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], "prerequisite_validation")
            self.assertIn("checkpoint is empty", status["error"])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_dry_run_rejects_incompatible_checkpoint_before_success_plan(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "run_ttt_compare.py"
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            config = root / "tiny.json"
            config.write_text(
                json.dumps(
                    {
                        "dataset": {"root": ".", "split": "val", "format": "yolo"},
                        "model": {
                            "num_classes": 3,
                            "hidden_dim": 64,
                            "num_queries": 10,
                            "use_uncertainty": False,
                            "backbone_name": "tiny_cnn",
                            "stem_channels": 8,
                            "backbone_channels": [16, 32, 64],
                            "stage_blocks": [1, 1, 1],
                            "num_encoder_layers": 0,
                            "num_decoder_layers": 1,
                            "nhead": 8,
                        },
                        "loss": {
                            "name": "default",
                            "task_aligner": "none",
                            "weights": {},
                        },
                        "train": {"batch_size": 1, "lr": 0.0001, "epochs": 1},
                    }
                ),
                encoding="utf-8",
            )
            boilerplate = root / "custom_tent.json"
            boilerplate.write_text(
                json.dumps(
                    {
                        "method": "tent",
                        "preset": "safe",
                        "reset": "sample",
                        "common_export_args": ["--config", str(config)],
                        "extra_export_args": [],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "incompatible.pt"
            torch.save({"wrong.weight": torch.zeros(1)}, checkpoint)
            run_dir = root / "incompatible"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "-b",
                    str(boilerplate),
                    "-d",
                    str(dataset),
                    "-c",
                    str(checkpoint),
                    "-r",
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
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            status = payload["execution_status"]
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], "prerequisite_validation")
            self.assertIn("checkpoint is incompatible", status["error"])
            self.assertEqual(
                (status.get("error_report") or {}).get("status"),
                "incompatible",
            )

    def test_force_removes_stale_outputs_and_records_export_failure(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            run_dir = root / "stale"
            run_dir.mkdir()
            stale_paths = [
                run_dir / "baseline_predictions.json",
                run_dir / "tent_predictions.json",
                run_dir / "tent_ttt_log.json",
                run_dir / "baseline_eval.json",
                run_dir / "tent_eval.json",
                run_dir / "tent_before_after_compare.json",
                run_dir / "tent_before_after_compare.md",
            ]
            for path in stale_paths:
                path.write_text("stale", encoding="utf-8")

            failed = {
                "command": ["fake-export"],
                "returncode": 9,
                "stdout": "",
                "stderr": "intentional failure",
            }
            with (
                mock.patch.object(run_ttt_compare, "_run", return_value=failed),
                mock.patch.object(
                    run_ttt_compare,
                    "_model_checkpoint_preflight",
                    side_effect=self._preflight_stub,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "baseline export failed"):
                    run_ttt_compare.main(
                        [
                            "-b",
                            "tent",
                            "-d",
                            str(dataset),
                            "-c",
                            str(checkpoint),
                            "-r",
                            str(run_dir),
                            "--force",
                        ]
                    )

            for path in stale_paths:
                self.assertFalse(path.exists(), f"stale artifact survived: {path}")
            payload = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            status = payload["execution_status"]
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], "baseline_export")
            self.assertIn("intentional failure", status["error"])

    def test_every_execution_boundary_records_failed_stage(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset = self._make_dataset(root)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            def invoke(
                case,
                *,
                run_failure_at=None,
                proxy_failure=False,
                artifact_failure=False,
                comparison_failure=False,
                report_failure=False,
            ):
                run_dir = root / case
                call_count = 0

                def fake_run(command, *, cwd):
                    nonlocal call_count
                    call_count += 1
                    if call_count == run_failure_at:
                        return {
                            "command": list(command),
                            "returncode": 7,
                            "stdout": "",
                            "stderr": f"{case} injected failure",
                        }
                    if proxy_failure and call_count == 3:
                        return {
                            "command": list(command),
                            "returncode": 1,
                            "stdout": "",
                            "stderr": "RuntimeError: pycocotools is required",
                        }
                    if "--output" in command:
                        output = Path(command[command.index("--output") + 1])
                        if "export" in command:
                            output.write_text(
                                json.dumps(
                                    {
                                        "predictions": [
                                            {
                                                "image": str(
                                                    dataset
                                                    / "images"
                                                    / "val"
                                                    / "000001.jpg"
                                                ),
                                                "detections": [],
                                            }
                                        ]
                                    }
                                ),
                                encoding="utf-8",
                            )
                        else:
                            output.write_text(
                                json.dumps(
                                    {
                                        "results": [
                                            {
                                                "metrics": {
                                                    "map50": 0.0,
                                                    "map50_95": 0.0,
                                                    "map75": 0.0,
                                                    "ar100": 0.0,
                                                }
                                            }
                                        ]
                                    }
                                ),
                                encoding="utf-8",
                            )
                    return {
                        "command": list(command),
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                    }

                real_atomic_text = run_ttt_compare._atomic_write_text

                def maybe_fail_report(path, text):
                    if (
                        report_failure
                        and Path(path).name == "tent_before_after_compare.json"
                    ):
                        raise OSError(f"{case} injected report failure")
                    return real_atomic_text(path, text)

                parity = {
                    "ok": True,
                    "images": 1,
                    "reference": "baseline",
                    "candidate": "adapted",
                    "results": [],
                }
                patches = [
                    mock.patch.object(run_ttt_compare, "_run", side_effect=fake_run),
                    mock.patch.object(
                        run_ttt_compare,
                        "_model_checkpoint_preflight",
                        side_effect=self._preflight_stub,
                    ),
                    mock.patch.object(
                        run_ttt_compare,
                        "compare_predictions",
                        side_effect=(
                            RuntimeError(f"{case} injected comparison failure")
                            if comparison_failure
                            else None
                        ),
                        return_value=parity,
                    ),
                    mock.patch.object(
                        run_ttt_compare,
                        "_atomic_write_text",
                        side_effect=maybe_fail_report,
                    ),
                ]
                if proxy_failure:
                    patches.append(
                        mock.patch.object(
                            run_ttt_compare,
                            "_build_simple_map_proxy_eval",
                            side_effect=RuntimeError(f"{case} injected proxy failure"),
                        )
                    )
                if artifact_failure:

                    def remove_adapted_after_export(result, *, method):
                        (run_dir / "tent_predictions.json").unlink()

                    patches.append(
                        mock.patch.object(
                            run_ttt_compare,
                            "_raise_ttt_compare_failure",
                            side_effect=remove_adapted_after_export,
                        )
                    )

                for patcher in patches:
                    patcher.start()
                try:
                    with self.assertRaises(BaseException):
                        run_ttt_compare.main(
                            [
                                "-b",
                                "tent",
                                "-d",
                                str(dataset),
                                "-c",
                                str(checkpoint),
                                "-r",
                                str(run_dir),
                                "--force",
                            ]
                        )
                finally:
                    for patcher in reversed(patches):
                        patcher.stop()

                status = json.loads(
                    (run_dir / "plan.json").read_text(encoding="utf-8")
                )["execution_status"]
                self.assertEqual(status["state"], "failed")
                self.assertTrue(status.get("error"))
                return status["stage"]

            self.assertEqual(
                invoke("baseline_export_failure", run_failure_at=1),
                "baseline_export",
            )
            self.assertEqual(
                invoke("adapted_export_failure", run_failure_at=2),
                "adapted_export",
            )
            self.assertEqual(
                invoke("baseline_eval_failure", run_failure_at=3),
                "baseline_eval_result",
            )
            self.assertEqual(
                invoke("baseline_proxy_failure", proxy_failure=True),
                "baseline_eval_proxy",
            )
            self.assertEqual(
                invoke("artifact_failure", artifact_failure=True),
                "artifact_validation",
            )
            self.assertEqual(
                invoke("comparison_failure", comparison_failure=True),
                "prediction_comparison",
            )
            self.assertEqual(
                invoke("report_failure", report_failure=True),
                "report_write",
            )

    def test_completed_state_is_written_atomically(self):
        plan = {"execution_status": {"state": "running", "started_at": "start"}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plan.json"
            run_ttt_compare._set_plan_status(
                plan,
                path,
                state="completed",
                stage="complete",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["execution_status"]["state"], "completed")
        self.assertEqual(payload["execution_status"]["stage"], "complete")

    def test_simple_map_proxy_detector_matches_pycocotools_failure(self):
        result = {
            "returncode": 1,
            "stderr": "RuntimeError: pycocotools is required for COCO mAP evaluation.",
            "stdout": "",
        }
        self.assertTrue(run_ttt_compare._should_use_simple_map_proxy(result))

    def test_ttt_summary_counts_rollback_steps(self):
        payload = {
            "meta": {
                "ttt": {
                    "enabled": True,
                    "method": "tent",
                    "report": {
                        "method": "tent",
                        "seconds": 0.1,
                        "steps_run": 2,
                        "stopped_early": True,
                        "step_metrics": [
                            {"rolled_back": False},
                            {"rolled_back": True},
                        ],
                    },
                }
            }
        }
        summary = run_ttt_compare._extract_ttt_summary(payload)
        self.assertEqual(summary.get("rollback_steps"), 1)
        self.assertEqual(summary.get("stopped_early_count"), 1)


if __name__ == "__main__":
    unittest.main()
