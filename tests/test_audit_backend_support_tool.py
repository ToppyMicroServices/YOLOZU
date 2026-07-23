import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestAuditBackendSupportTool(unittest.TestCase):
    @staticmethod
    def _write_tiny_dataset(root: Path) -> Path:
        dataset_root = root / "dataset"
        images = dataset_root / "images" / "val2017"
        labels = dataset_root / "labels" / "val2017"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / "000001.jpg").write_bytes(b"")
        (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return dataset_root

    def test_audit_backend_support_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._write_tiny_dataset(root)

            out = root / "backend_support_audit.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out),
                ],
                text=True,
                capture_output=True,
                check=False,
                cwd=str(repo_root),
            )
            if proc.returncode != 0:
                self.fail(f"audit_backend_support.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("ok")))
            results = payload.get("results") or []
            self.assertEqual(len(results), 4)
            self.assertTrue(all(bool(item.get("ok")) for item in results))
            self.assertTrue(all(bool(item.get("dry_run", True)) for item in results))
            self.assertTrue(all(item.get("execution_evidence_error") is None for item in results))
            self.assertEqual(payload.get("verified_non_dry_backends"), [])
            coverage = payload.get("multitask_coverage") or {}
            self.assertTrue(bool(coverage))
            self.assertGreaterEqual(int(coverage.get("supported_task_count", 0)), 5)
            self.assertEqual(int(coverage.get("error_count", 0)), 0)

    def test_audit_backend_support_require_non_dry_fails_without_selection(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._write_tiny_dataset(root)

            out = root / "backend_support_audit.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out),
                    "--require-non-dry",
                ],
                text=True,
                capture_output=True,
                check=False,
                cwd=str(repo_root),
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(bool(payload.get("ok")))
            warnings = payload.get("warnings") or []
            self.assertTrue(any("require-non-dry" in str(w) for w in warnings))
            coverage = payload.get("multitask_coverage") or {}
            self.assertIn("gaps", coverage)

    def test_audit_rejects_non_dry_yolox_without_model_inputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._write_tiny_dataset(root)
            out = root / "backend_support_audit.json"
            work_dir = root / "audit-work"
            work_dir.mkdir()
            stale_yolox = work_dir / "predictions_yolox.json"
            stale_yolox.write_text(
                json.dumps(
                    {
                        "predictions": [],
                        "meta": {
                            "extra": {
                                "dry_run": False,
                                "execution_status": "completed",
                                "runtime_executed": True,
                                "inference_calls": 1,
                                "runtime_error": None,
                                "model_provenance": {"weights_sha256": "stale"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset-root",
                    str(dataset_root),
                    "--split",
                    "val2017",
                    "--max-images",
                    "1",
                    "--output",
                    str(out),
                    "--work-dir",
                    str(work_dir),
                    "--non-dry-backend",
                    "yolox",
                    "--require-non-dry",
                ],
                text=True,
                capture_output=True,
                check=False,
                cwd=str(repo_root),
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(bool(payload.get("ok")))
            self.assertEqual(payload.get("verified_non_dry_backends"), [])
            yolox = next(item for item in payload["results"] if item["backend"] == "yolox")
            self.assertNotEqual(yolox.get("returncode"), 0)
            self.assertFalse(bool(yolox.get("ok")))
            self.assertEqual(yolox.get("output_error"), "output_missing")
            self.assertFalse(stale_yolox.exists())

    def test_execution_evidence_allows_empty_detections_after_runtime_call(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"
        spec = importlib.util.spec_from_file_location("audit_backend_support_tool", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        error = module._validate_execution_evidence(
            meta={
                "extra": {
                    "dry_run": False,
                    "execution_status": "completed",
                    "runtime_executed": True,
                    "inference_calls": 1,
                    "runtime_error": None,
                    "model_provenance": {"weights_sha256": "abc123"},
                }
            },
            dry_run=False,
        )
        self.assertIsNone(error)

    def test_execution_evidence_rejects_invalid_inference_count(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"
        spec = importlib.util.spec_from_file_location("audit_backend_support_tool_invalid", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        error = module._validate_execution_evidence(
            meta={
                "extra": {
                    "dry_run": False,
                    "execution_status": "completed",
                    "runtime_executed": True,
                    "inference_calls": "not-a-number",
                    "runtime_error": None,
                    "model_provenance": {"weights_sha256": "abc123"},
                }
            },
            dry_run=False,
        )
        self.assertEqual(error, "inference_calls_invalid")

    def test_require_non_dry_accepts_verified_runtime_evidence(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_backend_support.py"
        spec = importlib.util.spec_from_file_location("audit_backend_support_tool_verified", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            dataset_root = self._write_tiny_dataset(root)
            report_path = root / "backend_support_audit.json"

            def fake_run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                del cwd
                output_path = Path(cmd[cmd.index("--output") + 1])
                dry_run = "--dry-run" in cmd
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(
                        {
                            "predictions": [],
                            "meta": {
                                "extra": {
                                    "dry_run": dry_run,
                                    "execution_status": "dry_run" if dry_run else "completed",
                                    "runtime_executed": not dry_run,
                                    "inference_calls": 0 if dry_run else 1,
                                    "runtime_error": None if not dry_run else "dry_run",
                                    "model_provenance": {"weights_sha256": "abc123"},
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout=str(output_path), stderr="")

            coverage = {
                "ok": True,
                "gaps": [],
                "error_count": 0,
                "warning_count": 0,
                "supported_task_count": 5,
            }
            with (
                mock.patch.object(module, "_run", side_effect=fake_run),
                mock.patch.object(module, "_build_multitask_coverage", return_value=coverage),
            ):
                rc = module.main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--output",
                        str(report_path),
                        "--non-dry-backend",
                        "yolox",
                        "--require-non-dry",
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["verified_non_dry_backends"], ["yolox"])


if __name__ == "__main__":
    unittest.main()
