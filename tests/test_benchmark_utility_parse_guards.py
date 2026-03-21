from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkUtilityParseGuards(unittest.TestCase):
    def test_adapter_parity_run_parity_preserves_non_json_stdout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("adapter_parity_suite_test", repo_root / "tools" / "adapter_parity_suite.py")
        fake_proc = subprocess.CompletedProcess(args=["dummy"], returncode=1, stdout="not-json", stderr="")

        with mock.patch.object(module.subprocess, "run", return_value=fake_proc):
            ok, payload, err = module._run_parity(
                reference=repo_root / "ref.json",
                candidate=repo_root / "cand.json",
                image_size=None,
                iou_thresh=0.99,
                score_atol=1e-4,
                bbox_atol=1e-4,
            )

        self.assertFalse(ok)
        self.assertIsNone(payload)
        self.assertEqual(err, "not-json")

    def test_backend_parity_run_parity_preserves_non_json_stdout(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("backend_parity_matrix_test", repo_root / "tools" / "backend_parity_matrix.py")
        fake_proc = subprocess.CompletedProcess(args=["dummy"], returncode=1, stdout="not-json", stderr="")

        with mock.patch.object(module.subprocess, "run", return_value=fake_proc):
            rc, payload, err = module._run_parity(
                reference=repo_root / "ref.json",
                candidate=repo_root / "cand.json",
                image_size=None,
                max_images=None,
                iou_thresh=0.99,
                score_atol=1e-4,
                bbox_atol=1e-4,
            )

        self.assertEqual(rc, 1)
        self.assertIsNone(payload)
        self.assertEqual(err, "not-json")

    def test_backend_parity_summarize_tolerates_invalid_bbox_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("backend_parity_matrix_summary_test", repo_root / "tools" / "backend_parity_matrix.py")
        summary = module._summarize_parity(
            {
                "images": 1,
                "results": [
                    {
                        "ok": False,
                        "counts": {"matched": 1, "extra_cand": 0},
                        "failures": [
                            {
                                "type": "value_mismatch",
                                "ref": {"score": 0.9, "bbox": {"cx": "bad", "cy": 0.1, "w": 0.2, "h": 0.3}},
                                "cand": {"score": 0.8, "bbox": {"cx": 0.2, "cy": 0.1, "w": 0.2, "h": 0.3}},
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["failure_images"], 1)
        self.assertAlmostEqual(summary["score_abs_max"], 0.1, places=6)

    def test_benchmark_latency_git_head_tolerates_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("benchmark_latency_test", repo_root / "tools" / "benchmark_latency.py")

        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head())

    def test_audit_backend_support_helpers_handle_invalid_json_and_unreadable_text(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("audit_backend_support_test", repo_root / "tools" / "audit_backend_support.py")

        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            bad_json = root / "bad.json"
            bad_json.write_text("{bad", encoding="utf-8")
            count, err = module._load_predictions_count(bad_json)
            self.assertEqual(count, 0)
            self.assertIsInstance(err, str)
            self.assertIn("invalid_json", err)

            with mock.patch.object(Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"x", 0, 1, "boom")):
                self.assertEqual(module._read_text(root / "dummy.txt"), "")


if __name__ == "__main__":
    unittest.main()
