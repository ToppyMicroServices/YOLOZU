from __future__ import annotations

import importlib.util
import subprocess
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


class TestSmallUtilityParseGuards(unittest.TestCase):
    def test_calibrate_scores_parse_grid_skips_invalid_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("calibrate_scores_test", repo_root / "tools" / "calibrate_scores.py")
        self.assertEqual(module._parse_grid("0.5,bad,1.0,,2"), [0.5, 1.0, 2.0])

    def test_check_mcp_settings_read_text_handles_unicode_errors(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("check_mcp_settings_test", repo_root / "tools" / "check_mcp_settings.py")
        with mock.patch.object(Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"x", 0, 1, "boom")):
            self.assertIsNone(module._read_text(repo_root / "docs" / "generated" / "missing.md"))

    def test_benchmark_sar_safe_float_falls_back_on_invalid_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("benchmark_sar_robustness_test", repo_root / "tools" / "benchmark_sar_robustness.py")
        self.assertEqual(module._safe_float("bad", default=1.25), 1.25)

    def test_measure_trt_git_head_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("measure_trt_latency_parse_test", repo_root / "tools" / "measure_trt_latency.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head())


if __name__ == "__main__":
    unittest.main()
