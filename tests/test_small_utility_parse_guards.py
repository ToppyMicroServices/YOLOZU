from __future__ import annotations

import importlib.util
import sys
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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

    def test_benchmark_eata_safe_float_falls_back_on_invalid_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("benchmark_eata_stability_test", repo_root / "tools" / "benchmark_eata_stability.py")
        self.assertEqual(module._safe_float("bad", default=2.5), 2.5)

    def test_measure_trt_git_head_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("measure_trt_latency_parse_test", repo_root / "tools" / "measure_trt_latency.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head())

    def test_update_map_targets_git_head_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("update_map_targets_parse_test", repo_root / "tools" / "update_map_targets_from_suite.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head(repo_root))

    def test_import_yolo26_baseline_git_head_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("import_yolo26_baseline_parse_test", repo_root / "tools" / "import_yolo26_baseline.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head())

    def test_tune_gate_weights_git_head_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("tune_gate_weights_parse_test", repo_root / "tools" / "tune_gate_weights.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertIsNone(module._git_head())

    def test_tune_gate_weights_as_float_list_skips_invalid_items(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("tune_gate_weights_float_list_test", repo_root / "tools" / "tune_gate_weights.py")
        self.assertEqual(module._as_float_list(["1.0", "bad", None, 2]), [1.0, 2.0])

    def test_report_dependency_licenses_git_info_handles_git_failures(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("report_dependency_licenses_parse_test", repo_root / "tools" / "report_dependency_licenses.py")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        ), mock.patch.object(
            module.subprocess,
            "call",
            side_effect=FileNotFoundError("git"),
        ):
            info = module._git_info()
        self.assertEqual(info, {"sha": None, "dirty": None})

    def test_summarize_gpu_ngc_read_json_returns_none_on_invalid_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("summarize_gpu_ngc_parse_test", repo_root / "tools" / "ci" / "summarize_gpu_ngc_run.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            self.assertIsNone(module._read_json(path))

    def test_import_ultralytics_extract_class_names_skips_invalid_dict_keys(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("import_ultralytics_data_yaml_parse_test", repo_root / "tools" / "import_ultralytics_data_yaml.py")
        cfg = {"names": {"0": "person", "bad": "skip", None: "skip", "2": "car"}}
        self.assertEqual(module._extract_class_names(cfg), ["person", "class_1", "car"])

    def test_validate_run_meta_main_reports_invalid_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("validate_run_meta_parse_test", repo_root / "tools" / "validate_run_meta.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_meta.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                module.main([str(path)])
        self.assertIn("failed to parse json", str(ctx.exception))

    def test_audit_manifest_run_help_handles_os_errors(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("audit_manifest_inputs_vs_help_parse_test", repo_root / "tools" / "audit_manifest_inputs_vs_help.py")
        with mock.patch.object(module.subprocess, "run", side_effect=FileNotFoundError("python3")):
            _, error = module._run_help("tools/yolozu.py", timeout_s=1.0)
        self.assertIn("failed to run --help", str(error))

    def test_gen_smoke_dataset_main_reports_invalid_hw(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("gen_smoke_dataset_parse_test", repo_root / "tools" / "ci" / "gen_smoke_dataset.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit) as ctx:
                module.main(["--out", tmpdir, "--hw", "bad"])
        self.assertIn("invalid --hw", str(ctx.exception))

    def test_make_subset_link_or_copy_falls_back_on_symlink_os_error(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("make_subset_dataset_parse_test", repo_root / "tools" / "make_subset_dataset.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "source.txt"
            dst = root / "nested" / "dest.txt"
            src.write_text("payload", encoding="utf-8")
            with mock.patch.object(module.os, "symlink", side_effect=OSError("boom")):
                module._link_or_copy(src, dst, copy=False)
            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_text(encoding="utf-8"), "payload")

    def test_image_size_pil_size_wraps_os_errors(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("image_size_parse_test", repo_root / "yolozu" / "core" / "image_size.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.avif"
            path.write_bytes(b"not-an-image")
            with self.assertRaises(module.ImageSizeError):
                module._pil_size(path)

    def test_min_adapter_collect_class_names_skips_invalid_ids(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("min_adapter_parse_test", repo_root / "yolozu" / "integrations" / "min_adapter.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = root / "labels" / "train"
            labels.mkdir(parents=True, exist_ok=True)
            (labels / "000001.txt").write_text("bad 0.1 0.2 0.3 0.4\n1 0.1 0.2 0.3 0.4\n", encoding="utf-8")
            self.assertEqual(module._collect_class_names(root, "train"), ["class_0", "class_1"])

    def test_lbfgs_as_float_list_returns_none_on_invalid_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("lbfgs_scale_k_parse_test", repo_root / "yolozu" / "calibration" / "lbfgs_scale_k.py")
        self.assertIsNone(module._as_float_list([1.0, "bad", 2.0]))

    def test_lbfgs_extract_det_bbox_returns_none_for_invalid_bbox(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("lbfgs_scale_k_bbox_test", repo_root / "yolozu" / "calibration" / "lbfgs_scale_k.py")
        self.assertIsNone(module._extract_det_bbox({"bbox": {"cx": "bad", "cy": 0.5, "w": 0.2, "h": 0.2}}))

    def test_lbfgs_get_image_hw_returns_none_for_invalid_dict_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        module = _load_module("lbfgs_scale_k_hw_test", repo_root / "yolozu" / "calibration" / "lbfgs_scale_k.py")
        self.assertIsNone(module._get_image_hw({"image_size": {"height": "bad", "width": 100}}))


if __name__ == "__main__":
    unittest.main()
