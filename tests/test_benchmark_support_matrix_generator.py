import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yolozu.eval import benchmark_mode


def _load_generator():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "generate_benchmark_support_matrix.py"
    spec = importlib.util.spec_from_file_location("generate_benchmark_support_matrix", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkSupportMatrixGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.metadata = self.repo_root / "yolozu" / "data" / "manifest" / "benchmark_support.json"
        self.output = self.repo_root / "docs" / "benchmark_support_matrix.md"

    def test_generated_matrix_is_current(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/generate_benchmark_support_matrix.py",
                "--check",
                "--json",
            ],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"benchmark support matrix drifted:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["rows"], 49)
        self.assertFalse(payload["drifted"])

    def test_metadata_matches_benchmark_runtime_surface(self) -> None:
        meta = json.loads(self.metadata.read_text(encoding="utf-8"))
        formats = [item["id"] for item in meta["formats"]]
        tasks = [item["id"] for item in meta["tasks"]]
        support_pairs = {(item["format"], item["task"]) for item in meta["support"]}
        flag_applicability = meta["flag_applicability"]

        self.assertEqual(formats, list(benchmark_mode.PHASE1_FORMATS))
        self.assertEqual(tasks, list(benchmark_mode.TASK_SEMANTICS))
        self.assertEqual(
            flag_applicability["defaults"],
            benchmark_mode.BACKEND_EXECUTION_FLAG_DEFAULTS,
        )
        self.assertEqual(
            set(flag_applicability["artifact_eval_tasks"]),
            benchmark_mode.ARTIFACT_EVAL_TASKS,
        )
        self.assertEqual(
            set(flag_applicability["artifact_eval_rejected_nondefault_flags"]),
            benchmark_mode.ARTIFACT_EVAL_INERT_BACKEND_FLAGS,
        )
        auto_artifact_row = next(
            item
            for item in flag_applicability["matrix"]
            if item["requested_latency_sources"] == ["auto"]
            and item["effective_latency_source"] == "artifact_eval"
        )
        self.assertEqual(auto_artifact_row["formats"], list(benchmark_mode.REAL_BACKEND_FORMATS))
        self.assertEqual(
            set(auto_artifact_row["rejected_nondefault_flags"]),
            benchmark_mode.ARTIFACT_EVAL_INERT_BACKEND_FLAGS,
        )
        explicit_artifact_row = next(
            item
            for item in flag_applicability["matrix"]
            if item["requested_latency_sources"] == ["artifact_eval"]
        )
        self.assertEqual(explicit_artifact_row["formats"], list(benchmark_mode.PHASE1_FORMATS))
        self.assertEqual(
            set(explicit_artifact_row["rejected_nondefault_flags"]),
            benchmark_mode.ARTIFACT_EVAL_INERT_BACKEND_FLAGS,
        )
        self.assertEqual(len(support_pairs), len(formats) * len(tasks))
        for fmt in benchmark_mode.PHASE1_FORMATS:
            for task in benchmark_mode.TASK_SEMANTICS:
                self.assertIn((fmt, task), support_pairs)

    def test_check_mode_fails_on_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stale = Path(td) / "benchmark_support_matrix.md"
            stale.write_text("# stale\n", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/generate_benchmark_support_matrix.py",
                    "--output",
                    str(stale),
                    "--check",
                    "--json",
                ],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertEqual(proc.returncode, 1)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["drifted"])

    def test_renderer_validates_complete_format_task_grid(self) -> None:
        generator = _load_generator()
        meta = json.loads(self.metadata.read_text(encoding="utf-8"))
        meta["support"] = meta["support"][:-1]
        with self.assertRaises(SystemExit):
            generator.render_markdown(meta, metadata_path=self.metadata)

    def test_renderer_sorts_flag_defaults_independently_of_json_order(self) -> None:
        generator = _load_generator()
        meta = json.loads(self.metadata.read_text(encoding="utf-8"))
        defaults = meta["flag_applicability"]["defaults"]
        meta["flag_applicability"]["defaults"] = {
            name: defaults[name]
            for name in reversed(tuple(defaults))
        }

        rendered = generator.render_markdown(meta, metadata_path=self.metadata)

        self.assertIn(
            "Default values are always accepted: `--batch 1`, `--no-half`, `--no-nms`.",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
