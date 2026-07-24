import re
import unittest
from pathlib import Path


class TestBenchmarkBreadthDocs(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.mode = self._read("docs/benchmark_mode.md")
        self.audit = self._read("docs/benchmark_mode_gap_audit.md")
        self.manual = self._read("manual/chapters/09_parity_bench_protocols.tex")
        self.spec = self._read("docs/benchmark_mode_spec_parity_target.md")

    def _read(self, relative_path: str) -> str:
        return (self.repo_root / relative_path).read_text(encoding="utf-8")

    def test_runtime_and_flag_breadth_is_evidence_gated(self) -> None:
        for name, text in {
            "benchmark mode": self.mode,
            "gap audit": self.audit,
            "manual": self.manual,
            "parity target": self.spec,
        }.items():
            with self.subTest(document=name):
                self.assertRegex(text, re.compile(r"repeated qualified adopter demand", re.IGNORECASE))
                self.assertRegex(
                    text,
                    re.compile(
                        r"maintainable test artifact or reproducible\s+workflow",
                        re.IGNORECASE,
                    ),
                )
                self.assertRegex(text, re.compile(r"runtime.{0,24}license", re.IGNORECASE | re.DOTALL))

    def test_current_grouping_surface_is_documented(self) -> None:
        grouping_flags = (
            "--run-id",
            "--output",
            "--history",
            "--predictions-output",
            "--eval-output",
            "--parity-output",
        )
        for name, text in {
            "benchmark mode": self.mode,
            "gap audit": self.audit,
            "manual": self.manual,
            "parity target": self.spec,
        }.items():
            with self.subTest(document=name):
                for flag in grouping_flags:
                    self.assertIn(flag, text)

    def test_docs_do_not_restore_unconditional_breadth_priorities(self) -> None:
        combined = "\n".join((self.mode, self.audit, self.manual, self.spec))
        forbidden = (
            r"most important remaining gaps",
            r"promote\s+`ncnn`\s+and\s+`rknn`",
            r"add\s+`--name`\s+for user-friendly artifact grouping",
        )
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertNotRegex(combined, re.compile(pattern, re.IGNORECASE))

    def test_named_candidates_remain_deferred(self) -> None:
        for name, text in {
            "benchmark mode": self.mode,
            "gap audit": self.audit,
            "manual": self.manual,
        }.items():
            with self.subTest(document=name):
                for candidate in ("ncnn", "rknn", "--name", "--keras", "INT8"):
                    self.assertIn(candidate, text)
                self.assertRegex(text, re.compile(r"deferred", re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
