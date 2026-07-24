import argparse
import re
import subprocess
import sys
import unittest
from pathlib import Path

from yolozu.inference.adapter import ModelAdapter, RTDETRPoseAdapter
from yolozu.tta import integration
from yolozu.tta.cli_options import TTT_METHOD_CHOICES, add_ttt_arguments
from yolozu.tta.config import SUPPORTED_TTT_METHODS, TTTConfig


class TestTTTDocsImplementationAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.plan = (self.repo_root / "docs" / "ttt_integration_plan.md").read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _section(text: str, heading: str) -> str:
        marker = f"## {heading}"
        start = text.find(marker)
        if start < 0:
            raise AssertionError(f"missing section: {marker}")
        end = text.find("\n## ", start + len(marker))
        return text[start : end if end >= 0 else len(text)]

    @staticmethod
    def _table_rows(section: str) -> list[list[str]]:
        rows = []
        for line in section.splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not cells or cells[0] in {"CLI value", "Hook", "Method"}:
                continue
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rows.append(cells)
        return rows

    def _ttt_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        add_ttt_arguments(parser, include_enable_flag=True)
        return parser

    def test_documented_methods_match_shared_implementation_and_cli_choices(self) -> None:
        method_section = self._section(self.plan, "Implemented Research-lane methods")
        plan_methods = tuple(
            row[0].strip("`").lower()
            for row in self._table_rows(method_section)
            if row[1] == "Implemented"
        )

        matrix = (self.repo_root / "docs" / "tta_support_matrix.md").read_text(
            encoding="utf-8"
        )
        algorithm_section = self._section(matrix, "Algorithms")
        matrix_methods = tuple(
            row[0].lower()
            for row in self._table_rows(algorithm_section)
            if row[1] == "Supported"
        )

        parser = self._ttt_parser()
        method_action = parser._option_string_actions["--ttt-method"]
        self.assertEqual(plan_methods, SUPPORTED_TTT_METHODS)
        self.assertEqual(matrix_methods, SUPPORTED_TTT_METHODS)
        self.assertEqual(tuple(method_action.choices or ()), SUPPORTED_TTT_METHODS)
        self.assertEqual(TTT_METHOD_CHOICES, SUPPORTED_TTT_METHODS)
        self.assertEqual(integration.SUPPORTED_TTT_METHODS, SUPPORTED_TTT_METHODS)
        self.assertEqual(method_action.default, TTTConfig().method)

        for method in SUPPORTED_TTT_METHODS:
            with self.subTest(method=method):
                parsed = parser.parse_args(["--ttt-method", method])
                self.assertEqual(parsed.ttt_method, method)

    def test_export_help_renders_the_implemented_method_choices_and_opt_in_default(self) -> None:
        proc = subprocess.run(
            [sys.executable, "tools/export_predictions.py", "--help"],
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        help_text = proc.stdout + proc.stderr
        match = re.search(r"--ttt-method\s+\{([^}]+)\}", help_text)
        self.assertIsNotNone(match, help_text)
        rendered_methods = tuple(part.strip() for part in match.group(1).split(","))
        self.assertEqual(rendered_methods, SUPPORTED_TTT_METHODS)

        defaults = self._ttt_parser().parse_args([])
        self.assertFalse(defaults.ttt)
        self.assertEqual(defaults.ttt_method, TTTConfig().method)

    def test_documented_ttt_flags_are_exactly_the_export_cli_surface(self) -> None:
        cli_section = self._section(self.plan, "CLI and configuration surface")
        documented_flags = set(re.findall(r"`(--ttt[a-z0-9-]*)`", cli_section))

        parser = self._ttt_parser()
        implemented_flags = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option.startswith("--ttt")
        }
        self.assertEqual(documented_flags, implemented_flags)

    def test_documented_adapter_hooks_match_the_built_in_adapter_interface(self) -> None:
        hook_section = self._section(self.plan, "Implemented adapter interface")
        documented_hooks = {
            row[0].strip("`").split("(", 1)[0]
            for row in self._table_rows(hook_section)
        }
        expected_hooks = {"supports_ttt", "get_model", "build_loader"}
        self.assertEqual(documented_hooks, expected_hooks)

        base = ModelAdapter()
        self.assertFalse(base.supports_ttt())
        self.assertIsNone(base.get_model())
        with self.assertRaisesRegex(RuntimeError, "does not support TTT"):
            base.build_loader([])

        for hook in documented_hooks:
            with self.subTest(hook=hook):
                self.assertTrue(callable(getattr(ModelAdapter, hook)))
                self.assertIn(hook, RTDETRPoseAdapter.__dict__)
        adapter = object.__new__(RTDETRPoseAdapter)
        self.assertTrue(adapter.supports_ttt())

    def test_mim_is_masked_image_modeling_with_masked_reconstruction(self) -> None:
        for rel in (
            "docs/tta_support_matrix.md",
            "yolozu/data/docs/tta_support_matrix.md",
        ):
            with self.subTest(path=rel):
                text = (self.repo_root / rel).read_text(encoding="utf-8")
                algorithm_section = self._section(text, "Algorithms")
                rows = {
                    row[0].lower(): row
                    for row in self._table_rows(algorithm_section)
                }
                mim_note = rows["mim"][2].lower()
                self.assertIn("masked image modeling", mim_note)
                self.assertIn("masked reconstruction", mim_note)
                self.assertNotIn("mutual-information", mim_note)

    def test_related_ssot_docs_list_the_shared_supported_methods(self) -> None:
        surfaces = {
            "docs/adapter_contract.md": (
                "## TTT (Test-Time Training) hooks",
                "\n## ",
            ),
            "docs/yolozu_spec.md": (
                "### 8) Test-time adaptation (TTA / TTT)",
                "\n### ",
            ),
        }
        supported = set(SUPPORTED_TTT_METHODS)
        for rel, (heading, next_heading) in surfaces.items():
            with self.subTest(path=rel):
                text = (self.repo_root / rel).read_text(encoding="utf-8")
                marker = heading
                start = text.find(marker)
                self.assertGreaterEqual(start, 0, f"{rel}: missing {marker}")
                end = text.find(next_heading, start + len(marker))
                section = text[start : end if end >= 0 else len(text)]
                documented = set(re.findall(r"`([a-z]+)`", section)) & supported
                self.assertEqual(documented, supported)


if __name__ == "__main__":
    unittest.main()
