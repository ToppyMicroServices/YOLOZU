import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text.replace("|", "\\|")


def _render_benchmark_option_reference() -> str:
    from yolozu.eval import benchmark_mode

    parser = benchmark_mode.build_parser()
    lines: list[str] = []
    for action in parser._actions:
        option_strings = list(action.option_strings or [])
        if not option_strings:
            continue
        option_label = ", ".join(option_strings)
        if action.nargs != 0:
            if action.choices is not None:
                metavar = "{" + ",".join(str(choice) for choice in action.choices) + "}"
            elif action.metavar is not None:
                metavar = str(action.metavar)
            else:
                metavar = str(action.dest).upper()
            option_label += f" {metavar}"
        if bool(action.required):
            option_label += " [required]"
        lines.append(option_label)
        help_text = " ".join(str(action.help or "").split())
        if help_text:
            lines.append(f"  {help_text}")
    return "\n".join(lines)


def render_cli_reference(repo_root: Path) -> str:
    manifest = json.loads((repo_root / "tools" / "manifest.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["COLUMNS"] = "1000"
    proc = subprocess.run(
        [sys.executable, "-m", "yolozu", "--help"],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"{sys.executable} -m yolozu --help failed:\n{proc.stdout}\n{proc.stderr}")
    benchmark_options = _render_benchmark_option_reference()

    lines = [
        "# Generated CLI Reference",
        "",
        "This file is generated from `python3 -m yolozu --help`, "
        "the benchmark parser, and `tools/manifest.json`.",
        "Keep narrative docs short and link here for the full command surface.",
        "",
        "## Top-level `yolozu --help`",
        "",
        "```text",
        proc.stdout.rstrip(),
        "```",
        "",
        "## `yolozu benchmark` option reference",
        "",
        "```text",
        benchmark_options,
        "```",
        "",
        "## Manifest Tool Registry",
        "",
        "| Tool ID | Maturity | Entry point | Summary |",
        "|---|---|---|---|",
    ]
    for tool in sorted(manifest.get("tools") or [], key=lambda item: str(item.get("id") or "")):
        if not isinstance(tool, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(tool.get("id")),
                    _cell(tool.get("maturity")),
                    _cell(tool.get("entrypoint")),
                    _cell(tool.get("summary")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Smoke Coverage", ""])
    lines.extend(
        [
            "- `tests/test_docs_examples_drift.py` checks documented shell examples against help/manifest flags.",
            "- `tests/test_manual_cli_drift_audit.py` checks manual chapter 04 command references against top-level help.",
            "- `tests/test_generated_cli_reference.py` fails when this generated reference drifts.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


class TestGeneratedCliReference(unittest.TestCase):
    def test_generated_cli_reference_is_current(self):
        repo_root = Path(__file__).resolve().parents[1]
        generated = repo_root / "docs" / "generated" / "cli_reference.md"
        self.assertTrue(generated.is_file(), "missing generated CLI reference")
        expected = render_cli_reference(repo_root)
        actual = generated.read_text(encoding="utf-8")
        self.assertEqual(
            actual,
            expected,
            "generated CLI reference drifted; regenerate from tests.test_generated_cli_reference.render_cli_reference",
        )
        self.assertIn("--openvino-model", actual)
        self.assertIn(
            "--parity-reference-backend {auto,torch,onnx,engine,torchscript,openvino}",
            actual,
        )


if __name__ == "__main__":
    unittest.main()
