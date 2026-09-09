import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.audit_manual_cli_drift import _extract_manual_yolozu_commands


class TestManualCliDriftAudit(unittest.TestCase):
    def test_manual_cli_drift_audit_passes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "tools" / "audit_manual_cli_drift.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--allowlist",
                "docs/manual_cli_drift_allowlist.json",
                "--json",
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(f"audit_manual_cli_drift.py failed:\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(proc.stdout)
        self.assertTrue(payload.get("ok"))
        self.assertIn("benchmark", payload.get("documented_commands") or [])
        self.assertIn("train", payload.get("documented_commands") or [])
        self.assertIn("benchmark", payload.get("canonical_commands") or [])
        self.assertEqual(len(payload["manual_files"]), len(list((repo_root / "manual" / "chapters").glob("*.tex"))))

    def test_unknown_command_in_listing_fails(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            chapter = Path(tmp) / "installation.tex"
            chapter.write_text(
                "\\begin{lstlisting}[language=bash]\nyolozu eval-suite --help\n\\end{lstlisting}\n",
                encoding="utf-8",
            )
            proc = subprocess.run(
                [sys.executable, str(repo_root / "tools" / "audit_manual_cli_drift.py"),
                 "--manual", str(chapter), "--skip-wrapper", "--json"],
                cwd=repo_root, capture_output=True, text=True, check=False,
            )
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["missing_from_cli"], ["eval-suite"])

    def test_extracts_commands_and_ignores_comments_and_placeholders(self):
        source = r"""
% \cmd{yolozu deleted-comment}
Text % \cmd{yolozu inline-comment}
\cmd{yolozu <command>}
\cmd{yolozu --help}
\cmd{python3 -m yolozu doctor}
\begin{lstlisting}[language=bash]
# yolozu commented-command
$ yolozu validate predictions demo.json --strict
python -m yolozu \
  eval-coco --help
yolozu <placeholder> --help
\end{lstlisting}
"""
        with tempfile.TemporaryDirectory() as tmp:
            chapter = Path(tmp) / "chapter.tex"
            chapter.write_text(source, encoding="utf-8")
            commands = _extract_manual_yolozu_commands(chapter, allowlist={})
        self.assertEqual(commands, ["doctor", "eval-coco", "validate"])


if __name__ == "__main__":
    unittest.main()
