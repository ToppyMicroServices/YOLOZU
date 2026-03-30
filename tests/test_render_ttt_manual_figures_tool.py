import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRenderTTTManualFiguresTool(unittest.TestCase):
    def test_help(self) -> None:
        proc = subprocess.run(
            ["python3", "tools/render_ttt_manual_figures.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Render beginner-friendly TTT manual figures", proc.stdout)

    def test_render_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            docs_dir = Path(tmp) / "docs_assets"
            manual_dir = Path(tmp) / "manual_figures"
            proc = subprocess.run(
                [
                    "python3",
                    "tools/render_ttt_manual_figures.py",
                    "--source-json",
                    "docs/assets/ttt_method_results_source.json",
                    "--docs-assets-dir",
                    str(docs_dir.relative_to(REPO_ROOT)),
                    "--manual-figures-dir",
                    str(manual_dir.relative_to(REPO_ROOT)),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            expected = {
                "ttt_method_results_summary.png",
                "ttt_compare_pipeline.png",
                "ttt_probe_example_panel.png",
            }
            self.assertEqual(expected, {p.name for p in docs_dir.glob("*.png")})
            self.assertEqual(expected, {p.name for p in manual_dir.glob("*.png")})


if __name__ == "__main__":
    unittest.main()
