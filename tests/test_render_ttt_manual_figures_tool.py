import subprocess
import tempfile
import unittest
from pathlib import Path
import shutil


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

    def test_render_outputs_without_probe_prediction_artifacts(self) -> None:
        pred_no_ttt = REPO_ROOT / "reports" / "ttt_improvement_probe" / "pred_no_ttt.json"
        pred_ttt = REPO_ROOT / "reports" / "ttt_improvement_probe" / "pred_ttt.json"
        backup_no_ttt = pred_no_ttt.with_suffix(".json.bak")
        backup_ttt = pred_ttt.with_suffix(".json.bak")
        shutil.move(pred_no_ttt, backup_no_ttt)
        shutil.move(pred_ttt, backup_ttt)
        try:
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
                self.assertTrue((docs_dir / "ttt_probe_example_panel.png").exists())
        finally:
            shutil.move(backup_no_ttt, pred_no_ttt)
            shutil.move(backup_ttt, pred_ttt)


if __name__ == "__main__":
    unittest.main()
