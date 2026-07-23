import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCES = ("ultralytics", "detectron2", "mmdetection", "yolox")


def _extract_blocks(text: str, kind: str) -> dict[str, str]:
    pattern = re.compile(
        rf"<!-- byop-{kind}:(?P<name>[a-z0-9_-]+):start -->\s*"
        rf"```bash\n(?P<body>.*?)\n```\s*"
        rf"<!-- byop-{kind}:(?P=name):end -->",
        re.DOTALL,
    )
    return {match.group("name"): match.group("body") for match in pattern.finditer(text)}


class TestByopQuickstarts(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.doc_path = self.repo_root / "docs" / "byop_quickstarts.md"
        self.text = self.doc_path.read_text(encoding="utf-8")

    def test_real_and_smoke_blocks_cover_all_declared_sources(self):
        real = _extract_blocks(self.text, "real")
        smoke = _extract_blocks(self.text, "smoke")
        self.assertEqual(tuple(real), SOURCES)
        self.assertEqual(tuple(smoke), SOURCES)

        for source, block in real.items():
            with self.subTest(source=source):
                self.assertNotIn("--dry-run", block)
                self.assertIn("tools/validate_predictions.py", block)
                self.assertIn("tools/eval_suite.py", block)
                self.assertIn("--predictions-glob", block)
                self.assertIn("--strict", block)
                self.assertIn('"$BYOP_RUN_DIR/predictions.json"', block)
                self.assertIn('"$BYOP_RUN_DIR/eval_report.json"', block)

    def test_documented_commands_match_declared_cli_surfaces(self):
        proc = subprocess.run(
            [
                sys.executable,
                "tools/audit_docs_examples_drift.py",
                "--docs",
                "docs/byop_quickstarts.md",
                "--skip-manual",
                "--skip-manifest",
                "--json",
            ],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"BYOP command drift audit failed:\n{proc.stdout}\n{proc.stderr}",
        )
        report = json.loads(proc.stdout)
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["checked_examples"], 25)

    def test_schema_smoke_blocks_produce_strict_valid_common_reports(self):
        blocks = _extract_blocks(self.text, "smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for source in SOURCES:
                with self.subTest(source=source):
                    run_dir = root / source
                    env = dict(os.environ)
                    env["BYOP_RUN_DIR"] = str(run_dir)
                    proc = subprocess.run(
                        ["bash", "-eu", "-o", "pipefail", "-c", blocks[source]],
                        cwd=str(self.repo_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=120,
                    )
                    self.assertEqual(
                        proc.returncode,
                        0,
                        msg=f"{source} schema smoke failed:\n{proc.stdout}\n{proc.stderr}",
                    )

                    predictions_path = run_dir / "predictions.json"
                    report_path = run_dir / "eval_report.json"
                    self.assertTrue(predictions_path.is_file())
                    self.assertTrue(report_path.is_file())

                    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
                    self.assertEqual(len(payload["predictions"]), 2)
                    extra = payload["meta"]["extra"]
                    self.assertEqual(extra["exporter"], source)
                    self.assertEqual(extra["execution_status"], "dry_run")
                    self.assertIs(extra["runtime_executed"], False)
                    self.assertEqual(extra["inference_calls"], 0)

                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(len(report["results"]), 1)
                    result = report["results"][0]
                    self.assertIs(result["dry_run"], True)
                    self.assertEqual(result["counts"]["images"], 2)
                    self.assertIsInstance(result["export_settings"], dict)
                    self.assertEqual(result["export_settings"]["bbox_format"], "cxcywh_norm")
                    self.assertEqual(result["export_settings"]["score_threshold"], 0.25)
                    self.assertEqual(result["export_settings"]["max_detections"], 300)
                    self.assertEqual(result["predictions_meta_ref"]["exporter"], source)


if __name__ == "__main__":
    unittest.main()
