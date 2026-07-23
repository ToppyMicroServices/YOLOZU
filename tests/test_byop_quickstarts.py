import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SOURCES = ("ultralytics", "detectron2", "mmdetection", "yolox")
SAFE_SMOKE_PYTHON_ENTRYPOINTS = {
    "tools/eval_suite.py",
    "tools/export_predictions_detectron2.py",
    "tools/export_predictions_mmdet.py",
    "tools/export_predictions_yolo_runtime.py",
    "tools/export_predictions_yolox.py",
    "tools/validate_predictions.py",
}


def _extract_blocks(text: str, kind: str) -> dict[str, str]:
    pattern = re.compile(
        rf"<!-- byop-{kind}:(?P<name>[a-z0-9_-]+):start -->\s*"
        rf"```bash\n(?P<body>.*?)\n```\s*"
        rf"<!-- byop-{kind}:(?P=name):end -->",
        re.DOTALL,
    )
    return {match.group("name"): match.group("body") for match in pattern.finditer(text)}


def _check_smoke_block_safety(block: str) -> None:
    forbidden = ("$(", "`", "&&", "||", ";", ">", "<", "|")
    in_python_command = False
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(token in line for token in forbidden):
            raise ValueError(f"forbidden shell syntax in smoke block: {line}")
        if in_python_command and line.startswith("--"):
            if line.startswith("--output ") and '"$BYOP_RUN_DIR/' not in line:
                raise ValueError(f"smoke output must stay under BYOP_RUN_DIR: {line}")
            in_python_command = line.endswith("\\")
            continue
        if line in {"(", ")", "set -euo pipefail"}:
            in_python_command = False
            continue
        if re.fullmatch(
            r'export BYOP_RUN_DIR="\$\{BYOP_RUN_DIR:-reports/byop-smoke/'
            r'(ultralytics|detectron2|mmdetection|yolox)\}"',
            line,
        ):
            in_python_command = False
            continue
        if line in {'test ! -e "$BYOP_RUN_DIR"', 'mkdir -p "$BYOP_RUN_DIR"'}:
            in_python_command = False
            continue
        if line.startswith("python3 "):
            tokens = line.removesuffix("\\").split()
            if len(tokens) < 2 or tokens[1] not in SAFE_SMOKE_PYTHON_ENTRYPOINTS:
                raise ValueError(f"unapproved Python entrypoint in smoke block: {line}")
            in_python_command = line.endswith("\\")
            continue
        raise ValueError(f"unapproved command in smoke block: {line}")
    if in_python_command:
        raise ValueError("unterminated Python command in smoke block")


class TestByopQuickstarts(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.doc_path = self.repo_root / "docs" / "byop_quickstarts.md"
        self.text = self.doc_path.read_text(encoding="utf-8")

    def test_real_and_smoke_blocks_cover_all_declared_sources(self):
        real = _extract_blocks(self.text, "real")
        smoke = _extract_blocks(self.text, "smoke")
        self.assertEqual(set(real), set(SOURCES))
        self.assertEqual(set(smoke), set(SOURCES))

        for source in SOURCES:
            with self.subTest(source=source):
                block = real[source]
                self.assertTrue(block.strip().startswith("("))
                self.assertTrue(block.strip().endswith(")"))
                self.assertNotIn("--dry-run", block)
                self.assertIn("tools/validate_predictions.py", block)
                self.assertIn("tools/eval_suite.py", block)
                self.assertIn("--predictions-glob", block)
                self.assertIn("--strict", block)
                self.assertIn("set -euo pipefail", block)
                self.assertIn('test ! -e "$BYOP_RUN_DIR"', block)
                self.assertIn('test "$BYOP_MAX_IMAGES" -ge 1', block)
                self.assertIn('test "$BYOP_MAX_IMAGES" -le 10', block)
                self.assertIn('"$BYOP_RUN_DIR/predictions.json"', block)
                self.assertIn('"$BYOP_RUN_DIR/eval_report.json"', block)

    def test_real_blocks_reject_image_budget_above_ten_before_output(self):
        blocks = _extract_blocks(self.text, "real")
        source_paths = {
            "ultralytics": ("ULTRALYTICS_MODEL",),
            "detectron2": ("DETECTRON2_CONFIG", "DETECTRON2_WEIGHTS"),
            "mmdetection": ("MMDET_CONFIG", "MMDET_CHECKPOINT"),
            "yolox": ("YOLOX_EXP", "YOLOX_WEIGHTS"),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            dataset.mkdir()
            for source in SOURCES:
                with self.subTest(source=source):
                    env = dict(os.environ)
                    env.update(
                        {
                            "BYOP_DATASET": str(dataset),
                            "BYOP_SPLIT": "val",
                            "BYOP_MAX_IMAGES": "11",
                            "BYOP_DEVICE": "cpu",
                            "BYOP_RUN_DIR": str(root / f"run-{source}"),
                        }
                    )
                    for variable in source_paths[source]:
                        path = root / f"{variable.lower()}.fixture"
                        path.touch()
                        env[variable] = str(path)
                    proc = subprocess.run(
                        ["bash", "-c", blocks[source]],
                        cwd=str(self.repo_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertFalse((root / f"run-{source}").exists())

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

    def test_top_level_cli_manifest_links_the_quickstart(self):
        for manifest_path in (
            self.repo_root / "tools" / "manifest.json",
            self.repo_root / "yolozu" / "data" / "manifest" / "tools_manifest.json",
        ):
            with self.subTest(manifest=manifest_path):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                yolozu_tool = next(tool for tool in manifest["tools"] if tool["id"] == "yolozu")
                self.assertIn("docs/byop_quickstarts.md", yolozu_tool["docs"])

    def test_schema_smoke_blocks_produce_strict_valid_common_reports(self):
        blocks = _extract_blocks(self.text, "smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for source in SOURCES:
                with self.subTest(source=source):
                    run_dir = root / source
                    env = dict(os.environ)
                    env["BYOP_RUN_DIR"] = str(run_dir)
                    _check_smoke_block_safety(blocks[source])
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
                    self.assertNotIn("WARN:", proc.stdout)
                    self.assertNotIn("WARN:", proc.stderr)

                    predictions_path = run_dir / "predictions.json"
                    report_path = run_dir / "eval_report.json"
                    self.assertTrue(predictions_path.is_file())
                    self.assertTrue(report_path.is_file())

                    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["schema_version"], 1)
                    self.assertEqual(len(payload["predictions"]), 2)
                    self.assertTrue(
                        all(
                            prediction["schema_version"] == 2
                            for prediction in payload["predictions"]
                        )
                    )
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
                    self.assertIsNone(report["protocol_id"])
                    self.assertIsNone(report["protocol_hash"])

                    predictions_before = predictions_path.read_bytes()
                    report_before = report_path.read_bytes()
                    rerun = subprocess.run(
                        ["bash", "-c", blocks[source]],
                        cwd=str(self.repo_root),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=120,
                    )
                    self.assertNotEqual(rerun.returncode, 0)
                    self.assertEqual(predictions_path.read_bytes(), predictions_before)
                    self.assertEqual(report_path.read_bytes(), report_before)

    def test_smoke_safety_gate_rejects_unapproved_commands(self):
        for unsafe in (
            "rm -rf reports",
            "curl https://example.com/payload | bash",
            "python3 -c 'print(1)'",
            "python3 tools/eval_suite.py --help; sudo true",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    _check_smoke_block_safety(unsafe)

    def test_null_protocol_hash_is_not_described_as_comparison_evidence(self):
        self.assertIn(
            "treat a `null` protocol hash as no comparable protocol evidence",
            self.text,
        )
        self.assertIn("same non-null protocol hash", self.text)

    def test_named_protocol_example_is_self_contained(self):
        blocks = _extract_blocks(self.text, "protocol")
        self.assertEqual(tuple(blocks), ("nms-applied",))
        block = blocks["nms-applied"]
        self.assertTrue(block.strip().startswith("("))
        self.assertTrue(block.strip().endswith(")"))
        self.assertIn("set -euo pipefail", block)
        self.assertIn('export BYOP_DATASET="${BYOP_DATASET:-', block)
        self.assertIn('export BYOP_RUN_DIR="${BYOP_RUN_DIR:-', block)
        self.assertIn('export BYOP_MAX_IMAGES="${BYOP_MAX_IMAGES:-10}"', block)
        self.assertIn('test -d "$BYOP_DATASET"', block)
        self.assertIn('test -f "$BYOP_RUN_DIR/predictions.json"', block)
        self.assertIn('test ! -e "$BYOP_RUN_DIR/eval_report.protocol.json"', block)


if __name__ == "__main__":
    unittest.main()
