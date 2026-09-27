from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from yolozu.qualification import qualify_release


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / ".github" / "actions" / "qualify-release"
SUMMARY = ACTION / "render_summary.py"


class TestReleaseQualificationAction(unittest.TestCase):
    def test_action_metadata_exposes_one_spec_gate_and_pinned_dependencies(self):
        metadata = yaml.safe_load((ACTION / "action.yml").read_text(encoding="utf-8"))

        self.assertEqual(metadata["runs"]["using"], "composite")
        self.assertEqual(metadata["inputs"]["spec"]["default"], "yolozu.yaml")
        self.assertEqual(
            metadata["inputs"]["package-spec"]["default"],
            "yolozu[coco]==4.11.0",
        )
        self.assertEqual(
            set(metadata["outputs"]),
            {"decision", "pack-digest", "pack-path", "artifact-id"},
        )
        steps = metadata["runs"]["steps"]
        setup = next(step for step in steps if step["name"] == "Set up Python")
        upload = next(step for step in steps if step.get("id") == "upload")
        qualify = next(step for step in steps if step.get("id") == "qualify")
        self.assertRegex(setup["uses"], r"^actions/setup-python@[0-9a-f]{40}$")
        self.assertRegex(upload["uses"], r"^actions/upload-artifact@[0-9a-f]{40}$")
        self.assertIn(
            'yolozu qualify-release create --spec "$YOLOZU_SPEC"', qualify["run"]
        )
        self.assertNotIn("${{ inputs.spec }}", qualify["run"])

    def test_summary_verifies_hold_pack_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            result = qualify_release(
                ROOT / "data" / "smoke",
                ROOT / "data" / "smoke" / "predictions" / "predictions_dummy.json",
                pack,
                split="val",
                max_images=2,
                dry_run=True,
                min_map50_95=0.0,
            )
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
            output = root / "github-output"
            summary = root / "github-summary"
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "GITHUB_OUTPUT": str(output),
                "GITHUB_STEP_SUMMARY": str(summary),
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY),
                    "--result",
                    str(result_path),
                    "--command-status",
                    "3",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs = dict(
                line.split("=", 1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(outputs["decision"], "hold")
            self.assertEqual(outputs["pack-exists"], "true")
            self.assertEqual(len(outputs["pack-digest"]), 64)
            rendered = summary.read_text(encoding="utf-8")
            self.assertIn("**hold**", rendered)
            self.assertIn("candidate_metrics_available", rendered)

    def test_summary_rejects_command_status_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "result.json"
            result.write_text(
                json.dumps({"decision": "pass", "pack_dir": str(root / "missing")}),
                encoding="utf-8",
            )
            output = root / "github-output"
            summary = root / "github-summary"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SUMMARY),
                    "--result",
                    str(result),
                    "--command-status",
                    "0",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_STEP_SUMMARY": str(summary),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("decision=error", output.read_text(encoding="utf-8"))
            self.assertIn("pack-exists=false", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
