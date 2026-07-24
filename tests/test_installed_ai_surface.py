from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestInstalledAiSurface(unittest.TestCase):
    def test_fresh_wheel_works_from_external_workspace(self) -> None:
        if importlib.util.find_spec("setuptools") is None:
            self.skipTest("wheel build backend is not installed")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheel_dir = root / "wheel"
            target = root / "site"
            workspace = root / "consumer"
            wheel_dir.mkdir()
            target.mkdir()
            workspace.mkdir()
            shutil.copytree(repo_root / "data" / "smoke", workspace / "data" / "smoke")

            build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(repo_root),
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                build.returncode,
                0,
                msg=f"wheel build failed:\n{build.stdout}\n{build.stderr}",
            )
            wheels = sorted(wheel_dir.glob("yolozu-*.whl"))
            self.assertEqual(len(wheels), 1)

            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheels[0]),
                ],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                install.returncode,
                0,
                msg=f"wheel install failed:\n{install.stdout}\n{install.stderr}",
            )

            outside = root / "outside.json"
            script = f"""
import json
from pathlib import Path
import yolozu
from yolozu.integrations.ai_surface import list_manifest_tools, review_config
from yolozu.integrations.tool_runner import eval_coco, validate_predictions

relative = validate_predictions(
    "data/smoke/predictions/predictions_dummy.json",
)
absolute = validate_predictions(
    str(Path.cwd() / "data/smoke/predictions/predictions_dummy.json"),
)
outside = validate_predictions({str(outside)!r})
evaluation = eval_coco(
    "data/smoke",
    "data/smoke/predictions/predictions_dummy.json",
    split="val",
    dry_run=True,
    output="reports/eval.json",
    max_images=2,
)
unsafe_review = review_config(
    {{
        "tool": "eval_coco",
        "arguments": {{"output": {str(outside)!r}}},
        "safety": {{}},
    }},
    workspace_root=".",
)
print(json.dumps({{
    "module": yolozu.__file__,
    "guaranteed_ids": list_manifest_tools(
        guaranteed=True,
        ids_only=True,
    ),
    "supported_ids": list_manifest_tools(
        supported=True,
        ids_only=True,
    ),
    "relative": relative,
    "absolute": absolute,
    "outside": outside,
    "evaluation": evaluation,
    "report_exists": Path("reports/eval.json").is_file(),
    "unsafe_review": unsafe_review,
}}, sort_keys=True))
"""
            env = os.environ.copy()
            env["PYTHONPATH"] = str(target)
            env["PYTHONNOUSERSITE"] = "1"
            mcp_entry = target / "bin" / "yolozu-mcp"
            self.assertTrue(
                mcp_entry.is_file(),
                msg=f"installed console entry is missing: {mcp_entry}",
            )
            help_run = subprocess.run(
                [str(mcp_entry), "--help"],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                help_run.returncode,
                0,
                msg=f"installed yolozu-mcp --help failed:\n"
                f"{help_run.stdout}\n{help_run.stderr}",
            )
            self.assertIn("--print-tools", help_run.stdout)
            self.assertFalse((workspace / "runs").exists())
            discovery = subprocess.run(
                [
                    str(mcp_entry),
                    "--print-tools",
                    "--guaranteed",
                    "--ids-only",
                ],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                discovery.returncode,
                0,
                msg=f"installed discovery failed:\n"
                f"{discovery.stdout}\n{discovery.stderr}",
            )
            discovery_payload = json.loads(discovery.stdout)
            self.assertEqual(
                discovery_payload["selected_tool_ids"],
                [
                    "doctor",
                    "generate_config",
                    "review_config",
                    "validate_predictions",
                ],
            )
            self.assertLess(len(discovery.stdout.encode("utf-8")), 1_500)
            self.assertFalse((workspace / "runs").exists())
            missing_extra = subprocess.run(
                [sys.executable, "-S", str(mcp_entry)],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(missing_extra.returncode, 2)
            self.assertIn("yolozu[mcp]", missing_extra.stderr)
            self.assertFalse((workspace / "runs").exists())

            run = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                run.returncode,
                0,
                msg=f"external workspace run failed:\n{run.stdout}\n{run.stderr}",
            )
            payload = json.loads(run.stdout)
            self.assertTrue(str(payload["module"]).startswith(str(target)))
            self.assertEqual(
                payload["guaranteed_ids"],
                [
                    "doctor",
                    "generate_config",
                    "review_config",
                    "validate_predictions",
                ],
            )
            self.assertEqual(len(payload["supported_ids"]), 25)
            self.assertTrue(payload["relative"]["ok"])
            self.assertTrue(payload["relative"]["validation"]["ok"])
            self.assertTrue(payload["absolute"]["ok"])
            self.assertFalse(payload["outside"]["ok"])
            self.assertIn("path escapes workspace", payload["outside"]["error"])
            self.assertTrue(
                payload["evaluation"]["ok"],
                msg=json.dumps(payload["evaluation"], indent=2),
            )
            self.assertTrue(payload["report_exists"])
            self.assertFalse(payload["unsafe_review"]["ok"])


if __name__ == "__main__":
    unittest.main()
