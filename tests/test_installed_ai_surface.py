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
            source = root / "source"
            target = root / "site"
            workspace = root / "consumer"
            wheel_dir.mkdir()
            target.mkdir()
            workspace.mkdir()
            shutil.copytree(
                repo_root,
                source,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "build",
                    "dist",
                    "*.egg-info",
                    "__pycache__",
                ),
            )
            shutil.copytree(repo_root / "data" / "smoke", workspace / "data" / "smoke")
            invalid_path = workspace / "invalid_predictions.json"
            invalid_payload = json.loads(
                (
                    workspace
                    / "data"
                    / "smoke"
                    / "predictions"
                    / "predictions_dummy.json"
                ).read_text(encoding="utf-8")
            )
            invalid_payload["predictions"][0]["detections"][0]["score"] = 1.5
            invalid_path.write_text(
                json.dumps(invalid_payload),
                encoding="utf-8",
            )

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
                    str(source),
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
from yolozu.api import (
    APIError,
    evaluate_coco as api_evaluate_coco,
    validate_predictions as api_validate_predictions,
)
from yolozu.integrations.ai_surface import list_manifest_tools, review_config
from yolozu.integrations.tool_runner import (
    eval_coco,
    validate_predictions as runner_validate_predictions,
)

relative = runner_validate_predictions(
    "data/smoke/predictions/predictions_dummy.json",
)
absolute = runner_validate_predictions(
    str(Path.cwd() / "data/smoke/predictions/predictions_dummy.json"),
)
outside = runner_validate_predictions({str(outside)!r})
strict_invalid = runner_validate_predictions(
    "invalid_predictions.json",
    strict=True,
)
repaired_invalid = runner_validate_predictions(
    "invalid_predictions.json",
    strict=False,
)
evaluation = eval_coco(
    "data/smoke",
    "data/smoke/predictions/predictions_dummy.json",
    split="val",
    dry_run=True,
    output="reports/eval.json",
    max_images=2,
)
strict_evaluation = eval_coco(
    "data/smoke",
    "invalid_predictions.json",
    split="val",
    dry_run=True,
    output="reports/eval_strict_failed.json",
    max_images=2,
)
repaired_evaluation = eval_coco(
    "data/smoke",
    "invalid_predictions.json",
    split="val",
    dry_run=True,
    output="reports/eval_repaired.json",
    max_images=2,
    repair=True,
)

valid_api = api_validate_predictions(
    Path.cwd() / "data/smoke/predictions/predictions_dummy.json",
)
try:
    api_validate_predictions(Path.cwd() / "invalid_predictions.json")
except APIError as exc:
    strict_api_error = exc.to_dict()
else:
    strict_api_error = None
repaired_api = api_validate_predictions(
    Path.cwd() / "invalid_predictions.json",
    repair=True,
)
try:
    api_evaluate_coco(
        Path.cwd() / "data/smoke",
        Path.cwd() / "invalid_predictions.json",
        split="val",
        dry_run=True,
        max_images=2,
    )
except APIError as exc:
    strict_api_eval_error = exc.to_dict()
else:
    strict_api_eval_error = None
repaired_api_eval = api_evaluate_coco(
    Path.cwd() / "data/smoke",
    Path.cwd() / "invalid_predictions.json",
    split="val",
    dry_run=True,
    max_images=2,
    repair=True,
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
    "strict_invalid": strict_invalid,
    "repaired_invalid": repaired_invalid,
    "evaluation": evaluation,
    "strict_evaluation": strict_evaluation,
    "repaired_evaluation": repaired_evaluation,
    "strict_evaluation_report": json.loads(
        Path("reports/eval_strict_failed.json").read_text()
    ),
    "repaired_evaluation_report": json.loads(
        Path("reports/eval_repaired.json").read_text()
    ),
    "valid_api": valid_api.to_dict(include_entries=False),
    "strict_api_error": strict_api_error,
    "repaired_api": repaired_api.to_dict(include_entries=False),
    "strict_api_eval_error": strict_api_eval_error,
    "repaired_api_eval": repaired_api_eval.to_dict(),
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
            self.assertFalse(payload["strict_invalid"]["ok"])
            self.assertEqual(
                payload["strict_invalid"]["validation"]["mode"],
                "strict",
            )
            self.assertTrue(payload["repaired_invalid"]["ok"])
            self.assertEqual(
                payload["repaired_invalid"]["validation"]["mode"],
                "repair",
            )
            self.assertTrue(
                payload["repaired_invalid"]["validation"][
                    "repair_enabled"
                ]
            )
            self.assertTrue(
                any(
                    "score: out of range" in warning
                    for warning in payload["repaired_invalid"][
                        "validation"
                    ]["warnings"]
                )
            )
            self.assertTrue(
                payload["evaluation"]["ok"],
                msg=json.dumps(payload["evaluation"], indent=2),
            )
            self.assertFalse(payload["strict_evaluation"]["ok"])
            self.assertEqual(
                payload["strict_evaluation_report"]["status"],
                "failed",
            )
            self.assertTrue(payload["repaired_evaluation"]["ok"])
            repaired_report = payload["repaired_evaluation_report"]
            self.assertEqual(repaired_report["status"], "ok")
            self.assertEqual(
                repaired_report["validation"]["mode"],
                "repair",
            )
            self.assertEqual(
                repaired_report["counts"],
                {
                    "dataset_images_total": 10,
                    "detections": 10,
                    "detections_excluded": 40,
                    "detections_input": 50,
                    "images": 2,
                    "prediction_images_evaluated": 2,
                    "prediction_images_excluded": 8,
                    "prediction_images_total": 10,
                    "selected_images_without_predictions": 0,
                },
            )
            self.assertTrue(payload["valid_api"]["ok"])
            self.assertEqual(payload["valid_api"]["mode"], "strict")
            self.assertIsNotNone(payload["strict_api_error"])
            self.assertEqual(
                payload["strict_api_error"]["code"],
                "E_PREDICTIONS_INVALID",
            )
            self.assertEqual(payload["repaired_api"]["mode"], "repair")
            self.assertIsNotNone(payload["strict_api_eval_error"])
            self.assertEqual(
                payload["strict_api_eval_error"]["code"],
                "E_PREDICTIONS_INVALID",
            )
            self.assertEqual(
                payload["repaired_api_eval"]["counts"],
                repaired_report["counts"],
            )
            self.assertTrue(payload["report_exists"])
            self.assertFalse(payload["unsafe_review"]["ok"])

            if importlib.util.find_spec("mcp") is not None:
                live_script = f"""
import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    child_env = os.environ.copy()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "yolozu.integrations.mcp_server"],
        cwd={str(workspace)!r},
        env=child_env,
    )
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            listed = await session.list_tools()
            called = await session.call_tool(
                "validate_predictions",
                {{
                    "path": "data/smoke/predictions/predictions_dummy.json",
                    "strict": True,
                }},
            )
            print(json.dumps({{
                "names": [tool.name for tool in listed.tools],
                "called_error": called.isError,
                "called": json.loads(called.content[0].text),
            }}, sort_keys=True))

asyncio.run(main())
"""
                live = subprocess.run(
                    [sys.executable, "-c", live_script],
                    cwd=str(workspace),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    live.returncode,
                    0,
                    msg=(
                        "installed-wheel MCP round trip failed:\n"
                        f"{live.stdout}\n{live.stderr}"
                    ),
                )
                live_payload = json.loads(live.stdout)
                self.assertEqual(
                    set(live_payload["names"]),
                    set(payload["supported_ids"]),
                )
                self.assertFalse(live_payload["called_error"])
                self.assertTrue(live_payload["called"]["ok"])
                self.assertEqual(
                    live_payload["called"]["validation"]["mode"],
                    "strict",
                )


if __name__ == "__main__":
    unittest.main()
