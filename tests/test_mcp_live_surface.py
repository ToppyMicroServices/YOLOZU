from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from yolozu.integrations.tool_reference import (
    build_tool_surface_reference,
    collect_surface_parity_errors,
)


class TestMcpOptionalDependencyMetadata(unittest.TestCase):
    def test_supported_mcp_sdk_range_and_ci_lock_are_aligned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        optional = project["project"]["optional-dependencies"]
        self.assertIn("mcp>=1.26,<2", optional["mcp"])
        self.assertIn("mcp>=1.26,<2", optional["full"])
        lock = (root / "requirements-locks" / "requirements-docs-actions.lock").read_text(
            encoding="utf-8"
        )
        self.assertIn("mcp==1.26.0", lock.splitlines())


@unittest.skipUnless(
    importlib.util.find_spec("mcp") is not None,
    "optional mcp dependency is not installed",
)
class TestMcpLiveSurface(unittest.TestCase):
    def test_stdio_round_trip_lists_and_calls_live_tools(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        repo_root = Path(__file__).resolve().parents[1]

        async def round_trip(workspace: Path):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "yolozu.integrations.mcp_server"],
                cwd=str(workspace),
                env=env,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    called = await session.call_tool(
                        "ai_tools",
                        {
                            "guaranteed": True,
                            "ids_only": True,
                        },
                    )
                    return listed, called

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            listed, called = asyncio.run(round_trip(workspace))
            self.assertFalse((workspace / "runs").exists())
        reference = build_tool_surface_reference()
        self.assertEqual(
            [tool.name for tool in listed.tools],
            [
                item["name"]
                for item in reference["mcp_live_tools"]
            ],
        )
        self.assertFalse(called.isError)
        self.assertEqual(len(called.content), 1)
        payload = json.loads(called.content[0].text)
        self.assertEqual(
            payload["selected_tool_ids"],
            [
                "doctor",
                "generate_config",
                "review_config",
                "validate_predictions",
            ],
        )
        self.assertEqual(payload["surface_counts"]["mcp_live"], 26)

    def test_live_names_and_input_schemas_match_generated_reference(self) -> None:
        from yolozu.integrations.mcp_server import app

        reference = build_tool_surface_reference()
        live_tools = asyncio.run(app.list_tools())
        expected = list(reference["mcp_live_tools"])

        self.assertEqual(
            [tool.name for tool in live_tools],
            [item["name"] for item in expected],
        )
        self.assertFalse(
            any(tool.name.endswith("_tool") for tool in live_tools),
            "internal Python function names leaked into MCP",
        )

        for live, item in zip(live_tools, expected, strict=True):
            with self.subTest(tool=live.name):
                self.assertEqual(live.inputSchema, item["input_schema"])

    def test_actions_shared_parameters_match_full_mcp_schemas(self) -> None:
        reference = build_tool_surface_reference()
        self.assertEqual(collect_surface_parity_errors(reference), [])
        for item in reference["tools"]:
            with self.subTest(tool=item["canonical_name"]):
                self.assertEqual(
                    item["mcp"]["parameter_schema"],
                    item["actions"]["parameter_schema"],
                )
                self.assertTrue(item["parity"]["mcp_vs_actions_schema"])

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi") is not None,
        "optional actions dependency is not installed",
    )
    def test_actions_runtime_schemas_match_generated_reference(self) -> None:
        from yolozu.integrations import actions_api

        reference = build_tool_surface_reference()
        openapi = actions_api.app.openapi()

        def strip_titles(value):
            if isinstance(value, dict):
                return {
                    key: strip_titles(item)
                    for key, item in value.items()
                    if key != "title"
                }
            if isinstance(value, list):
                return [strip_titles(item) for item in value]
            return value

        for item in reference["tools"]:
            bindings = [item["actions"], *item["actions"]["aliases"]]
            for binding in bindings:
                with self.subTest(
                    tool=item["canonical_name"],
                    path=binding["path"],
                ):
                    model_name = binding["request_model"]
                    if model_name:
                        actual = getattr(
                            actions_api,
                            model_name,
                        ).model_json_schema()
                        self.assertEqual(actual, binding["input_schema"])
                        continue

                    operation = openapi["paths"][binding["path"]][
                        binding["method"].lower()
                    ]
                    properties = {}
                    required = []
                    for parameter in operation.get("parameters", []):
                        name = parameter["name"]
                        properties[name] = strip_titles(parameter["schema"])
                        if parameter.get("required"):
                            required.append(name)
                    self.assertEqual(
                        {
                            "properties": properties,
                            "required": required,
                        },
                        binding["parameter_schema"],
                    )

    def test_live_and_actions_surface_sets_are_explicit_and_distinct(self) -> None:
        reference = build_tool_surface_reference()
        surfaces = reference["surfaces"]
        live = set(surfaces["mcp_live"]["tool_ids"])
        actions = set(surfaces["actions_public"]["tool_ids"])
        guaranteed = set(surfaces["guaranteed_ai_safe"]["tool_ids"])
        config_review = set(surfaces["config_review"]["tool_ids"])

        self.assertTrue(actions.issubset(live))
        self.assertTrue(guaranteed.issubset(live))
        self.assertTrue(config_review.issubset(guaranteed))
        self.assertNotIn("generate_config", actions)
        self.assertIn("generate_config", live)

    def test_manifest_override_is_consistent_and_confined_to_workspace(self) -> None:
        from yolozu.integrations.mcp_server import ai_tools_tool

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            manifest = workspace / "manifest.json"
            surface = {"availability": "test", "tool_ids": ["custom_tool"]}
            manifest.write_text(
                json.dumps(
                    {
                        "ai_surfaces": {
                            "mcp_live": surface,
                            "guaranteed_ai_safe": surface,
                            "config_review": surface,
                            "actions_public": surface,
                        },
                        "tools": [
                            {
                                "id": "custom_tool",
                                "maturity": "stable",
                                "tags": ["custom"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                accepted = ai_tools_tool(
                    manifest_path="manifest.json",
                    ids_only=True,
                )
                self.assertTrue(accepted["ok"])
                self.assertEqual(
                    accepted["supported_mcp_tools"],
                    ["custom_tool"],
                )
                self.assertEqual(
                    accepted["selected_tool_ids"],
                    ["custom_tool"],
                )
                self.assertEqual(
                    accepted["surface_counts"]["mcp_live"],
                    1,
                )
                self.assertEqual(
                    accepted["manifest_tools"],
                    ["custom_tool"],
                )
                expanded = ai_tools_tool(
                    manifest_path="manifest.json",
                    guaranteed=True,
                )
                self.assertEqual(
                    expanded["guaranteed_mcp_tools"],
                    ["custom_tool"],
                )
                self.assertEqual(
                    expanded["live_mcp_tools"],
                    ["custom_tool"],
                )
                self.assertEqual(
                    expanded["surfaces"]["mcp_live"]["tool_ids"],
                    ["custom_tool"],
                )
                for unsafe in ("../outside.json", str(outside)):
                    with self.subTest(path=unsafe):
                        rejected = ai_tools_tool(manifest_path=unsafe)
                        self.assertFalse(rejected["ok"])
                        self.assertEqual(
                            rejected["error"]["code"],
                            "unsafe_manifest_path",
                        )
            finally:
                os.chdir(previous)

    def test_review_config_file_read_rejects_traversal_and_absolute_escape(self) -> None:
        from yolozu.integrations.mcp_server import review_config_tool

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                for unsafe in (
                    "nested/../../outside.json",
                    str(outside),
                ):
                    with self.subTest(path=unsafe):
                        rejected = review_config_tool(unsafe)
                        self.assertFalse(rejected["ok"])
                        self.assertEqual(
                            rejected["error"]["code"],
                            "unsafe_or_invalid_config",
                        )
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
