from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image
from tools import prepare_image_service_plugin as builder


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "yolozu-image-service"
CLIENT = PLUGIN / "scripts" / "image_service_client.py"
spec = importlib.util.spec_from_file_location("yolozu_plugin_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def png_bytes() -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(data, format="PNG")
    return data.getvalue()


class TestPluginFiles(unittest.TestCase):
    def test_identity_surface_and_references(self):
        manifest = json.loads((PLUGIN / ".codex-plugin/plugin.json").read_text())
        self.assertEqual(manifest["name"], PLUGIN.name)
        self.assertEqual(manifest["mcpServers"], "./.mcp.json")
        self.assertTrue((PLUGIN / manifest["skills"] / PLUGIN.name / "SKILL.md").is_file())
        self.assertNotIn("hooks", manifest)
        self.assertNotIn("apps", manifest)
        servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
        self.assertEqual(set(servers), {PLUGIN.name})
        self.assertEqual(servers[PLUGIN.name]["args"], client.SERVER_ARGS)
        self.assertNotIn("url", servers[PLUGIN.name])
        source_manifest = json.loads((ROOT / "tools/manifest.json").read_text())
        self.assertEqual(set(source_manifest["ai_surfaces"]["image_service_safe"]["tool_ids"]), client.TOOL_NAMES)

    def test_help_requires_no_optional_runtime(self):
        for path in (CLIENT, ROOT / "tools/prepare_image_service_plugin.py"):
            run = subprocess.run([sys.executable, "-S", str(path), "--help"], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("usage:", run.stdout)

    def test_default_is_preview(self):
        args = client.parse_args(["--image", "a.png", "--class", "cat"])
        self.assertFalse(args.execute)
        self.assertEqual(args.timeout, 60)

    def test_bad_arguments(self):
        cases = (["--image", "a.png"], ["--capabilities", "--execute"],
                 ["--capabilities", "--class", "cat"],
                 *(["--capabilities", "--timeout", n] for n in ("nan", "inf", "0", "121")))
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                client.parse_args(args)

    def test_image_bytes_not_suffix_determine_mime(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "not-a-jpeg.jpg"
            path.write_bytes(png_bytes())
            result = client.read_image(path)
            self.assertEqual(result["media_type"], "image/png")
            self.assertEqual(base64.b64decode(result["content_base64"]), path.read_bytes())

    def test_invalid_oversize_nonregular_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.png"
            path.write_bytes(b"not an image")
            with self.assertRaises((ValueError, OSError)):
                client.read_image(path)
            path.write_bytes(png_bytes())
            with patch.object(client, "MAX_IMAGE_BYTES", 4), self.assertRaises(ValueError):
                client.read_image(path)
            linked = Path(td) / "link.png"
            linked.symlink_to(path)
            with self.assertRaises(ValueError):
                client.read_image(linked)
            with self.assertRaises((ValueError, OSError)):
                client.read_image(Path(td))
            if hasattr(os, "mkfifo"):
                fifo = Path(td) / "pipe"
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):
                    client.read_image(fifo)

    def test_format_and_dimension_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image"
            Image.new("RGB", (8, 6)).save(path, format="BMP")
            with self.assertRaises(ValueError):
                client.read_image(path)
            path.write_bytes(png_bytes())
            with patch.object(client, "MAX_DIMENSION", 4), self.assertRaises(ValueError):
                client.read_image(path)
            with patch.object(client, "MAX_PIXELS", 4), self.assertRaises(ValueError):
                client.read_image(path)

    def test_builder_refuses_existing_output_before_runtime_check(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / PLUGIN.name
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep")
            with patch.object(builder.subprocess, "run") as run, self.assertRaises(ValueError):
                builder.prepare(target, Path(sys.executable))
            run.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_builder_rejects_wrong_surface_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / PLUGIN.name
            wrong = SimpleNamespace(returncode=0, stdout='["ai_tools"]')
            with patch.object(builder.subprocess, "run", return_value=wrong), self.assertRaises(ValueError):
                builder.prepare(target, Path(sys.executable))
            self.assertFalse(target.exists())


class TestClientWorkflow(unittest.IsolatedAsyncioTestCase):
    def session(self, names=client.TOOL_NAMES):
        return SimpleNamespace(initialize=AsyncMock(), list_tools=AsyncMock(return_value=SimpleNamespace(
            tools=[SimpleNamespace(name=name) for name in names])), call_tool=AsyncMock())

    async def test_wrong_surface_stops_before_upload(self):
        session = self.session({"ai_tools"})
        with patch.object(client, "call", new_callable=AsyncMock) as call, self.assertRaises(ValueError):
            await client.request(session, SimpleNamespace(), {})
        call.assert_not_called()

    async def test_preview_and_abstention_are_preserved(self):
        final = {"ok": True, "job": {"status": "succeeded", "result": {"outcome": "abstained"}}}
        responses = [{"ok": True}, {"asset": {"asset_id": "asset_test"}}, {"job": {"job_id": "job_test"}}, final]
        args = argparse.Namespace(classes=["cat"], execute=False, timeout=60)
        with patch.object(client, "call", new_callable=AsyncMock, side_effect=responses) as call:
            self.assertEqual(await client.request(self.session(), args, {}), final)
            self.assertFalse(call.call_args_list[2].args[2]["execute"])

    async def test_timeout_requests_cancel_once(self):
        responses = [{"ok": True}, {"asset": {"asset_id": "asset_test"}},
                     {"job": {"job_id": "job_test"}}, {"job": {"status": "running"}},
                     {"ok": True, "cancelled": True}]
        args = argparse.Namespace(classes=["cat"], execute=False, timeout=0)
        with patch.object(client, "call", new_callable=AsyncMock, side_effect=responses) as call:
            result = await client.request(self.session(), args, {})
            self.assertEqual(result["outcome"], "timed_out")
            self.assertEqual(call.call_args_list[-1].args[1], "cancel_image_job")
            self.assertEqual(call.call_count, 5)

    async def test_tool_error_is_not_success(self):
        session = self.session()
        session.call_tool.return_value = SimpleNamespace(isError=False, structuredContent={"ok": False})
        with self.assertRaises(ValueError):
            await client.call(session, "put_image_asset", {})


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional mcp dependency")
class TestPluginLive(unittest.TestCase):
    def test_prepared_relocated_plugin_and_image_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "relocated" / PLUGIN.name
            prepared = builder.prepare(target, Path(sys.executable))
            self.assertFalse(prepared["installed_in_host"])
            self.assertFalse(prepared["models_qualified"])
            settings = json.loads((target / ".mcp.json").read_text())["mcpServers"][PLUGIN.name]
            self.assertEqual(settings["command"], os.path.abspath(sys.executable))
            self.assertTrue((target / "LICENSE").is_file())
            self.assertEqual({str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}, set(builder.FILES) | {"LICENSE"})

            async def roundtrip():
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                params = StdioServerParameters(**settings, cwd=str(root))
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self.assertEqual({tool.name for tool in (await session.list_tools()).tools}, client.TOOL_NAMES)
                        result = await client.call(session, "image_service_capabilities")
                        self.assertEqual(result["service"]["selection_policy"], "qualified_registered_pipeline_or_abstain")

            asyncio.run(asyncio.wait_for(roundtrip(), 20))
            self.assertFalse((root / "runs").exists())
            image = root / "input.png"
            image.write_bytes(png_bytes())
            for execute in ([], ["--execute"]):
                run = subprocess.run([
                    settings["command"], "-I", str(target / "scripts/image_service_client.py"),
                    "--image", str(image), "--class", "cat", *execute,
                ], cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                result = json.loads(run.stdout)
                self.assertEqual(result["job"]["result"]["outcome"], "abstained")
                self.assertNotIn(base64.b64encode(image.read_bytes()).decode(), run.stdout)
                self.assertFalse((root / "runs").exists())
            self.assertEqual(image.read_bytes(), png_bytes())


if __name__ == "__main__":
    unittest.main()
