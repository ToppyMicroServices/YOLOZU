"""Exercise the service MCP protocol locally, with no provider API calls."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

from tests.test_image_service import _png_bytes


class TestHTTPBounds(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, messages, *, headers=(), slow=False):
        from yolozu.integrations.image_service_http import ImageServiceHTTPBounds

        responses = []
        forwarded = []

        async def receive():
            if slow:
                await asyncio.sleep(1)
            return messages.pop(0)

        async def send(message):
            responses.append(message)

        async def app(scope, receive, send):
            forwarded.append(await receive())

        with (
            patch("yolozu.integrations.image_service_http.MAX_HTTP_BODY_BYTES", 8),
            patch("yolozu.integrations.image_service_http.HTTP_UPLOAD_TIMEOUT_SECONDS", 0.02),
        ):
            await ImageServiceHTTPBounds(app)({"type": "http", "method": "POST", "headers": headers}, receive, send)
        return responses, forwarded

    async def test_declared_and_chunked_oversize_rejected_before_mcp(self):
        for headers, messages in (
            ([(b"content-length", b"9")], []),
            ([], [{"type": "http.request", "body": b"1234", "more_body": True},
                  {"type": "http.request", "body": b"56789", "more_body": False}]),
        ):
            responses, forwarded = await self.exercise(messages, headers=headers)
            self.assertEqual(responses[0]["status"], 413)
            self.assertEqual(forwarded, [])

    async def test_stalled_upload_times_out_before_mcp(self):
        responses, forwarded = await self.exercise([], slow=True)
        self.assertEqual(responses[0]["status"], 408)
        self.assertEqual(forwarded, [])

    async def test_bounded_chunked_upload_is_replayed_exactly(self):
        responses, forwarded = await self.exercise([
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ])
        self.assertEqual(responses, [])
        self.assertEqual(forwarded, [{"type": "http.request", "body": b"12345678", "more_body": False}])


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional mcp dependency")
class TestImageServiceHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_five_tool_roundtrip_abstains_without_a_model(self):
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.server.auth.settings import AuthSettings
        from mcp.server.transport_security import TransportSecuritySettings

        from yolozu.integrations.image_service import close_image_service, configure_image_service
        from yolozu.integrations.mcp_server import _StaticTokenVerifier, service_app

        token = "local-test-credential-" + "x" * 32
        settings = service_app.settings
        saved = (settings.auth, settings.stateless_http, settings.json_response, settings.transport_security, service_app._token_verifier)
        settings.stateless_http = True
        settings.json_response = True
        settings.auth = AuthSettings(issuer_url="https://vision.example", resource_server_url=None, required_scopes=["yolozu:invoke"])
        settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["vision.example"],
        )
        service_app._token_verifier = _StaticTokenVerifier(token)
        with tempfile.TemporaryDirectory() as td:
            configure_image_service(workspace=td)
            try:
                application = service_app.streamable_http_app()
                async with application.router.lifespan_context(application):
                    transport = httpx.ASGITransport(app=application)
                    async with httpx.AsyncClient(transport=transport) as unauthenticated:
                        denied = await unauthenticated.post("https://vision.example/mcp", json={})
                        self.assertEqual(denied.status_code, 401)
                    async with httpx.AsyncClient(
                        transport=transport, headers={"Authorization": f"Bearer {token}"},
                    ) as client:
                        async with streamable_http_client("https://vision.example/mcp", http_client=client) as (read, write, _):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                listed = await session.list_tools()
                                self.assertEqual({tool.name for tool in listed.tools}, {
                                    "image_service_capabilities", "put_image_asset",
                                    "submit_image_job", "get_image_job", "cancel_image_job",
                                })

                                async def call(name, arguments=None):
                                    response = await session.call_tool(name, arguments or {})
                                    self.assertFalse(response.isError)
                                    if response.structuredContent is not None:
                                        return response.structuredContent
                                    return json.loads(response.content[0].text)

                                capabilities = await call("image_service_capabilities")
                                self.assertEqual(capabilities["service"]["request_limits_per_60_seconds"]["upload"], 12)
                                uploaded = await call("put_image_asset", {
                                    "content_base64": base64.b64encode(_png_bytes()).decode(), "media_type": "image/png",
                                })
                                submitted = await call("submit_image_job", {
                                    "asset_id": uploaded["asset"]["asset_id"],
                                    "fixed_classes": ["cat"], "execute": True,
                                })
                                job_id = submitted["job"]["job_id"]
                                result = None
                                for _ in range(100):
                                    result = await call("get_image_job", {"job_id": job_id})
                                    if result["job"]["status"] not in {"queued", "running"}:
                                        break
                                    await asyncio.sleep(0.05)
                                self.assertEqual(result["job"]["result"]["outcome"], "abstained")
                                self.assertFalse(result["job"]["result"]["executed"])
                                cancelled = await call("cancel_image_job", {"job_id": job_id})
                                self.assertFalse(cancelled["job"]["cancelled"])
            finally:
                close_image_service()
                configure_image_service()
                (settings.auth, settings.stateless_http, settings.json_response, settings.transport_security, service_app._token_verifier) = saved

    async def test_non_ascii_bearer_input_is_rejected_without_server_error(self):
        from yolozu.integrations.mcp_server import _StaticTokenVerifier

        verifier = _StaticTokenVerifier("x" * 32)
        self.assertIsNone(await verifier.verify_token("日本語"))
        self.assertIsNotNone(await verifier.verify_token("x" * 32))


if __name__ == "__main__":
    unittest.main()
