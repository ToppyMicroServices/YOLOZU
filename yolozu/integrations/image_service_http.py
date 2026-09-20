"""Bound HTTP upload bytes and duration before MCP JSON parsing."""

from __future__ import annotations

import asyncio
import time

MAX_HTTP_BODY_BYTES = 16 * 1024 * 1024
HTTP_UPLOAD_TIMEOUT_SECONDS = 30


class ImageServiceHTTPBounds:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        async def reject(status: int, code: str) -> None:
            body = ('{"error":"' + code + '"}').encode("ascii")
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii")),
            ]})
            await send({"type": "http.response.body", "body": body})

        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    await reject(400, "invalid_content_length")
                    return
                if declared < 0:
                    await reject(400, "invalid_content_length")
                    return
                if declared > MAX_HTTP_BODY_BYTES:
                    await reject(413, "request_too_large")
                    return
        body = bytearray()
        deadline = time.monotonic() + HTTP_UPLOAD_TIMEOUT_SECONDS
        while True:
            try:
                message = await asyncio.wait_for(receive(), timeout=max(0, deadline - time.monotonic()))
            except TimeoutError:
                await reject(408, "upload_timeout")
                return
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_HTTP_BODY_BYTES:
                await reject(413, "request_too_large")
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)
