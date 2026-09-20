#!/usr/bin/env python3
"""Transfer one local image through the five-tool stdio service, without a model fallback."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import warnings


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 64_000_000
MAX_DIMENSION = 16_384
TOOL_NAMES = frozenset({
    "image_service_capabilities", "put_image_asset", "submit_image_job",
    "get_image_job", "cancel_image_job",
})
SERVER_ARGS = [
    "-I", "-m", "yolozu.integrations.mcp_cli",
    "--transport", "stdio", "--surface", "image-service",
]


def read_image(path: Path) -> dict[str, str]:
    """Bound reads before decoding; do not follow a final symlink or open a FIFO."""
    from PIL import Image

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("image must not be a symlink")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_IMAGE_BYTES:
            raise ValueError("image must be a regular file of 1..8388608 bytes")
        data = stream.read(MAX_IMAGE_BYTES + 1)
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the byte limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(image.format)
            if media_type is None:
                raise ValueError("only PNG, JPEG, and WebP images are supported")
            width, height = image.size
            if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
                raise ValueError("image dimensions exceed the service limits")
            image.verify()
    return {"content_base64": base64.b64encode(data).decode("ascii"), "media_type": media_type}


async def call(session, name: str, arguments: dict | None = None) -> dict:
    reply = await session.call_tool(name, arguments or {})
    payload = reply.structuredContent
    if payload is None:
        payload = json.loads(reply.content[0].text)
    if reply.isError or not isinstance(payload, dict) or payload.get("ok") is not True:
        # Do not echo arbitrary server responses or user image bytes on failure.
        raise ValueError(f"{name} failed; no inference success is implied")
    return payload


async def request(session, args: argparse.Namespace, image: dict | None) -> dict:
    await session.initialize()
    listed = await session.list_tools()
    if {tool.name for tool in listed.tools} != TOOL_NAMES:
        raise ValueError("server does not expose exactly the bounded five-tool surface")
    capabilities = await call(session, "image_service_capabilities")
    if image is None:
        return capabilities
    uploaded = await call(session, "put_image_asset", image)
    submitted = await call(session, "submit_image_job", {
        "asset_id": uploaded["asset"]["asset_id"],
        "fixed_classes": args.classes,
        "execute": args.execute,
    })
    job_id = submitted["job"]["job_id"]
    deadline = asyncio.get_running_loop().time() + args.timeout
    while True:
        result = await call(session, "get_image_job", {"job_id": job_id})
        if result["job"]["status"] not in {"queued", "running"}:
            return result
        if asyncio.get_running_loop().time() >= deadline:
            cancellation = await call(session, "cancel_image_job", {"job_id": job_id})
            return {"ok": False, "outcome": "timed_out", "cancellation": cancellation}
        await asyncio.sleep(1)


async def run(args: argparse.Namespace, image: dict | None) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # A fresh session avoids writing into a caller's project or plugin cache.
    # No token or provider API key is needed for this local child process.
    with tempfile.TemporaryDirectory(prefix="yolozu-plugin-") as workspace:
        params = StdioServerParameters(
            command=sys.executable, args=SERVER_ARGS, cwd=workspace,
            env={},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                return await asyncio.wait_for(request(session, args, image), args.timeout + 15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capabilities", action="store_true", help="Inspect capabilities without uploading an image.")
    source.add_argument("--image", type=Path, help="User-selected local PNG/JPEG/WebP; at most 8 MiB.")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Target class; repeat for multiple labels. Required with --image.")
    parser.add_argument("--execute", action="store_true", help="Request actual processing after user authorization; still abstains without a qualified model.")
    parser.add_argument("--timeout", type=float, default=60, help="Job wait in seconds, 1..120 (default: 60); cancel on expiry.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 120:
        parser.error("--timeout must be finite and in 1..120")
    if args.image is not None and (not args.classes or len(args.classes) > 100 or any(not label.strip() or len(label) > 128 for label in args.classes)):
        parser.error("--image requires 1..100 non-empty --class labels of at most 128 characters")
    if args.capabilities and (args.classes or args.execute):
        parser.error("--class and --execute require --image")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        image = read_image(args.image) if args.image is not None else None
        result = asyncio.run(run(args, image))
    except KeyboardInterrupt:
        return 130
    except Exception:
        print(json.dumps({"ok": False, "error": "Local image request failed. Check the file, installed yolozu[mcp] runtime, and service diagnostics."}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    job = result.get("job", {})
    if result.get("ok") is not True or job.get("status") in {"failed", "cancelled"}:
        return 2
    return 0  # Transport success can still contain result.outcome == abstained.


if __name__ == "__main__":
    raise SystemExit(main())
