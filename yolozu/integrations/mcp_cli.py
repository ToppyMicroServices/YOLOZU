from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from urllib.parse import urlsplit

from .ai_surface import (
    ai_surface_sets,
    discover_manifest_tools,
    generate_config,
    review_config,
)
from .manifest_resources import resolve_workspace_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolozu-mcp",
        description=(
            "Run the YOLOZU MCP server over stdio or Streamable HTTP, "
            "or inspect its AI surface."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--surface",
        choices=("full", "image-service"),
        default="full",
        help=(
            "Tool surface. image-service exposes only the bounded CNN service "
            "tools (default: full)."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Streamable HTTP bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Streamable HTTP bind port (default: 8000).",
    )
    parser.add_argument(
        "--http-path",
        default="/mcp",
        help="Streamable HTTP MCP path (default: /mcp).",
    )
    parser.add_argument(
        "--public-url",
        default=None,
        help=(
            "External HTTPS MCP URL when running behind a TLS proxy. Required "
            "for a non-loopback bind."
        ),
    )
    parser.add_argument(
        "--auth-token-env",
        default="YOLOZU_MCP_AUTH_TOKEN",
        help=(
            "Environment variable containing a bearer token. The token is "
            "never accepted as a command-line value."
        ),
    )
    parser.add_argument(
        "--tenant-id",
        default="local",
        help="Credential-bound lowercase tenant identifier (default: local).",
    )
    parser.add_argument(
        "--retention-seconds",
        type=int,
        default=86_400,
        help="Private asset/output retention in 300..604800 seconds.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional tool manifest override (default: manifest packaged with yolozu).",
    )
    parser.add_argument(
        "--print-tools",
        action="store_true",
        help="Print manifest-backed tool metadata as JSON and exit.",
    )
    parser.add_argument(
        "--guaranteed",
        action="store_true",
        help="With --print-tools, keep tools in the guaranteed AI-safe MCP set.",
    )
    parser.add_argument(
        "--supported",
        action="store_true",
        help=(
            "Compatibility flag: with --print-tools, keep tools registered on "
            "the live MCP surface; this is not an execution guarantee."
        ),
    )
    parser.add_argument(
        "--maturity",
        action="append",
        default=None,
        help="With --print-tools, keep one maturity (repeatable).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="With --print-tools, require a manifest tag (repeatable).",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help=(
            "With --print-tools, emit compact JSON with sorted selected ids "
            "instead of full tool records."
        ),
    )
    parser.add_argument(
        "--sample-generate-config",
        action="store_true",
        help="Emit sample generate_config JSON and exit.",
    )
    parser.add_argument(
        "--sample-review-config",
        default=None,
        help="Review the given config JSON path and exit.",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for safety review checks.",
    )
    return parser.parse_args(argv)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _server_options(args: argparse.Namespace) -> tuple[dict[str, object], str | None]:
    if not 1 <= args.port <= 65_535:
        raise ValueError("--port must be in 1..65535")
    if (
        not isinstance(args.http_path, str)
        or not args.http_path.startswith("/")
        or ".." in args.http_path.split("/")
        or any(ord(character) < 32 or ord(character) == 127 for character in args.http_path)
    ):
        raise ValueError("--http-path must be an absolute path without traversal")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.tenant_id) is None:
        raise ValueError("--tenant-id must be a bounded lowercase identifier")
    if not 300 <= args.retention_seconds <= 604_800:
        raise ValueError("--retention-seconds must be in 300..604800")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", args.auth_token_env) is None:
        raise ValueError("--auth-token-env must name one environment variable")

    auth_token = os.environ.get(args.auth_token_env)
    if auth_token is not None and not 32 <= len(auth_token.encode("utf-8")) <= 4096:
        raise ValueError("the bearer token must contain 32..4096 UTF-8 bytes")
    if auth_token is not None and any(
        ord(character) < 32 or ord(character) == 127 for character in auth_token
    ):
        raise ValueError("the bearer token must not contain control characters")
    public_url = args.public_url
    if public_url is not None:
        parsed = urlsplit(public_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "--public-url must be an HTTPS URL without credentials, query, or fragment"
            )
        if parsed.path != args.http_path:
            raise ValueError("--public-url path must match --http-path")

    loopback = _is_loopback_host(args.host)
    externally_reachable = not loopback or public_url is not None
    if args.transport == "streamable-http" and externally_reachable:
        if args.surface != "image-service":
            raise ValueError("external HTTP may expose only --surface image-service")
        if public_url is None:
            raise ValueError("non-loopback HTTP requires --public-url behind a TLS proxy")
        if auth_token is None:
            raise ValueError(
                f"external HTTP requires a bearer token in {args.auth_token_env}"
            )
    return (
        {
            "transport": args.transport,
            "surface": args.surface,
            "host": args.host,
            "port": args.port,
            "streamable_http_path": args.http_path,
            "tenant_id": args.tenant_id,
            "retention_seconds": args.retention_seconds,
            "auth_token": auth_token,
            "public_url": public_url,
        },
        auth_token,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.print_tools:
        try:
            surfaces = ai_surface_sets(args.manifest)
            discovery = discover_manifest_tools(
                manifest_path=args.manifest,
                guaranteed=bool(args.guaranteed),
                supported=bool(args.supported),
                maturity=args.maturity,
                tag=args.tag,
                ids_only=bool(args.ids_only),
            )
            tools = discovery["tools"]
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ok": False,
                        "error": {
                            "code": "invalid_manifest",
                            "message": str(exc),
                        },
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        payload = {
            "schema_version": 1,
            "ok": True,
            "supported_mcp_tools": list(
                surfaces["guaranteed_ai_safe"]["tool_ids"]
            ),
            "supported_mcp_tools_semantics": (
                "compatibility view of guaranteed_ai_safe tool ids"
            ),
            "filters": {
                "guaranteed": bool(args.guaranteed),
                "supported": bool(args.supported),
                "maturity": list(args.maturity or []),
                "tags": list(args.tag or []),
                "ids_only": bool(args.ids_only),
            },
            "manifest_tools": tools,
        }
        if args.ids_only:
            payload["selected_tool_ids"] = list(tools)
            payload["surface_counts"] = {
                name: len(surface["tool_ids"])
                for name, surface in surfaces.items()
            }
        else:
            payload["guaranteed_mcp_tools"] = list(
                surfaces["guaranteed_ai_safe"]["tool_ids"]
            )
            payload["live_mcp_tools"] = list(
                surfaces["mcp_live"]["tool_ids"]
            )
            payload["surfaces"] = surfaces
        if args.maturity or args.tag:
            payload["filter_diagnostics"] = discovery[
                "filter_diagnostics"
            ]
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=None if args.ids_only else 2,
            )
        )
        return 0
    if args.sample_generate_config:
        print(json.dumps(generate_config(), ensure_ascii=False, indent=2))
        return 0
    if args.sample_review_config:
        try:
            path = resolve_workspace_path(args.sample_review_config)
            safe_workspace_root = resolve_workspace_path(args.workspace_root)
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ok": False,
                        "error": {
                            "code": "unsafe_or_invalid_config",
                            "message": str(exc),
                        },
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        review = review_config(
            doc,
            workspace_root=str(safe_workspace_root),
        )
        print(json.dumps(review, ensure_ascii=False, indent=2))
        return 0 if bool(review.get("ok")) else 1

    try:
        from .mcp_server import run_server
    except ModuleNotFoundError as exc:
        if exc.name == "mcp" or str(exc.name or "").startswith("mcp."):
            print(
                "error: MCP support is not installed; "
                "run `python3 -m pip install 'yolozu[mcp]'`",
                file=sys.stderr,
            )
            return 2
        raise

    try:
        server_options, _auth_token = _server_options(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_server(**server_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
