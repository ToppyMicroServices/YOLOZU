#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.integrations.ai_surface import generate_config, list_manifest_tools, review_config, supported_mcp_tool_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_mcp_server.py",
        description="Run YOLOZU MCP server (stdio) and inspect AI-first surface.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="MCP transport. stdio is supported in this wrapper.",
    )
    parser.add_argument("--manifest", default="tools/manifest.json", help="Tool manifest path for surface introspection.")
    parser.add_argument("--print-tools", action="store_true", help="Print manifest-backed tool metadata as JSON and exit.")
    parser.add_argument("--sample-generate-config", action="store_true", help="Emit sample generate_config JSON and exit.")
    parser.add_argument("--sample-review-config", default=None, help="Review the given config JSON path and exit.")
    parser.add_argument("--workspace-root", default=".", help="Workspace root used for safety review checks.")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.print_tools:
        payload = {
            "schema_version": 1,
            "supported_mcp_tools": supported_mcp_tool_ids(),
            "manifest_tools": list_manifest_tools(manifest_path=args.manifest, only_supported=False),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.sample_generate_config:
        print(json.dumps(generate_config(), ensure_ascii=False, indent=2))
        return 0
    if args.sample_review_config:
        p = Path(args.sample_review_config).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        doc = json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(review_config(doc, workspace_root=str(args.workspace_root)), ensure_ascii=False, indent=2))
        return 0

    from yolozu.integrations.mcp_server import main as run_mcp_stdio

    run_mcp_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
