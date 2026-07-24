from __future__ import annotations

import argparse
import json
import sys

from .ai_surface import (
    ai_surface_sets,
    generate_config,
    list_manifest_tools,
    review_config,
)
from .manifest_resources import resolve_workspace_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolozu-mcp",
        description="Run the YOLOZU MCP server (stdio) or inspect its AI surface.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="MCP transport. stdio is supported.",
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
        help="With --print-tools, keep tools registered on the supported live MCP surface.",
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.print_tools:
        try:
            surfaces = ai_surface_sets(args.manifest)
            tools = list_manifest_tools(
                manifest_path=args.manifest,
                guaranteed=bool(args.guaranteed),
                supported=bool(args.supported),
                maturity=args.maturity,
                tag=args.tag,
                ids_only=bool(args.ids_only),
            )
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
        from .mcp_server import main as run_mcp_stdio
    except ModuleNotFoundError as exc:
        if exc.name == "mcp" or str(exc.name or "").startswith("mcp."):
            print(
                "error: MCP support is not installed; "
                "run `python3 -m pip install 'yolozu[mcp]'`",
                file=sys.stderr,
            )
            return 2
        raise

    run_mcp_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
