#!/usr/bin/env python3
"""Validate MCP integration settings and generated references."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yolozu.integrations.ai_surface import (  # noqa: E402
    load_tool_manifest,
    supported_mcp_tool_ids as _supported_mcp_tool_ids,
)
from yolozu.integrations.tool_reference import (  # noqa: E402
    build_tool_surface_reference,
    collect_surface_parity_errors,
    render_tool_surface_markdown,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check MCP settings, manifest alignment, and generated reference freshness.")
    p.add_argument("--manifest", default="tools/manifest.json", help="Tool manifest path (default: tools/manifest.json).")
    p.add_argument(
        "--json-ref",
        default="docs/generated/mcp_actions_tool_reference.json",
        help="Generated MCP/Actions JSON reference path.",
    )
    p.add_argument(
        "--md-ref",
        default="docs/generated/mcp_actions_tool_reference.md",
        help="Generated MCP/Actions Markdown reference path.",
    )
    p.add_argument("--output", default="reports/mcp_settings_check.json", help="Output report path.")
    p.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors.")
    return p


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = Path(str(args.manifest))
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    json_ref = Path(str(args.json_ref))
    if not json_ref.is_absolute():
        json_ref = (Path.cwd() / json_ref).resolve()
    md_ref = Path(str(args.md_ref))
    if not md_ref.is_absolute():
        md_ref = (Path.cwd() / md_ref).resolve()
    out = Path(str(args.output))
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []

    manifest_doc = load_tool_manifest(str(manifest_path))
    manifest_tool_ids = {str(t.get("id")) for t in list(manifest_doc.get("tools") or []) if isinstance(t, dict) and t.get("id")}

    supported_ids = [str(x) for x in _supported_mcp_tool_ids()]
    if not supported_ids:
        errors.append("supported_mcp_tool_ids() returned empty list")

    generated_canonicals = sorted(
        {
            str(row.get("canonical_name"))
            for row in list((build_tool_surface_reference().get("tools") or []))
            if isinstance(row, dict) and row.get("canonical_name")
        }
    )
    built_in_non_generated = {"generate_config", "review_config"}
    missing_from_generated = sorted(
        [tid for tid in supported_ids if tid not in generated_canonicals and tid not in built_in_non_generated]
    )
    if missing_from_generated:
        errors.append("supported MCP tool ids missing from generated integration reference: " + ", ".join(missing_from_generated))
    missing_built_ins = sorted([tid for tid in supported_ids if tid in built_in_non_generated and tid not in generated_canonicals])
    if missing_built_ins:
        warnings.append(
            "MCP-lite deterministic helpers are intentionally outside generated MCP/Actions parity surface: "
            + ", ".join(missing_built_ins)
        )

    # MCP-lite includes "generate_config/review_config" as deterministic helpers
    # that are intentionally not direct manifest tool ids.
    built_in_non_manifest = {"generate_config", "review_config"}
    manifest_aliases = {"doctor": "yolozu", "validate_predictions": "validate_predictions"}
    missing_supported = sorted(
        [
            tid
            for tid in supported_ids
            if tid not in built_in_non_manifest
            and tid not in manifest_tool_ids
            and manifest_aliases.get(tid, "") not in manifest_tool_ids
        ]
    )
    if missing_supported:
        errors.append(f"supported MCP tool ids missing in manifest: {', '.join(missing_supported)}")
    found_supported_manifest_entries = sorted(
        {
            manifest_aliases.get(tid, tid)
            for tid in supported_ids
            if tid not in built_in_non_manifest and (tid in manifest_tool_ids or manifest_aliases.get(tid, "") in manifest_tool_ids)
        }
    )

    reference = build_tool_surface_reference()
    parity_errors = list(collect_surface_parity_errors(reference))
    if parity_errors:
        errors.extend([f"tool surface parity: {msg}" for msg in parity_errors])

    expected_json = json.dumps(reference, indent=2, ensure_ascii=False) + "\n"
    expected_md = render_tool_surface_markdown(reference)
    current_json = _read_text(json_ref)
    current_md = _read_text(md_ref)
    if current_json is None:
        errors.append(f"missing generated JSON reference: {json_ref}")
    elif current_json != expected_json:
        errors.append(f"outdated generated JSON reference: {json_ref}")
    if current_md is None:
        errors.append(f"missing generated Markdown reference: {md_ref}")
    elif current_md != expected_md:
        errors.append(f"outdated generated Markdown reference: {md_ref}")

    if not manifest_path.exists():
        errors.append(f"manifest not found: {manifest_path}")
    elif manifest_path.resolve() != (REPO_ROOT / "tools" / "manifest.json").resolve():
        warnings.append("non-default --manifest path was used; verify packaged manifest sync separately")

    ok = not errors and (not args.strict or not warnings)
    report: dict[str, Any] = {
        "task": "mcp_settings_check",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok),
        "strict": bool(args.strict),
        "manifest": str(manifest_path),
        "json_ref": str(json_ref),
        "md_ref": str(md_ref),
        "supported_mcp_tool_ids": supported_ids,
        "generated_canonical_names": generated_canonicals,
        "found_supported_tools": found_supported_manifest_entries,
        "checks": {
            "manifest_contains_supported_tools": len(missing_supported) == 0,
            "supported_tools_in_generated_reference": len(missing_from_generated) == 0,
            "tool_surface_parity_errors": len(parity_errors),
            "generated_json_uptodate": current_json == expected_json if current_json is not None else False,
            "generated_md_uptodate": current_md == expected_md if current_md is not None else False,
        },
        "warnings": warnings,
        "errors": errors,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
