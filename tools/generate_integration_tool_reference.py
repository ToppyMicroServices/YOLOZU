from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yolozu.integrations.tool_reference import (
    build_tool_surface_reference,
    collect_surface_parity_errors,
    render_tool_surface_markdown,
)


def _write_if_changed(path: Path, content: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _check_exact(path: Path, expected: str) -> bool:
    if not path.exists():
        print(f"missing generated file: {path}")
        return False
    current = path.read_text(encoding="utf-8")
    if current == expected:
        return True
    print(f"outdated generated file: {path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MCP/Actions tool reference from integration backend sources."
    )
    parser.add_argument(
        "--json-out",
        default="docs/generated/mcp_actions_tool_reference.json",
        help="Output JSON reference path.",
    )
    parser.add_argument(
        "--md-out",
        default="docs/generated/mcp_actions_tool_reference.md",
        help="Output Markdown reference path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if checked-in generated files are outdated.",
    )
    args = parser.parse_args()

    reference = build_tool_surface_reference()
    errors = collect_surface_parity_errors(reference)
    if errors:
        for err in errors:
            print(err)
        return 1

    json_text = json.dumps(reference, indent=2, ensure_ascii=False) + "\n"
    md_text = render_tool_surface_markdown(reference)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)

    if args.check:
        ok = _check_exact(json_path, json_text) and _check_exact(md_path, md_text)
        if not ok:
            print(
                "Regenerate with: python3 tools/generate_integration_tool_reference.py "
                f"--json-out {json_path} --md-out {md_path}"
            )
            return 1
        print("integration tool reference is up to date")
        return 0

    changed_json = _write_if_changed(json_path, json_text)
    changed_md = _write_if_changed(md_path, md_text)
    status = "updated" if (changed_json or changed_md) else "unchanged"
    print(f"{status}: {json_path}")
    print(f"{status}: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
