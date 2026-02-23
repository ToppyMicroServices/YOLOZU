#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]


_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9\-]*")


@dataclass(frozen=True)
class ToolHelpScan:
    tool_id: str
    entrypoint: str
    extracted_flags: set[str]
    declared_flags: set[str]
    missing: list[str]
    error: str | None


def _resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_declared_flags(tool: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    inputs = tool.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if not isinstance(item, dict):
                continue
            flag = item.get("flag")
            if isinstance(flag, str) and flag.startswith("--"):
                flags.add(flag)
    return flags


def _extract_long_flags(help_text: str, *, include_help: bool) -> set[str]:
    flags: set[str] = set()
    for line in help_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("-") or "--" not in stripped:
            continue
        head = re.split(r"\s{2,}", stripped, maxsplit=1)[0]
        flags |= {f for f in _FLAG_RE.findall(head) if not f.endswith("-")}
    if not include_help:
        flags.discard("--help")
    return flags


def _run_help(entrypoint: str, *, timeout_s: float) -> tuple[str, str | None]:
    ep = _resolve_repo_path(entrypoint)
    if not ep.exists():
        return "", f"entrypoint not found: {entrypoint}"

    try:
        proc = subprocess.run(
            [sys.executable, str(ep), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout_s:.1f}s"
    except Exception as exc:
        return "", f"failed to run --help: {exc}"

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0 and not out.strip():
        return out, f"--help exited {proc.returncode}"
    return out, None


def scan_manifest(manifest_path: Path, *, timeout_s: float, include_help: bool) -> list[ToolHelpScan]:
    manifest = _load_json(manifest_path)
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise SystemExit("manifest.tools must be a list")

    results: list[ToolHelpScan] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = tool.get("id")
        runner = tool.get("runner")
        entrypoint = tool.get("entrypoint")

        if not isinstance(tool_id, str) or not tool_id:
            continue
        if runner != "python3":
            continue
        if not isinstance(entrypoint, str) or not entrypoint:
            continue

        declared_flags = _collect_declared_flags(tool)
        help_text, error = _run_help(entrypoint, timeout_s=timeout_s)
        extracted_flags = _extract_long_flags(help_text, include_help=include_help) if not error else set()

        missing = sorted(extracted_flags - declared_flags)
        results.append(
            ToolHelpScan(
                tool_id=tool_id,
                entrypoint=entrypoint,
                extracted_flags=extracted_flags,
                declared_flags=declared_flags,
                missing=missing,
                error=error,
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit tools/manifest.json inputs vs each tool's --help flags")
    parser.add_argument("--manifest", default=str(_REPO_ROOT / "tools" / "manifest.json"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--include-help", action="store_true", help="Include --help in extracted flags")
    parser.add_argument("--show-ok", action="store_true", help="Also print tools with no missing flags")
    parser.add_argument("--max-flags", type=int, default=60, help="Max missing flags to print per tool")
    args = parser.parse_args()

    manifest_path = _resolve_repo_path(args.manifest)
    results = scan_manifest(manifest_path, timeout_s=args.timeout, include_help=args.include_help)

    errored = [r for r in results if r.error]
    missing = [r for r in results if (not r.error) and r.missing]

    print(f"scanned_tools {len(results)}")
    print(f"error_tools {len(errored)}")
    print(f"missing_tools {len(missing)}")

    for r in sorted(errored, key=lambda x: x.tool_id):
        print(f"ERROR {r.tool_id}: {r.error}")

    for r in sorted(missing, key=lambda x: (-len(x.missing), x.tool_id)):
        print(f"MISSING {r.tool_id} ({len(r.missing)}):")
        flags = r.missing[: max(args.max_flags, 0)]
        for f in flags:
            print(f"  {f}")
        if args.max_flags >= 0 and len(r.missing) > args.max_flags:
            print(f"  ... (+{len(r.missing) - args.max_flags} more)")

    if args.show_ok:
        ok = [r for r in results if (not r.error) and (not r.missing)]
        for r in sorted(ok, key=lambda x: x.tool_id):
            print(f"OK {r.tool_id}")

    return 0 if (not errored and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
