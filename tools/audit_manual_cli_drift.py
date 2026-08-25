#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL = repo_root / "manual" / "chapters" / "04_cli_reference.tex"
DEFAULT_ALLOWLIST = repo_root / "docs" / "manual_cli_drift_allowlist.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit manual chapter 04 against the canonical yolozu CLI help surface. "
            "The audit extracts documented top-level `yolozu <command>` references, "
            "checks command availability/help, and optionally checks the legacy wrapper."
        )
    )
    p.add_argument("--manual", default=str(DEFAULT_MANUAL), help="Manual chapter to audit.")
    p.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST), help="JSON allowlist for intentional non-CLI tokens.")
    p.add_argument("--python", default=sys.executable, help="Python executable used for CLI probes.")
    p.add_argument("--skip-wrapper", action="store_true", help="Skip legacy tools/yolozu.py passthrough help checks.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return p.parse_args(argv)


def _load_allowlist(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ignored_manual_tokens": [], "repo_only_prefixes": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"allowlist must be a JSON object: {path}")
    return payload


def _latex_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\\_", "_")
    return text


def _extract_manual_yolozu_commands(path: Path, *, allowlist: dict[str, Any]) -> list[str]:
    text = _latex_text(path)
    ignored = set(str(x) for x in allowlist.get("ignored_manual_tokens", []))
    commands: set[str] = set()
    for match in re.finditer(r"\\cmd\{yolozu\s+([^}\s]+)", text):
        token = match.group(1).strip()
        if not token or token.startswith("-") or token in ignored:
            continue
        if token.startswith("<") and token.endswith(">"):
            continue
        commands.add(token)
    return sorted(commands)


def _run(cmd: list[str], *, timeout: float = 12.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )


def _canonical_commands(python: str) -> tuple[set[str], str]:
    proc = _run([python, "-m", "yolozu", "--help"])
    if proc.returncode != 0:
        raise SystemExit(f"python3 -m yolozu --help failed:\n{proc.stdout}\n{proc.stderr}")
    commands: set[str] = set()
    for line in proc.stdout.splitlines():
        match = re.match(r"\s{2,}([a-z][a-z0-9-]+)(?:\s+\(([^)]+)\))?\s{2,}", line)
        if match is None:
            match = re.fullmatch(r"\s{2,}([a-z][a-z0-9-]+)\s*", line)
        if not match:
            continue
        commands.add(match.group(1))
        aliases = match.group(2) if (match.lastindex or 0) >= 2 else None
        if aliases:
            for alias in re.split(r"[, ]+", aliases):
                alias = alias.strip()
                if alias:
                    commands.add(alias)
    return commands, proc.stdout


def _check_command_help(python: str, command: str) -> dict[str, Any]:
    proc = _run([python, "-m", "yolozu", command, "--help"])
    return {
        "command": command,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-1000:],
    }


def _check_wrapper_help(python: str, command: str) -> dict[str, Any]:
    proc = _run([python, "tools/yolozu.py", command, "--help"])
    return {
        "command": command,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-1000:],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manual = Path(args.manual)
    if not manual.is_absolute():
        manual = (repo_root / manual).resolve()
    allowlist_path = Path(args.allowlist)
    if not allowlist_path.is_absolute():
        allowlist_path = (repo_root / allowlist_path).resolve()
    allowlist = _load_allowlist(allowlist_path)

    documented = _extract_manual_yolozu_commands(manual, allowlist=allowlist)
    canonical, _help = _canonical_commands(str(args.python))
    missing = [cmd for cmd in documented if cmd not in canonical]
    help_checks = [_check_command_help(str(args.python), cmd) for cmd in documented if cmd in canonical]
    wrapper_checks = [] if args.skip_wrapper else [_check_wrapper_help(str(args.python), cmd) for cmd in documented if cmd in canonical]
    failing_help = [item for item in help_checks if not item["ok"]]
    failing_wrapper = [item for item in wrapper_checks if not item["ok"]]

    payload = {
        "schema_version": 1,
        "kind": "manual_cli_drift_audit",
        "manual": str(manual),
        "allowlist": str(allowlist_path),
        "documented_commands": documented,
        "canonical_commands": sorted(canonical),
        "missing_from_cli": missing,
        "failing_help": failing_help,
        "failing_wrapper_help": failing_wrapper,
        "ok": not missing and not failing_help and not failing_wrapper,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"manual commands: {len(documented)}")
        print(f"canonical commands: {len(canonical)}")
        if missing:
            print("missing from CLI: " + ", ".join(missing))
        if failing_help:
            print("failing command help: " + ", ".join(item["command"] for item in failing_help))
        if failing_wrapper:
            print("failing wrapper help: " + ", ".join(item["command"] for item in failing_wrapper))
        if payload["ok"]:
            print("manual/CLI drift audit: ok")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
