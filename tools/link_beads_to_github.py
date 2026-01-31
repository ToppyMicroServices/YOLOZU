#!/usr/bin/env python3
"""Create/link GitHub Issues for Beads issues via --external-ref.

- Source of truth: .beads/issues.jsonl (Beads)
- Target: GitHub Issues in the current repo (via gh CLI)

This script is intentionally conservative:
- If a Beads issue already has external_ref, it is skipped.
- If an exact-title GitHub issue exists, it links to that.
- Otherwise it creates a new GitHub issue and links to it.

Usage:
  python3 tools/link_beads_to_github.py --dry-run
  python3 tools/link_beads_to_github.py

Optional:
  --repo owner/name     # override target repo
  --only YOLOZU-xxm.2   # only link specific bead IDs (repeatable)
    --sync-close          # if Beads issue is closed/done, close linked GitHub issue
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


def run(cmd: list[str], *, input_text: str | None = None) -> str:
    proc = subprocess.run(
        cmd,
        input=None if input_text is None else input_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return proc.stdout.decode("utf-8", errors="replace")


def detect_repo() -> str:
    # Prefer gh's view of the current repo (respects forks/remotes)
    out = run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]).strip()
    if not out:
        raise RuntimeError("Failed to detect repo via gh repo view")
    return out


def load_beads_jsonl(path: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        issues.append(json.loads(line))
    return issues


def find_exact_title_match(repo: str, title: str) -> int | None:
    # Fetch a bounded list and exact-match client-side.
    # Note: gh's search is substring-based.
    out = run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--search",
            f"{title} in:title",
            "--json",
            "number,title",
            "--limit",
            "50",
        ]
    )
    items = json.loads(out)
    for item in items:
        if item.get("title") == title:
            return int(item["number"])
    return None


def gh_create_issue(repo: str, title: str, body: str) -> int:
    # Use API to get structured JSON response (avoids parsing URL output).
    out = run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/issues",
            "-f",
            f"title={title}",
            "-f",
            f"body={body}",
        ]
    )
    data = json.loads(out)
    return int(data["number"])


def gh_get_state(repo: str, number: int) -> str:
    return run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{number}",
            "--jq",
            ".state",
        ]
    ).strip()


def gh_close_issue(repo: str, number: int, *, dry_run: bool) -> None:
    cmd = [
        "gh",
        "api",
        "--method",
        "PATCH",
        f"repos/{repo}/issues/{number}",
        "-f",
        "state=closed",
    ]
    if dry_run:
        print("DRY:", " ".join(cmd))
        return
    run(cmd)


def bd_link_external_ref(bead_id: str, number: int, *, dry_run: bool) -> None:
    ext = f"gh-{number}"
    note = f"Linked GitHub issue #{number} ({ext})"
    cmd = ["bd", "update", bead_id, "--external-ref", ext, "--notes", note, "--quiet"]
    if dry_run:
        print("DRY:", " ".join(cmd))
        return
    run(cmd)


def format_body(bead: dict[str, Any]) -> str:
    # Keep it readable on GitHub; include bead ID for round-tripping.
    parts = [
        f"Beads ID: {bead.get('id')}",
        "",
    ]
    desc = bead.get("description") or ""
    if desc:
        parts.append(desc)
        parts.append("")
    notes = bead.get("notes") or ""
    if notes:
        parts.append("---")
        parts.append("Beads Notes")
        parts.append(notes)
        parts.append("")
    parts.append("---")
    parts.append("Managed in-repo via Beads (.beads/issues.jsonl).")
    return "\n".join(parts).strip() + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--repo", default=None, help="GitHub repo as owner/name")
    p.add_argument("--only", action="append", default=[], help="Only link specified Beads IDs (repeatable)")
    p.add_argument(
        "--sync-close",
        action="store_true",
        help="If Beads issue is closed/done and has external_ref gh-<n>, close the GitHub issue (one-way).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def parse_gh_external_ref(value: str | None) -> int | None:
    if not value:
        return None
    m = re.fullmatch(r"gh-(\d+)", str(value).strip())
    if not m:
        return None
    return int(m.group(1))


def main() -> int:
    args = parse_args()
    repo = args.repo or detect_repo()

    beads_path = Path(".beads/issues.jsonl")
    if not beads_path.exists():
        raise RuntimeError(".beads/issues.jsonl not found; run bd init/bd sync first")

    allow = set(args.only)
    issues = load_beads_jsonl(beads_path)
    targets = [i for i in issues if (not allow or i.get("id") in allow)]

    linked = 0
    closed = 0

    for bead in targets:
        bead_id = str(bead.get("id"))
        title = str(bead.get("title") or "").strip()
        if not bead_id or not title:
            continue

        number = parse_gh_external_ref(bead.get("external_ref"))

        # Linking mode: create/link external_ref if missing.
        if number is None:
            existing = find_exact_title_match(repo, title)
            if existing is not None:
                number = existing
                action = "link"
            else:
                number = gh_create_issue(repo, title, format_body(bead))
                action = "create+link"

            print(f"{action.upper()} {bead_id} -> #{number}")
            bd_link_external_ref(bead_id, number, dry_run=args.dry_run)
            linked += 1

        # Close-sync mode: if Beads is closed/done and external_ref exists, close GH issue.
        if args.sync_close and number is not None:
            status = str(bead.get("status") or "").lower()
            if status in {"closed", "done"}:
                state = gh_get_state(repo, number)
                if state != "closed":
                    print(f"CLOSE {bead_id} -> #{number}")
                    gh_close_issue(repo, number, dry_run=args.dry_run)
                    closed += 1

    print(f"Done. Linked {linked} Beads issues.")
    if args.sync_close:
        print(f"Done. Closed {closed} GitHub issues.")
    if args.dry_run:
        print("(dry-run: no changes written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
