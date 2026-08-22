#!/usr/bin/env python3
"""Release/tag operation helper for YOLOZU."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

if __package__:
    from .release_metadata import validate_release_metadata
else:
    from release_metadata import validate_release_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validate synchronized release metadata, then create/push a release tag "
            "and optionally create a GitHub Release (draft/publish)."
        )
    )
    p.add_argument("--version", default=None, help="Release version (default: parse from yolozu/__init__.py).")
    p.add_argument("--tag-prefix", default="v", help="Tag prefix (default: v).")
    p.add_argument("--title", default=None, help="Release title override (default: YOLOZU <version>).")
    p.add_argument("--notes-file", default=None, help="Release notes file path (default: --generate-notes).")
    p.add_argument(
        "--release-state",
        choices=("none", "draft", "publish"),
        default="none",
        help="GitHub release behavior: none/draft/publish (default: none).",
    )
    p.add_argument("--push-tag", action="store_true", help="Push created tag to origin.")
    p.add_argument("--run-checks", action="store_true", help="Run basic release quality checks before tagging.")
    p.add_argument("--allow-dirty", action="store_true", help="Allow dirty git working tree.")
    p.add_argument("--allow-non-main", action="store_true", help="Allow running outside main branch.")
    p.add_argument("--dry-run", action="store_true", help="Print plan without changing git/release state.")
    p.add_argument("--output", default="reports/release_tag_report.json", help="Output report path.")
    return p


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"cmd": cmd, "cwd": str(cwd), "ok": True, "returncode": 0, "stdout": "", "stderr": ""}
    if dry_run:
        entry["status"] = "dry_run"
        return entry
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    entry["returncode"] = int(proc.returncode)
    entry["stdout"] = str(proc.stdout or "")
    entry["stderr"] = str(proc.stderr or "")
    entry["ok"] = proc.returncode == 0
    return entry


def _parse_version_from_init() -> str:
    text = (REPO_ROOT / "yolozu" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise RuntimeError("could not parse __version__ from yolozu/__init__.py")
    return str(m.group(1))


def _git_stdout(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return str(proc.stdout or "")


def _check_tag_exists_local(tag: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _check_tag_exists_remote(tag: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    return bool(str(proc.stdout or "").strip())


def _quality_check_cmds() -> list[list[str]]:
    return [
        ["python3", "tools/validate_tool_manifest.py", "--manifest", "tools/manifest.json", "--require-declarative"],
        ["python3", "tools/generate_adaptive_vision_roadmap.py", "--check", "--json"],
        [
            "python3",
            "-m",
            "unittest",
            "tests.test_adaptive_vision_roadmap_generator",
            "tests.test_packaged_tools_manifest",
            "tests.test_manifest_docs_references",
        ],
        ["python3", "tools/check_mcp_settings.py", "--output", "reports/mcp_settings_check.release.json"],
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = Path(str(args.output))
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []
    steps: list[dict[str, Any]] = []

    try:
        parsed_version = _parse_version_from_init()
    except Exception as exc:
        parsed_version = ""
        errors.append(str(exc))

    version = str(args.version).strip() if args.version else parsed_version
    if not version:
        errors.append("release version is empty")
    if parsed_version and version and parsed_version != version:
        errors.append(
            f"version mismatch: yolozu/__init__.py has {parsed_version}, but --version requested {version}"
        )
    tag = f"{str(args.tag_prefix)}{version}"
    metadata_validation = validate_release_metadata(
        REPO_ROOT,
        expected_version=version or None,
        expected_tag=tag if version else None,
        tag_prefix=str(args.tag_prefix),
    )
    errors.extend(str(error) for error in metadata_validation.get("errors") or [])

    try:
        branch = _git_stdout("rev-parse", "--abbrev-ref", "HEAD").strip()
    except Exception as exc:
        branch = ""
        errors.append(str(exc))
    if branch and branch != "main" and not bool(args.allow_non_main):
        errors.append(f"current branch is '{branch}' (expected 'main'); use --allow-non-main to bypass")

    try:
        dirty = bool(_git_stdout("status", "--porcelain").strip())
    except Exception as exc:
        dirty = False
        errors.append(str(exc))
    if dirty and not bool(args.allow_dirty):
        errors.append("git working tree is dirty; commit/stash or use --allow-dirty")

    local_tag_exists = _check_tag_exists_local(tag) if tag else False
    remote_tag_exists = _check_tag_exists_remote(tag) if tag else False
    if local_tag_exists:
        errors.append(f"local tag already exists: {tag}")
    if remote_tag_exists:
        errors.append(f"remote tag already exists: {tag}")

    if errors:
        report = {
            "task": "release_tag",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ok": False,
            "version": version,
            "parsed_version": parsed_version,
            "tag": tag,
            "branch": branch,
            "dirty": dirty,
            "dry_run": bool(args.dry_run),
            "metadata_validation": metadata_validation,
            "warnings": warnings,
            "errors": errors,
            "steps": steps,
        }
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(str(out))
        return 1

    if bool(args.run_checks):
        for cmd in _quality_check_cmds():
            step = _run(cmd, cwd=REPO_ROOT, dry_run=bool(args.dry_run))
            step["type"] = "quality_check"
            steps.append(step)
            if not bool(step.get("ok")):
                errors.append(f"quality check failed: {' '.join(cmd)}")
                break

    if not errors:
        create_tag = _run(["git", "tag", "-a", tag, "-m", f"YOLOZU {version}"], cwd=REPO_ROOT, dry_run=bool(args.dry_run))
        create_tag["type"] = "create_tag"
        steps.append(create_tag)
        if not bool(create_tag.get("ok")):
            errors.append("failed to create tag")

    if not errors and bool(args.push_tag):
        push = _run(["git", "push", "origin", tag], cwd=REPO_ROOT, dry_run=bool(args.dry_run))
        push["type"] = "push_tag"
        steps.append(push)
        if not bool(push.get("ok")):
            errors.append("failed to push tag")

    if not errors and str(args.release_state) != "none":
        title = str(args.title).strip() if args.title else f"YOLOZU {version}"
        release_cmd = ["gh", "release", "create", tag, "--title", title]
        notes_file = str(args.notes_file).strip() if args.notes_file else ""
        if notes_file:
            release_cmd.extend(["--notes-file", notes_file])
        else:
            release_cmd.append("--generate-notes")
        if str(args.release_state) == "draft":
            release_cmd.append("--draft")
        release_step = _run(release_cmd, cwd=REPO_ROOT, dry_run=bool(args.dry_run))
        release_step["type"] = f"create_release_{args.release_state}"
        steps.append(release_step)
        if not bool(release_step.get("ok")):
            errors.append("failed to create GitHub release")

    if str(args.release_state) != "publish":
        warnings.append(
            "PyPI publish is triggered by release:published workflow; use --release-state publish (or publish the draft release) when ready."
        )

    report = {
        "task": "release_tag",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": len(errors) == 0,
        "version": version,
        "parsed_version": parsed_version,
        "tag": tag,
        "branch": branch,
        "dirty": dirty,
        "dry_run": bool(args.dry_run),
        "metadata_validation": metadata_validation,
        "release_state": str(args.release_state),
        "push_tag": bool(args.push_tag),
        "run_checks": bool(args.run_checks),
        "warnings": warnings,
        "errors": errors,
        "steps": steps,
        "next_steps": [
            "If release_state=draft, publish the GitHub release to trigger .github/workflows/publish.yml.",
            "If manual changed, run .github/workflows/manual_doi.yml (release trigger or workflow_dispatch).",
            "Validate final status with docs/release_reliability_checklist.md.",
        ],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
