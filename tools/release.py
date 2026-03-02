#!/usr/bin/env python3
"""Single-command release automation for YOLOZU.

Default behavior (`python3 tools/release.py`):
1) Read current package version from `yolozu/__init__.py`.
2) Classify change scale (small/medium/large) from git diff stats since latest semver tag.
3) Bump semantic version automatically.
4) Run release quality checks.
5) Update package version, commit, create/push git tag.
6) Create published GitHub release (which triggers PyPI workflow).
7) Trigger manual DOI workflow for Zenodo manual record update.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "yolozu" / "__init__.py"
PYTHON = sys.executable  # Use the same interpreter as release.sh selected.


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run full release flow with zero required options: auto-bump version, tag, "
            "GitHub Release publish, and Zenodo manual DOI workflow dispatch."
        )
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan/report only; do not mutate git/GitHub.")
    p.add_argument(
        "--output",
        default="reports/release_report.json",
        help="Output report path (default: reports/release_report.json).",
    )
    p.add_argument("--allow-dirty", action="store_true", help="Allow dirty git working tree.")
    p.add_argument("--allow-non-main", action="store_true", help="Allow running outside main branch.")
    p.add_argument("--skip-checks", action="store_true", help="Skip local quality checks.")
    p.add_argument("--skip-gh", action="store_true", help="Skip GitHub release + workflow steps.")
    p.add_argument("--skip-zenodo", action="store_true", help="Skip manual_doi workflow dispatch.")
    return p


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT, dry_run: bool = False) -> dict[str, Any]:
    step: dict[str, Any] = {
        "cmd": cmd,
        "cwd": str(cwd),
        "ok": True,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }
    if dry_run:
        step["status"] = "dry_run"
        return step
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    step["returncode"] = int(proc.returncode)
    step["stdout"] = str(proc.stdout or "")
    step["stderr"] = str(proc.stderr or "")
    step["ok"] = proc.returncode == 0
    return step


def _git_stdout(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return str(proc.stdout or "")


def _parse_version_from_init() -> str:
    text = INIT_PATH.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not m:
        raise RuntimeError("could not parse __version__ from yolozu/__init__.py")
    return str(m.group(1))


def _set_version_in_init(next_version: str) -> None:
    text = INIT_PATH.read_text(encoding="utf-8")
    nxt = re.sub(
        r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
        rf"\g<1>{next_version}\g<3>",
        text,
        count=1,
    )
    if nxt == text:
        raise RuntimeError("failed to update __version__ in yolozu/__init__.py")
    INIT_PATH.write_text(nxt, encoding="utf-8")


def _latest_semver_tag() -> str | None:
    out = _git_stdout("tag", "--list", "v[0-9]*", "--sort=-v:refname")
    for line in out.splitlines():
        tag = str(line).strip()
        if re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            return tag
    return None


def _shortstat_since(ref: str | None) -> tuple[int, int, int]:
    if ref:
        out = _git_stdout("diff", "--shortstat", f"{ref}..HEAD")
    else:
        out = _git_stdout("diff", "--shortstat", "--root", "HEAD")
    text = out.strip()
    if not text:
        return 0, 0, 0
    m_files = re.search(r"(\d+)\s+files?\s+changed", text)
    m_ins = re.search(r"(\d+)\s+insertions?\(\+\)", text)
    m_del = re.search(r"(\d+)\s+deletions?\(-\)", text)
    files = int(m_files.group(1)) if m_files else 0
    ins = int(m_ins.group(1)) if m_ins else 0
    dele = int(m_del.group(1)) if m_del else 0
    return files, ins, dele


def _classify_scale(files_changed: int, line_delta: int) -> str:
    if files_changed >= 25 or line_delta >= 800:
        return "large"
    if files_changed >= 8 or line_delta >= 150:
        return "medium"
    return "small"


def _bump_version(current: str, scale: str) -> str:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(current).strip())
    if not m:
        raise RuntimeError(f"unsupported version format (expected MAJOR.MINOR.PATCH): {current}")
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if scale == "small":
        patch += 1
    elif scale == "medium":
        minor += 1
        patch = 0
    elif scale == "large":
        major += 1
        minor = 0
        patch = 0
    else:
        raise RuntimeError(f"unknown bump scale: {scale}")
    return f"{major}.{minor}.{patch}"


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
        [PYTHON, "tools/validate_tool_manifest.py", "--manifest", "tools/manifest.json", "--require-declarative"],
        [PYTHON, "-m", "unittest", "tests.test_packaged_tools_manifest", "tests.test_manifest_docs_references"],
        [PYTHON, "tools/check_mcp_settings.py", "--output", "reports/mcp_settings_check.release.json"],
        [PYTHON, "tools/generate_integration_tool_reference.py", "--check"],
    ]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = Path(str(args.output))
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()

    warnings: list[str] = []
    errors: list[str] = []
    steps: list[dict[str, Any]] = []

    try:
        current_version = _parse_version_from_init()
    except Exception as exc:
        current_version = ""
        errors.append(str(exc))

    try:
        latest_tag = _latest_semver_tag()
    except Exception as exc:
        latest_tag = None
        errors.append(str(exc))

    try:
        files_changed, insertions, deletions = _shortstat_since(latest_tag)
    except Exception as exc:
        files_changed, insertions, deletions = 0, 0, 0
        errors.append(str(exc))

    line_delta = int(insertions + deletions)
    bump_scale = _classify_scale(files_changed, line_delta)
    bump_formulas = {
        "small": "X.Y.Z -> X.Y.(Z+1)  (1.1.1+add 相当)",
        "medium": "X.Y.Z -> X.(Y+1).0  (1.1+a.0 相当)",
        "large": "X.Y.Z -> (X+1).0.0  (1+a.0.0 相当)",
    }

    try:
        next_version = _bump_version(current_version, bump_scale) if current_version else ""
    except Exception as exc:
        next_version = ""
        errors.append(str(exc))
    tag = f"v{next_version}" if next_version else ""

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

    if tag:
        if _check_tag_exists_local(tag):
            errors.append(f"local tag already exists: {tag}")
        if _check_tag_exists_remote(tag):
            errors.append(f"remote tag already exists: {tag}")

    if not bool(args.skip_gh):
        gh_probe = _run(["gh", "--version"], dry_run=bool(args.dry_run))
        gh_probe["type"] = "gh_probe"
        steps.append(gh_probe)
        if not bool(gh_probe.get("ok")):
            errors.append("GitHub CLI `gh` is required for release/PyPI/Zenodo update steps")

    if not errors and not bool(args.skip_checks):
        for cmd in _quality_check_cmds():
            step = _run(cmd, dry_run=bool(args.dry_run))
            step["type"] = "quality_check"
            steps.append(step)
            if not bool(step.get("ok")):
                errors.append(f"quality check failed: {' '.join(cmd)}")
                break

    if not errors:
        if bool(args.dry_run):
            step = _run([PYTHON, "tools/release.py", "(set-version)", next_version], dry_run=True)
        else:
            try:
                _set_version_in_init(next_version)
                step = {
                    "type": "set_version",
                    "ok": True,
                    "returncode": 0,
                    "stdout": f"updated yolozu/__init__.py to {next_version}\n",
                    "stderr": "",
                    "cmd": [PYTHON, "tools/release.py", "(set-version)", next_version],
                    "cwd": str(REPO_ROOT),
                }
            except Exception as exc:
                step = {
                    "type": "set_version",
                    "ok": False,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": str(exc),
                    "cmd": [PYTHON, "tools/release.py", "(set-version)", next_version],
                    "cwd": str(REPO_ROOT),
                }
        step["type"] = "set_version"
        steps.append(step)
        if not bool(step.get("ok")):
            errors.append("failed to update package version")

    if not errors:
        for cmd, step_type, err in [
            (["git", "add", "yolozu/__init__.py"], "git_add_version", "failed to stage version file"),
            (["git", "commit", "-m", f"Release {tag}"], "git_commit", "failed to create release commit"),
            (["git", "tag", "-a", tag, "-m", f"YOLOZU {next_version}"], "git_tag", "failed to create release tag"),
            (["git", "push", "origin", "main"], "git_push_main", "failed to push main branch"),
            (["git", "push", "origin", tag], "git_push_tag", "failed to push release tag"),
        ]:
            step = _run(cmd, dry_run=bool(args.dry_run))
            step["type"] = step_type
            steps.append(step)
            if not bool(step.get("ok")):
                errors.append(err)
                break

    if not errors and not bool(args.skip_gh):
        release_step = _run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--title",
                f"YOLOZU {next_version}",
                "--generate-notes",
                "--latest",
            ],
            dry_run=bool(args.dry_run),
        )
        release_step["type"] = "gh_release_publish"
        steps.append(release_step)
        if not bool(release_step.get("ok")):
            errors.append("failed to publish GitHub release")

    if not errors and not bool(args.skip_gh):
        # release:published triggers publish.yml -> PyPI Trusted Publishing.
        pypi_step = _run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "publish.yml",
                "--limit",
                "1",
                "--json",
                "databaseId,status,conclusion,url,displayTitle,event",
            ],
            dry_run=bool(args.dry_run),
        )
        pypi_step["type"] = "pypi_workflow_probe"
        steps.append(pypi_step)
        if not bool(pypi_step.get("ok")):
            warnings.append("could not probe publish.yml run status via gh")

    if (
        not errors
        and not bool(args.skip_gh)
        and not bool(args.skip_zenodo)
    ):
        zenodo_step = _run(
            [
                "gh",
                "workflow",
                "run",
                "manual_doi.yml",
                "--ref",
                "main",
                "-f",
                "zenodo_environment=production",
                "-f",
                f"manual_version={next_version}",
                "-f",
                "publish_record=true",
            ],
            dry_run=bool(args.dry_run),
        )
        zenodo_step["type"] = "zenodo_workflow_dispatch"
        steps.append(zenodo_step)
        if not bool(zenodo_step.get("ok")):
            warnings.append("failed to dispatch manual_doi.yml via gh workflow run")
    elif not bool(args.skip_zenodo):
        warnings.append("Zenodo dispatch skipped because GitHub flow was skipped or failed")

    report = {
        "task": "release",
        "timestamp": _now_utc(),
        "ok": len(errors) == 0,
        "dry_run": bool(args.dry_run),
        "branch": branch,
        "dirty": dirty,
        "latest_tag": latest_tag,
        "current_version": current_version,
        "next_version": next_version,
        "tag": tag,
        "bump_scale": bump_scale,
        "bump_formula": bump_formulas.get(bump_scale),
        "scale_stats": {
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
            "line_delta": line_delta,
        },
        "release_actions": {
            "github_release_publish": not bool(args.skip_gh),
            "pypi_update_via_publish_workflow": not bool(args.skip_gh),
            "zenodo_manual_doi_dispatch": (not bool(args.skip_gh)) and (not bool(args.skip_zenodo)),
        },
        "warnings": warnings,
        "errors": errors,
        "steps": steps,
        "next_steps": [
            "Confirm publish.yml and manual_doi.yml runs in GitHub Actions.",
            "Verify PyPI package visibility and Zenodo record metadata after workflows complete.",
            "Share release notes and artifact links in announcement channels.",
        ],
    }
    _write_report(out, report)
    print(str(out))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
