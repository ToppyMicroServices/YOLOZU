#!/usr/bin/env python3
"""Single-command release automation for YOLOZU.

Default behavior (`python3 tools/release.py`):
1) Read current package version from `yolozu/__init__.py`.
2) Classify change scale (small/medium/large) from git diff stats since the latest tag that matches the active versioning scheme.
3) Bump semantic version automatically, using explicit breaking-change signals for major releases.
4) Run release quality checks.
5) Atomically synchronize package, changelog, citation, and current manifest-example metadata.
6) Commit and create/push the git tag.
7) Create the published GitHub release (which triggers PyPI and manual DOI workflows).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__:
    from .release_metadata import (
        ReleaseMetadataPlan,
        prepare_release_metadata,
        validate_release_metadata,
        write_release_metadata_atomic,
    )
else:
    from release_metadata import (
        ReleaseMetadataPlan,
        prepare_release_metadata,
        validate_release_metadata,
        write_release_metadata_atomic,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "yolozu" / "__init__.py"
PYTHON = sys.executable  # Use the same interpreter as release.sh selected.
SEMVER_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
CALVER_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d+)")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run full release flow with zero required options: auto-bump version, tag, "
            "and GitHub Release publish. The release event is the single automatic "
            "trigger for PyPI and the Zenodo manual DOI workflow."
        )
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan/report only; do not mutate git/GitHub.")
    p.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate current package, CHANGELOG, CITATION, tag-form, and manifest "
            "metadata without preparing or publishing a release."
        ),
    )
    p.add_argument(
        "--output",
        default="reports/release_report.json",
        help="Output report path (default: reports/release_report.json).",
    )
    p.add_argument("--allow-dirty", action="store_true", help="Allow dirty git working tree.")
    p.add_argument("--allow-non-main", action="store_true", help="Allow running outside main branch.")
    p.add_argument(
        "--allow-major",
        action="store_true",
        help="Allow a major SemVer bump when an explicit breaking-change signal is detected.",
    )
    p.add_argument("--skip-checks", action="store_true", help="Skip local quality checks.")
    p.add_argument("--skip-gh", action="store_true", help="Skip GitHub release + workflow steps.")
    p.add_argument(
        "--skip-zenodo",
        action="store_true",
        help=(
            "Deprecated fail-closed option. A published GitHub Release automatically "
            "triggers manual_doi.yml; use --skip-gh to avoid publishing the release."
        ),
    )
    p.add_argument(
        "--versioning",
        choices=("auto", "semver", "calver"),
        default="auto",
        help=(
            "Release versioning scheme. 'auto' detects SemVer (X.Y.Z) or CalVer "
            "(YYYY.MM.DD.MICRO) from yolozu.__version__."
        ),
    )
    return p


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def _run_git(cmd: list[str], *, dry_run: bool = False, max_attempts: int = 5) -> dict[str, Any]:
    """Run a git command with a small retry loop.

    We occasionally see transient failures creating `.git/index.lock` (e.g. file system races).
    Retrying is safe for `git add/commit/tag/push` in this release flow.
    """
    last: dict[str, Any] | None = None
    for attempt in range(1, int(max_attempts) + 1):
        step = _run(cmd, dry_run=dry_run)
        step["attempt"] = attempt
        last = step
        if bool(step.get("ok")):
            return step

        stderr = str(step.get("stderr") or "")
        # Heuristic: retry only for known transient index lock failures.
        if "index.lock" in stderr and ("Unable to create" in stderr or "File exists" in stderr or "Operation not permitted" in stderr):
            if not dry_run:
                time.sleep(min(1.0, 0.1 * (2** (attempt - 1))))
            continue
        return step

    return last if last is not None else _run(cmd, dry_run=dry_run)


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


def _release_subjects_since(ref: str | None) -> list[str]:
    if ref:
        out = _git_stdout("log", "--format=%s", f"{ref}..HEAD")
    else:
        out = _git_stdout("log", "--format=%s", "--root", "HEAD")
    subjects: list[str] = []
    for line in out.splitlines():
        subject = line.strip()
        if not subject:
            continue
        if subject.lower().startswith(("release ", "chore: release ")):
            continue
        subjects.append(subject)
    return subjects


def _entry_from_subject(subject: str) -> str:
    text = re.sub(r"\s*\(#\d+\)\s*$", "", subject.strip())
    conventional = re.match(r"^[a-z]+(?:\([^)]+\))?!?:\s*(.+)$", text)
    if conventional:
        text = conventional.group(1).strip()
    if not text:
        text = "Prepare release metadata"
    text = text[0].upper() + text[1:]
    if text[-1:] not in ".!?":
        text += "."
    return f"- {text}"


def _changelog_section(next_version: str, *, date: str, ref: str | None) -> str:
    entries = [_entry_from_subject(subject) for subject in _release_subjects_since(ref)]
    if not entries:
        entries = [f"- Prepare release metadata for v{next_version}."]
    return f"## [{next_version}] - {date}\n\n### Changed\n" + "\n".join(entries) + "\n"


def _detect_versioning_scheme(version: str) -> str:
    value = str(version).strip()
    if SEMVER_RE.fullmatch(value):
        return "semver"
    if CALVER_RE.fullmatch(value):
        return "calver"
    raise RuntimeError(
        f"unsupported version format: {version!r} (expected SemVer X.Y.Z or CalVer YYYY.MM.DD.MICRO)"
    )


def _latest_version_tag(versioning: str) -> str | None:
    out = _git_stdout("tag", "--list", "v[0-9]*", "--sort=-v:refname")
    for line in out.splitlines():
        tag = str(line).strip()
        body = tag[1:] if tag.startswith("v") else tag
        if versioning == "semver" and SEMVER_RE.fullmatch(body):
            return tag
        if versioning == "calver" and CALVER_RE.fullmatch(body):
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


def _contains_breaking_change_signal(body_text: str, subject_text: str) -> bool:
    if "BREAKING CHANGE" in body_text or "BREAKING-CHANGE" in body_text:
        return True
    return bool(re.search(r"^[^\n:]+!:", subject_text, flags=re.MULTILINE))


def _has_breaking_change_signal(ref: str | None) -> bool:
    if ref:
        body_out = _git_stdout("log", "--format=%B", f"{ref}..HEAD")
        subject_out = _git_stdout("log", "--format=%s", f"{ref}..HEAD")
    else:
        body_out = _git_stdout("log", "--format=%B", "--root", "HEAD")
        subject_out = _git_stdout("log", "--format=%s", "--root", "HEAD")
    return _contains_breaking_change_signal(str(body_out or ""), str(subject_out or ""))


def _recommended_semver_bump(scale: str, *, breaking: bool) -> str:
    if breaking:
        return "major"
    if scale in {"medium", "large"}:
        return "minor"
    if scale == "small":
        return "patch"
    raise RuntimeError(f"unknown release scale: {scale}")


def _bump_semver(current: str, bump: str) -> str:
    m = SEMVER_RE.fullmatch(str(current).strip())
    if not m:
        raise RuntimeError(f"unsupported SemVer format (expected MAJOR.MINOR.PATCH): {current}")
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if bump == "patch":
        patch += 1
    elif bump == "minor":
        minor += 1
        patch = 0
    elif bump == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise RuntimeError(f"unknown semver bump kind: {bump}")
    return f"{major}.{minor}.{patch}"


def _major_bump_requires_confirmation(*, versioning: str, semver_bump: str, allow_major: bool) -> bool:
    return versioning == "semver" and semver_bump == "major" and not bool(allow_major)


def _today_utc_calver_parts() -> tuple[int, int, int]:
    now = datetime.now(timezone.utc)
    return int(now.year), int(now.month), int(now.day)


def _bump_calver(current: str, *, today: tuple[int, int, int] | None = None) -> str:
    m = CALVER_RE.fullmatch(str(current).strip())
    if not m:
        raise RuntimeError(f"unsupported CalVer format (expected YYYY.MM.DD.MICRO): {current}")
    cur_year, cur_month, cur_day, cur_micro = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
    )
    next_year, next_month, next_day = today or _today_utc_calver_parts()
    if (cur_year, cur_month, cur_day) == (next_year, next_month, next_day):
        next_micro = cur_micro + 1
    else:
        next_micro = 0
    return f"{next_year:04d}.{next_month:02d}.{next_day:02d}.{next_micro}"


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
        [PYTHON, "tools/generate_adaptive_vision_roadmap.py", "--check", "--json"],
        [
            PYTHON,
            "-m",
            "unittest",
            "tests.test_adaptive_vision_roadmap_generator",
            "tests.test_packaged_tools_manifest",
            "tests.test_manifest_docs_references",
        ],
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

    if bool(args.skip_zenodo):
        errors.append(
            "--skip-zenodo cannot suppress the release-triggered manual DOI workflow; "
            "use --skip-gh to avoid publishing the GitHub Release"
        )

    try:
        current_version = _parse_version_from_init()
    except Exception as exc:
        current_version = ""
        errors.append(str(exc))

    metadata_validation = validate_release_metadata(
        REPO_ROOT,
        expected_version=current_version or None,
        expected_tag=f"v{current_version}" if current_version else None,
    )
    errors.extend(str(error) for error in metadata_validation.get("errors") or [])

    try:
        branch = _git_stdout("rev-parse", "--abbrev-ref", "HEAD").strip()
    except Exception as exc:
        branch = ""
        errors.append(str(exc))
    try:
        dirty = bool(_git_stdout("status", "--porcelain").strip())
    except Exception as exc:
        dirty = False
        errors.append(str(exc))

    if bool(args.check):
        report = {
            "task": "release",
            "timestamp": _now_utc(),
            "ok": len(errors) == 0,
            "check": True,
            "dry_run": bool(args.dry_run),
            "non_writing": True,
            "branch": branch,
            "dirty": dirty,
            "current_version": current_version,
            "tag": f"v{current_version}" if current_version else "",
            "metadata_validation": metadata_validation,
            "warnings": warnings,
            "errors": errors,
            "steps": steps,
        }
        _write_report(out, report)
        print(str(out))
        return 0 if report["ok"] else 1

    try:
        detected_versioning = _detect_versioning_scheme(current_version) if current_version else ""
    except Exception as exc:
        detected_versioning = ""
        errors.append(str(exc))

    versioning = str(args.versioning).strip()
    if versioning == "auto":
        versioning = detected_versioning
    if versioning not in {"semver", "calver"}:
        errors.append(f"could not resolve release versioning scheme from --versioning={args.versioning!r}")

    try:
        latest_tag = _latest_version_tag(versioning) if versioning else None
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
    try:
        breaking_change_detected = _has_breaking_change_signal(latest_tag) if versioning == "semver" else False
    except Exception as exc:
        breaking_change_detected = False
        errors.append(str(exc))
    try:
        semver_bump = _recommended_semver_bump(bump_scale, breaking=breaking_change_detected) if versioning == "semver" else ""
    except Exception as exc:
        semver_bump = ""
        errors.append(str(exc))
    bump_formulas = {
        "patch": "X.Y.Z -> X.Y.(Z+1)  (non-breaking small fix/add 相当)",
        "minor": "X.Y.Z -> X.(Y+1).0  (non-breaking feature/add 相当)",
        "major": "X.Y.Z -> (X+1).0.0  (explicit breaking change 相当)",
    }
    calver_formula = "YYYY.MM.DD.MICRO -> same UTC day: MICRO+1, new UTC day: YYYY.MM.DD.0"

    if _major_bump_requires_confirmation(
        versioning=versioning,
        semver_bump=semver_bump,
        allow_major=bool(args.allow_major),
    ):
        errors.append(
            "major release detected from breaking-change signal, but --allow-major was not provided; "
            "re-run with --allow-major only after confirming the breaking surface is intentional"
        )

    try:
        if not current_version:
            next_version = ""
        elif versioning == "semver":
            next_version = _bump_semver(current_version, semver_bump)
        elif versioning == "calver":
            next_version = _bump_calver(current_version)
        else:
            raise RuntimeError(f"unknown versioning scheme: {versioning}")
    except Exception as exc:
        next_version = ""
        errors.append(str(exc))
    tag = f"v{next_version}" if next_version else ""

    if branch and branch != "main" and not bool(args.allow_non_main):
        errors.append(f"current branch is '{branch}' (expected 'main'); use --allow-non-main to bypass")

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

    release_date = _today_utc()
    metadata_plan: ReleaseMetadataPlan | None = None
    if not errors:
        try:
            changelog_section = _changelog_section(
                next_version,
                date=release_date,
                ref=latest_tag,
            )
            metadata_plan = prepare_release_metadata(
                REPO_ROOT,
                current_version=current_version,
                next_version=next_version,
                release_date=release_date,
                changelog_section=changelog_section,
            )
        except Exception as exc:
            errors.append(f"failed to prepare synchronized release metadata: {exc}")

    if not errors and not bool(args.skip_checks):
        for cmd in _quality_check_cmds():
            step = _run(cmd, dry_run=bool(args.dry_run))
            step["type"] = "quality_check"
            steps.append(step)
            if not bool(step.get("ok")):
                errors.append(f"quality check failed: {' '.join(cmd)}")
                break

    if not errors and metadata_plan is not None:
        if bool(args.dry_run):
            step = _run(
                [
                    PYTHON,
                    "tools/release.py",
                    "(sync-release-metadata)",
                    *metadata_plan.changed_paths,
                ],
                dry_run=True,
            )
        else:
            try:
                write_release_metadata_atomic(REPO_ROOT, metadata_plan)
                step = {
                    "type": "sync_release_metadata",
                    "ok": True,
                    "returncode": 0,
                    "stdout": (
                        "updated synchronized release metadata: "
                        + ", ".join(metadata_plan.changed_paths)
                        + "\n"
                    ),
                    "stderr": "",
                    "cmd": [
                        PYTHON,
                        "tools/release.py",
                        "(sync-release-metadata)",
                        *metadata_plan.changed_paths,
                    ],
                    "cwd": str(REPO_ROOT),
                }
            except Exception as exc:
                step = {
                    "type": "sync_release_metadata",
                    "ok": False,
                    "returncode": 1,
                    "stdout": "",
                    "stderr": str(exc),
                    "cmd": [
                        PYTHON,
                        "tools/release.py",
                        "(sync-release-metadata)",
                        *metadata_plan.changed_paths,
                    ],
                    "cwd": str(REPO_ROOT),
                }
        step["type"] = "sync_release_metadata"
        steps.append(step)
        if not bool(step.get("ok")):
            errors.append("failed to atomically synchronize release metadata")

    if not errors and metadata_plan is not None:
        stage_paths = list(metadata_plan.changed_paths)
        for cmd, step_type, err in [
            (["git", "add", *stage_paths], "git_add_version", "failed to stage release files"),
            (["git", "commit", "-m", f"chore: release {tag}"], "git_commit", "failed to create release commit"),
            (["git", "tag", "-a", tag, "-m", f"YOLOZU {next_version}"], "git_tag", "failed to create release tag"),
            (["git", "push", "origin", "main"], "git_push_main", "failed to push main branch"),
            (["git", "push", "origin", tag], "git_push_tag", "failed to push release tag"),
        ]:
            step = _run_git(cmd, dry_run=bool(args.dry_run))
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

    report = {
        "task": "release",
        "timestamp": _now_utc(),
        "ok": len(errors) == 0,
        "check": False,
        "dry_run": bool(args.dry_run),
        "non_writing": bool(args.dry_run),
        "branch": branch,
        "dirty": dirty,
        "latest_tag": latest_tag,
        "current_version": current_version,
        "detected_versioning": detected_versioning,
        "versioning_scheme": versioning,
        "next_version": next_version,
        "tag": tag,
        "release_date": release_date,
        "metadata_validation": metadata_validation,
        "metadata_plan": metadata_plan.report() if metadata_plan is not None else {},
        "bump_scale": bump_scale,
        "breaking_change_detected": breaking_change_detected,
        "semver_bump": semver_bump if versioning == "semver" else "",
        "bump_formula": calver_formula if versioning == "calver" else bump_formulas.get(semver_bump),
        "scale_stats": {
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
            "line_delta": line_delta,
        },
        "release_actions": {
            "github_release_publish": not bool(args.skip_gh),
            "pypi_update_via_publish_workflow": not bool(args.skip_gh),
            "zenodo_manual_doi_via_release_event": not bool(args.skip_gh),
            "zenodo_manual_doi_dispatch": False,
        },
        "warnings": warnings,
        "errors": errors,
        "steps": steps,
        "next_steps": [
            "Confirm the release-triggered publish.yml and manual_doi.yml runs in GitHub Actions.",
            "Verify PyPI package visibility and Zenodo record metadata after workflows complete.",
            "Share release notes and artifact links in announcement channels.",
        ],
    }
    _write_report(out, report)
    print(str(out))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
