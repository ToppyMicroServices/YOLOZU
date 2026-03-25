#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_value(payload: dict[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return "missing"
        current = current.get(key)
    if isinstance(current, dict):
        value = current.get("status")
        if isinstance(value, str):
            return value
    if isinstance(current, str):
        return current
    return "missing"


def _enabled_flag(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if isinstance(value, dict):
        enabled = value.get("enabled")
        if isinstance(enabled, bool):
            return enabled
    if isinstance(value, bool):
        return value
    return None


def _add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    ok: bool,
    expected: Any,
    actual: Any,
    source: str,
    required: bool = True,
    remediation: str | None = None,
) -> None:
    entry = {
        "id": check_id,
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
        "source": source,
        "required": bool(required),
    }
    if remediation:
        entry["remediation"] = remediation
    checks.append(entry)


def _check_path(repo_root: Path, rel_path: str) -> bool:
    return (repo_root / rel_path).exists()


def run_audit(
    *,
    repo_root: Path,
    repo_json: dict[str, Any] | None,
    branch_protection_json: dict[str, Any] | None,
    require_reviews: int = 1,
    allow_missing_evidence: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing_evidence: list[str] = []

    local_evidence = [
        (".github/workflows/codeql.yml", "codeql_workflow_present"),
        (".github/workflows/scorecard.yml", "scorecard_workflow_present"),
        (".github/workflows/cflite_pr.yml", "clusterfuzzlite_pr_present"),
        (".github/workflows/cflite_batch.yml", "clusterfuzzlite_batch_present"),
        (".github/dependabot.yml", "dependabot_config_present"),
        ("docs/security_scorecard_governance.md", "scorecard_governance_doc_present"),
        ("docs/release_reliability_checklist.md", "release_checklist_doc_present"),
    ]
    for rel_path, check_id in local_evidence:
        exists = _check_path(repo_root, rel_path)
        _add_check(
            checks,
            check_id=check_id,
            ok=exists,
            expected=True,
            actual=exists,
            source=f"repo:{rel_path}",
            remediation=f"Add or restore {rel_path}.",
        )

    if repo_json is None:
        missing_evidence.append("repo_json")
    else:
        _add_check(
            checks,
            check_id="default_branch_main",
            ok=str(repo_json.get("default_branch")) == "main",
            expected="main",
            actual=repo_json.get("default_branch"),
            source="repo_json",
            remediation="Set the repository default branch to 'main'.",
        )

        dependabot_status = _status_value(repo_json, "security_and_analysis", "dependabot_security_updates")
        _add_check(
            checks,
            check_id="dependabot_security_updates_enabled",
            ok=dependabot_status == "enabled",
            expected="enabled",
            actual=dependabot_status,
            source="repo_json.security_and_analysis.dependabot_security_updates",
            remediation="Enable Dependabot security updates in repository security settings.",
        )

        secret_scanning_status = _status_value(repo_json, "security_and_analysis", "secret_scanning")
        _add_check(
            checks,
            check_id="secret_scanning_enabled",
            ok=secret_scanning_status == "enabled",
            expected="enabled",
            actual=secret_scanning_status,
            source="repo_json.security_and_analysis.secret_scanning",
            required=False,
            remediation="Enable secret scanning if your repository policy requires repository-side secret detection.",
        )

        push_protection_status = _status_value(
            repo_json,
            "security_and_analysis",
            "secret_scanning_push_protection",
        )
        _add_check(
            checks,
            check_id="secret_scanning_push_protection_enabled",
            ok=push_protection_status == "enabled",
            expected="enabled",
            actual=push_protection_status,
            source="repo_json.security_and_analysis.secret_scanning_push_protection",
            required=False,
            remediation="Enable push protection if your organization requires repository-side push-time secret blocking.",
        )

    if branch_protection_json is None:
        missing_evidence.append("branch_protection_json")
    else:
        reviews = branch_protection_json.get("required_pull_request_reviews") or {}
        required_review_count = reviews.get("required_approving_review_count")
        _add_check(
            checks,
            check_id="required_pr_reviews",
            ok=isinstance(required_review_count, int) and required_review_count >= require_reviews,
            expected=f">={require_reviews}",
            actual=required_review_count,
            source="branch_protection.required_pull_request_reviews.required_approving_review_count",
            remediation="Require at least one approving pull request review on main.",
        )
        _add_check(
            checks,
            check_id="dismiss_stale_reviews",
            ok=bool(reviews.get("dismiss_stale_reviews")),
            expected=True,
            actual=reviews.get("dismiss_stale_reviews"),
            source="branch_protection.required_pull_request_reviews.dismiss_stale_reviews",
            remediation="Enable stale review dismissal for protected branches.",
        )
        _add_check(
            checks,
            check_id="require_last_push_approval",
            ok=bool(reviews.get("require_last_push_approval")),
            expected=True,
            actual=reviews.get("require_last_push_approval"),
            source="branch_protection.required_pull_request_reviews.require_last_push_approval",
            remediation="Require approval after the most recent push.",
        )
        _add_check(
            checks,
            check_id="require_conversation_resolution",
            ok=_enabled_flag(branch_protection_json, "required_conversation_resolution") is True,
            expected=True,
            actual=_enabled_flag(branch_protection_json, "required_conversation_resolution"),
            source="branch_protection.required_conversation_resolution.enabled",
            remediation="Require conversation resolution before merging to main.",
        )
        _add_check(
            checks,
            check_id="require_linear_history",
            ok=_enabled_flag(branch_protection_json, "required_linear_history") is True,
            expected=True,
            actual=_enabled_flag(branch_protection_json, "required_linear_history"),
            source="branch_protection.required_linear_history.enabled",
            remediation="Enable linear history on main.",
        )
        _add_check(
            checks,
            check_id="enforce_admins",
            ok=_enabled_flag(branch_protection_json, "enforce_admins") is True,
            expected=True,
            actual=_enabled_flag(branch_protection_json, "enforce_admins"),
            source="branch_protection.enforce_admins.enabled",
            remediation="Include administrators in branch protection enforcement.",
        )
        _add_check(
            checks,
            check_id="force_pushes_disabled",
            ok=_enabled_flag(branch_protection_json, "allow_force_pushes") is False,
            expected=False,
            actual=_enabled_flag(branch_protection_json, "allow_force_pushes"),
            source="branch_protection.allow_force_pushes.enabled",
            remediation="Disable force pushes on main.",
        )
        _add_check(
            checks,
            check_id="deletions_disabled",
            ok=_enabled_flag(branch_protection_json, "allow_deletions") is False,
            expected=False,
            actual=_enabled_flag(branch_protection_json, "allow_deletions"),
            source="branch_protection.allow_deletions.enabled",
            remediation="Disable branch deletions on main.",
        )

    manual_followups = [
        {
            "id": "CodeReviewID",
            "status": "manual",
            "why": "Scorecard requires reviewed pull request history; branch protection alone is not sufficient evidence.",
            "operator_action": "Land changes through reviewed pull requests and avoid direct pushes to the protected default branch.",
        },
        {
            "id": "MaintainedID",
            "status": "manual",
            "why": "Repository age, release cadence, and issue hygiene are time-based signals.",
            "operator_action": "Keep releases regular, update stale P1/P2 issues, and keep default-branch CI green.",
        },
        {
            "id": "CIIBestPracticesID",
            "status": "manual",
            "why": "Badge enrollment and checklist completion happen outside the repository tree.",
            "operator_action": "Maintain the OpenSSF Best Practices badge project and keep the linked evidence current.",
        },
    ]

    failed_required = [check for check in checks if check["required"] and not check["ok"]]
    advisory_failures = [check for check in checks if not check["required"] and not check["ok"]]

    return {
        "ok": not failed_required and (allow_missing_evidence or not missing_evidence),
        "repo_root": str(repo_root),
        "inputs": {
            "repo_json_present": repo_json is not None,
            "branch_protection_json_present": branch_protection_json is not None,
            "allow_missing_evidence": bool(allow_missing_evidence),
        },
        "checks": checks,
        "missing_evidence": missing_evidence,
        "failed_required_checks": [check["id"] for check in failed_required],
        "failed_advisory_checks": [check["id"] for check in advisory_failures],
        "manual_followups": manual_followups,
        "collection_hints": {
            "repo_json": "gh api repos/ToppyMicroServices/YOLOZU > reports/github_governance/repo.json",
            "branch_protection_json": "gh api repos/ToppyMicroServices/YOLOZU/branches/main/protection > reports/github_governance/branch_protection_main.json",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit repository governance posture from local workflow evidence and exported GitHub settings snapshots.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to inspect for local workflow/docs evidence (default: current directory).",
    )
    parser.add_argument(
        "--repo-json",
        help="Path to a local `gh api repos/<owner>/<repo>` JSON snapshot.",
    )
    parser.add_argument(
        "--branch-protection-json",
        help="Path to a local `gh api repos/<owner>/<repo>/branches/main/protection` JSON snapshot.",
    )
    parser.add_argument(
        "--require-reviews",
        type=int,
        default=1,
        help="Minimum approving reviews expected on the protected default branch (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="reports/repo_governance_check.json",
        help="Write the JSON audit report here (default: reports/repo_governance_check.json).",
    )
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Treat missing GitHub snapshot inputs as informational so local-only evidence checks can still pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    repo_json = _load_json(Path(args.repo_json)) if args.repo_json else None
    branch_protection_json = _load_json(Path(args.branch_protection_json)) if args.branch_protection_json else None

    result = run_audit(
        repo_root=repo_root,
        repo_json=repo_json,
        branch_protection_json=branch_protection_json,
        require_reviews=max(0, int(args.require_reviews)),
        allow_missing_evidence=bool(args.allow_missing_evidence),
    )

    def _redact_scalar(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        lowered = value.lower()
        if any(token in lowered for token in ("token", "secret", "password", "apikey", "api_key")):
            return "<redacted>"
        return value

    def _redact_payload(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _redact_payload(_redact_scalar(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [_redact_payload(_redact_scalar(item)) for item in value]
        return _redact_scalar(value)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_redact_payload(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failed_required = result["failed_required_checks"]
    missing_evidence = result["missing_evidence"]
    summary = (
        f"repo governance check: {'OK' if result['ok'] else 'FAIL'} "
        f"(required_failed={len(failed_required)}, missing_evidence={len(missing_evidence)}, "
        f"manual_followups={len(result['manual_followups'])})"
    )
    print(summary)
    print(output_path)
    if failed_required:
        print(f"required failures count: {len(failed_required)}")
    if missing_evidence:
        print(f"missing evidence count: {len(missing_evidence)}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
