#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "yolozu" / "data" / "manifest" / "adaptive_vision_roadmap.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "adaptive_vision_roadmap.md"
MAX_SOURCE_BYTES = 256 * 1024
ISSUE_RE = re.compile(r"^YOLOZU-ll2\.81(?:\.[0-9]+)+$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

ROOT_KEYS = {
    "continuous_intake",
    "generated_report",
    "guardrails",
    "kind",
    "next_issue_id",
    "out_of_scope",
    "phases",
    "product_boundary",
    "schema_version",
    "scope",
    "snapshot",
    "source_of_truth",
}

INTAKE_FLOW = [
    "discover_from_monitored_primary_sources",
    "screen_provenance_license_runtime_and_interface",
    "prepare_approved_assets_in_isolation",
    "qualify_on_an_exact_environment_workload_and_protocol",
    "activate_reviewed_evidence",
    "promote_through_a_separate_human_review",
    "monitor_expiry_runtime_drift_and_regressions",
    "revoke_or_roll_back_without_reusing_invalid_evidence",
]

INTAKE_LABELS = {
    "discover_from_monitored_primary_sources": "Discover from monitored primary sources",
    "screen_provenance_license_runtime_and_interface": "Screen provenance, license, runtime, and interface compatibility",
    "prepare_approved_assets_in_isolation": "Prepare approved assets in isolation",
    "qualify_on_an_exact_environment_workload_and_protocol": "Qualify on an exact environment, workload, and protocol",
    "activate_reviewed_evidence": "Activate reviewed evidence",
    "promote_through_a_separate_human_review": "Promote through a separate human review",
    "monitor_expiry_runtime_drift_and_regressions": "Monitor expiry, runtime drift, and regressions",
    "revoke_or_roll_back_without_reusing_invalid_evidence": "Revoke or roll back without reusing invalid evidence",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the adaptive-vision roadmap report from its packaged JSON projection."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Repository-local roadmap projection JSON (default: yolozu/data/manifest/adaptive_vision_roadmap.json).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Repository-local generated Markdown report (default: reports/adaptive_vision_roadmap.md).",
    )
    parser.add_argument("--check", action="store_true", help="Fail without writing when the report is stale.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable generation summary.")
    return parser.parse_args(argv)


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "<outside-repository>"


def _resolve_repo_path(raw: str, where: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{where} must stay within the repository") from exc
    return resolved


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_projection(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"source not found: {_repo_rel(path)}") from exc
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError(f"source exceeds {MAX_SOURCE_BYTES} bytes: {_repo_rel(path)}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"source is not UTF-8: {_repo_rel(path)}") from exc
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid roadmap JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("roadmap root must be an object")
    validate_projection(value)
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{where} keys mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def _string(value: Any, where: str, *, maximum: int = 768) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string of at most {maximum} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where} must contain valid Unicode scalar values") from exc
    if len(value) > maximum:
        raise ValueError(f"{where} must be a non-empty string of at most {maximum} characters")
    return value


def _string_list(value: Any, where: str, *, maximum_items: int = 32, maximum_chars: int = 768) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > maximum_items:
        raise ValueError(f"{where} must contain 1..{maximum_items} strings")
    items = [_string(item, f"{where}[]", maximum=maximum_chars) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"{where} contains duplicates")
    return items


def _positive_int(value: Any, where: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{where} must be an integer in 1..{maximum}")
    return value


def validate_projection(value: dict[str, Any]) -> None:
    _exact_keys(value, ROOT_KEYS, "roadmap")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("schema_version must be integer 1")
    if value["kind"] != "adaptive_vision_roadmap_projection":
        raise ValueError("kind must be adaptive_vision_roadmap_projection")
    if value["generated_report"] != "reports/adaptive_vision_roadmap.md":
        raise ValueError("generated_report must name the canonical report")
    if value["continuous_intake"] != INTAKE_FLOW:
        raise ValueError("continuous_intake must use the governed ordered flow")

    next_issue_id = _string(value["next_issue_id"], "next_issue_id", maximum=64)
    if ISSUE_RE.fullmatch(next_issue_id) is None:
        raise ValueError("next_issue_id is not under YOLOZU-ll2.81")
    _string_list(value["guardrails"], "guardrails", maximum_items=16, maximum_chars=512)
    _string_list(value["out_of_scope"], "out_of_scope", maximum_items=32, maximum_chars=128)

    snapshot = value["snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be an object")
    _exact_keys(
        snapshot,
        {"beads_export_commit", "date", "planning_source", "planning_source_doc", "root_issue_id", "timezone"},
        "snapshot",
    )
    if snapshot["planning_source"] != "Beads" or snapshot["planning_source_doc"] != "docs/roadmap.md":
        raise ValueError("snapshot must identify the Beads planning source and docs/roadmap.md")
    if snapshot["root_issue_id"] != "YOLOZU-ll2.81" or snapshot["timezone"] != "Asia/Tokyo":
        raise ValueError("snapshot root issue or timezone is invalid")
    if (
        not isinstance(snapshot["beads_export_commit"], str)
        or GIT_COMMIT_RE.fullmatch(snapshot["beads_export_commit"]) is None
    ):
        raise ValueError("snapshot.beads_export_commit must be 40 lowercase hexadecimal characters")
    snapshot_date = _string(snapshot["date"], "snapshot.date", maximum=10)
    try:
        parsed_date = date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise ValueError("snapshot.date must be a valid YYYY-MM-DD date") from exc
    if parsed_date.isoformat() != snapshot_date:
        raise ValueError("snapshot.date must use canonical YYYY-MM-DD form")

    boundary = value["product_boundary"]
    if not isinstance(boundary, dict):
        raise ValueError("product_boundary must be an object")
    _exact_keys(
        boundary,
        {
            "current_stable_lane",
            "execution_boundary",
            "natural_language_boundary",
            "public_availability",
            "recommendation_boundary",
            "target_maturity",
        },
        "product_boundary",
    )
    _string(boundary["current_stable_lane"], "product_boundary.current_stable_lane", maximum=256)
    for key in ("execution_boundary", "natural_language_boundary", "recommendation_boundary"):
        _string(boundary[key], f"product_boundary.{key}", maximum=512)
    if boundary["public_availability"] != "future_experimental_work":
        raise ValueError("public_availability must remain future_experimental_work")
    if boundary["target_maturity"] != "Experimental":
        raise ValueError("target_maturity must remain Experimental")

    scope = value["scope"]
    if not isinstance(scope, dict):
        raise ValueError("scope must be an object")
    _exact_keys(scope, {"initial_inputs", "initial_tasks", "later_bounded_lanes", "selection_factors"}, "scope")
    if scope["initial_inputs"] != ["single_image", "bounded_directory"]:
        raise ValueError("scope.initial_inputs must use the bounded v1 order")
    if scope["initial_tasks"] != ["object_detection", "instance_segmentation"]:
        raise ValueError("scope.initial_tasks must use the bounded v1 order")
    if scope["later_bounded_lanes"] != ["local_stream", "session_tracking", "static_image_ocr"]:
        raise ValueError("scope.later_bounded_lanes must use the governed P2 order")
    _string_list(scope["selection_factors"], "scope.selection_factors", maximum_items=16, maximum_chars=128)

    phases = value["phases"]
    if not isinstance(phases, list) or len(phases) != 5:
        raise ValueError("phases must contain exactly five entries")
    phase_ids: set[str] = set()
    for index, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            raise ValueError(f"phases[{index - 1}] must be an object")
        _exact_keys(phase, {"child_issue_count", "deliverable", "issue_id", "order", "priority", "state", "title"}, f"phases[{index - 1}]")
        if _positive_int(phase["order"], f"phases[{index - 1}].order", maximum=5) != index:
            raise ValueError("phase orders must be exactly 1..5")
        expected_id = f"YOLOZU-ll2.81.{index}"
        if phase["issue_id"] != expected_id or phase["issue_id"] in phase_ids:
            raise ValueError(f"phase {index} must use unique issue_id {expected_id}")
        phase_ids.add(phase["issue_id"])
        if phase["priority"] not in {"P0", "P1", "P2"} or phase["state"] != "roadmap":
            raise ValueError(f"phase {index} priority or state is invalid")
        _positive_int(phase["child_issue_count"], f"phases[{index - 1}].child_issue_count", maximum=100)
        _string(phase["title"], f"phases[{index - 1}].title", maximum=128)
        _string(phase["deliverable"], f"phases[{index - 1}].deliverable", maximum=768)

    truth = value["source_of_truth"]
    if not isinstance(truth, dict):
        raise ValueError("source_of_truth must be an object")
    _exact_keys(truth, {"capability_boundary", "generated_human_report", "live_task_state"}, "source_of_truth")
    if truth["capability_boundary"] != "yolozu/data/manifest/adaptive_vision_roadmap.json":
        raise ValueError("source_of_truth.capability_boundary is invalid")
    if truth["generated_human_report"] != value["generated_report"]:
        raise ValueError("generated report paths disagree")
    _string(truth["live_task_state"], "source_of_truth.live_task_state", maximum=256)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def render_markdown(value: dict[str, Any], *, source_path: Path) -> str:
    validate_projection(value)
    snapshot = value["snapshot"]
    boundary = value["product_boundary"]
    scope = value["scope"]
    lines = [
        "# Adaptive Vision Roadmap Projection",
        "",
        "<!-- Generated by tools/generate_adaptive_vision_roadmap.py from "
        f"{_repo_rel(source_path)}. Do not edit by hand. -->",
        "",
        f"Snapshot: `{snapshot['date']}` ({snapshot['timezone']})",
        f"Planning root: `{snapshot['root_issue_id']}`",
        f"Beads export commit: `{snapshot['beads_export_commit']}`",
        "",
        "> This is a roadmap projection, not qualification evidence. It does not make the adaptive router, model adapters, streaming, tracking, or OCR available in the current release.",
        "",
        "## Current Support Boundary",
        "",
        f"YOLOZU's current Stable lane remains {boundary['current_stable_lane']}.",
        f"The target maturity for this future program is `{boundary['target_maturity']}`. Current availability is `{boundary['public_availability']}`.",
        "A successful import, smoke run, public model-card number, or candidate record does not change that boundary.",
        "",
        "## Target Behavior",
        "",
        f"- Natural language: {boundary['natural_language_boundary']}",
        f"- Recommendation: {boundary['recommendation_boundary']}",
        f"- Execution: {boundary['execution_boundary']}",
        "- Environment fit: task, hardware, runtime, precision, memory, latency/FPS, quality protocol, offline policy, and license constraints must match measured evidence.",
        "",
        "## Bounded Scope",
        "",
        f"Initial tasks: {_code_list(scope['initial_tasks'])}.",
        f"Initial inputs: {_code_list(scope['initial_inputs'])}.",
        f"Later bounded lanes: {_code_list(scope['later_bounded_lanes'])}.",
        "",
        "Selection considers:",
        "",
    ]
    lines.extend(f"- {item}" for item in scope["selection_factors"])
    lines.extend(
        [
            "",
            "## Delivery Epics",
            "",
            "| Order | Bead | Priority | Delivery state | Child tasks | Deliverable |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    for phase in value["phases"]:
        lines.append(
            f"| {phase['order']} | `{phase['issue_id']}` — {_md_escape(phase['title'])} | "
            f"`{phase['priority']}` | `{phase['state']}` | {phase['child_issue_count']} | "
            f"{_md_escape(phase['deliverable'])} |"
        )
    lines.extend(["", "## Continuous Algorithm Intake", ""])
    for index, item in enumerate(value["continuous_intake"], start=1):
        lines.append(f"{index}. {INTAKE_LABELS[item]}.")
    lines.extend(["", "Discovery never changes the selectable channel by itself.", "", "## Safety And Evidence Guardrails", ""])
    lines.extend(f"- {item}" for item in value["guardrails"])
    lines.extend(["", "## Outside This Roadmap", ""])
    lines.extend(f"- {item}" for item in value["out_of_scope"])
    lines.extend(
        [
            "",
            "## Authority And Sync Rule",
            "",
            "Beads is the source of truth for live issue state, dependencies, and completion.",
            f"The packaged JSON at `{value['source_of_truth']['capability_boundary']}` is the dated public scope projection.",
            f"This report is generated at `{value['source_of_truth']['generated_human_report']}`.",
            f"{value['source_of_truth']['live_task_state']}.",
            f"The next foundation decision is `{value['next_issue_id']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        raw_output_path = Path(args.output)
        if not raw_output_path.is_absolute():
            raw_output_path = REPO_ROOT / raw_output_path
        if raw_output_path.is_symlink():
            raise ValueError("output must not be a symbolic link")
        source_path = _resolve_repo_path(args.source, "source")
        output_path = _resolve_repo_path(args.output, "output")
        if source_path == output_path:
            raise ValueError("source and output must be different files")
        if source_path.exists() and output_path.exists() and source_path.samefile(output_path):
            raise ValueError("source and output must not reference the same file")
        projection = load_projection(source_path)
        rendered = render_markdown(projection, source_path=source_path)
    except ValueError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(str(exc), file=sys.stderr)
        return 2

    current = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    drifted = current != rendered
    summary = {
        "check": bool(args.check),
        "drifted": drifted,
        "output": _repo_rel(output_path),
        "phases": len(projection["phases"]),
        "root_issue_id": projection["snapshot"]["root_issue_id"],
        "source": _repo_rel(source_path),
        "snapshot_date": projection["snapshot"]["date"],
        "ok": not drifted,
    }

    if args.check:
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif drifted:
            print(f"adaptive vision roadmap report drifted: {_repo_rel(output_path)}", file=sys.stderr)
        return 1 if drifted else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    summary["ok"] = True
    summary["drifted"] = False
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_repo_rel(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
