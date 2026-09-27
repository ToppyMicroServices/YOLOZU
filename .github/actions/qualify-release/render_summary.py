#!/usr/bin/env python3
"""Verify an Action-created qualification pack and write bounded GitHub outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from yolozu.qualification import verify_qualification_pack


_EXPECTED_STATUS = {"pass": 0, "hold": 3, "fail": 4}


def _read_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": {"message": f"qualification result unavailable: {exc}"},
        }
    return (
        payload
        if isinstance(payload, dict)
        else {
            "ok": False,
            "error": {"message": "qualification result must be a JSON object"},
        }
    )


def _safe_text(value: Any, *, limit: int = 240) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return text[:limit]


def _append_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"unsafe multiline output: {key}")
            handle.write(f"{key}={value}\n")


def _summary(
    payload: dict[str, Any],
    *,
    command_status: int,
) -> tuple[str, str, str, bool, list[dict[str, Any]], str | None]:
    decision = payload.get("decision")
    pack_text = payload.get("pack_dir")
    if decision not in _EXPECTED_STATUS or not isinstance(pack_text, str):
        error = payload.get("error")
        message = (
            error.get("message")
            if isinstance(error, dict)
            else "qualification command failed"
        )
        return "error", "", "", False, [], _safe_text(message)

    pack = Path(pack_text).expanduser().resolve()
    verification = verify_qualification_pack(pack)
    if not verification.ok or verification.pack_digest is None:
        message = "; ".join(verification.errors) or "pack verification failed"
        return "error", "", str(pack), False, [], _safe_text(message)
    if command_status != _EXPECTED_STATUS[decision]:
        return (
            "error",
            verification.pack_digest,
            str(pack),
            True,
            [],
            f"command status {command_status} does not match decision {decision}",
        )

    qualification = payload.get("qualification")
    checks = qualification.get("checks", []) if isinstance(qualification, dict) else []
    if not isinstance(checks, list):
        checks = []
    return decision, verification.pack_digest, str(pack), True, checks, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--command-status", type=int, required=True)
    args = parser.parse_args()

    payload = _read_result(args.result)
    decision, digest, pack_path, pack_exists, checks, error = _summary(
        payload, command_status=args.command_status
    )
    output_path = os.environ.get("GITHUB_OUTPUT")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not output_path or not summary_path:
        raise SystemExit("GITHUB_OUTPUT and GITHUB_STEP_SUMMARY are required")

    _append_output(
        Path(output_path),
        {
            "decision": decision,
            "pack-digest": digest,
            "pack-path": pack_path,
            "pack-exists": "true" if pack_exists else "false",
        },
    )

    lines = [
        "## YOLOZU release qualification",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Decision | **{_safe_text(decision)}** |",
        f"| Pack digest | `{_safe_text(digest or 'unavailable')}` |",
    ]
    if error is not None:
        lines.extend(["", f"**Error:** {_safe_text(error)}"])
    if checks:
        lines.extend(
            [
                "",
                "### Gate results",
                "",
                "| Check | Status | Observed | Threshold |",
                "|---|---|---:|---:|",
            ]
        )
        for check in checks:
            if not isinstance(check, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _safe_text(check.get("id", "unknown")),
                        _safe_text(check.get("status", "unknown")),
                        _safe_text(check.get("observed", "")),
                        _safe_text(check.get("threshold", "")),
                    ]
                )
                + " |"
            )
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
