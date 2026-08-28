"""Read-only qualification expiry and governed-input drift monitor."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .bundle_registry import LoadedAlgorithmBundleRegistry, load_algorithm_bundle_registry
from .bundles import ZERO_DIGEST
from .canonical import canonical_json_v1
from .control_records import load_bounded_json_bytes
from .evidence import (
    MAX_EVIDENCE_ACTIVATION_BYTES,
    EvidenceActivationRecord,
    QualificationReport,
    compute_evidence_selection_key,
    load_evidence_activation_jsonl_bytes,
    validate_evidence_activation_record,
    validate_qualification_report,
)
from .managed_output import ManagedOutputLimits, ManagedOutputTransaction
from .qualification import qualification_report_has_code_owned_issuer

__all__ = [
    "QualificationFreshnessError",
    "check_qualification_freshness",
    "render_qualification_freshness_issue_body",
    "write_qualification_freshness_report",
]


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_CHECKSUM_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_ROWS = 8192


class QualificationFreshnessError(ValueError):
    """One bounded freshness-monitor input or integrity failure."""

    def __init__(self, code: str, detail: str) -> None:
        safe = detail.encode("utf-8", "replace")[:512].decode("utf-8", "ignore")
        super().__init__(f"{code}: {safe}")
        self.code = code


def _fail(code: str, detail: str) -> QualificationFreshnessError:
    return QualificationFreshnessError(code, detail)


def _utc(value: str | datetime | None) -> tuple[str, datetime]:
    if value is None:
        current = datetime.now(timezone.utc).replace(microsecond=0)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ"), current
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise _fail("clock_invalid", "as_of must be timezone-aware UTC")
        current = value.astimezone(timezone.utc).replace(microsecond=0)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ"), current
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise _fail("clock_invalid", "as_of must use exact RFC3339 UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise _fail("clock_invalid", "as_of is not a valid Gregorian instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise _fail("clock_invalid", "as_of is not canonical")
    return value, parsed


def _event_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise _fail("activation_conflict", f"{field} is invalid")
    try:
        return _utc(value)[1]
    except QualificationFreshnessError as exc:
        raise _fail("activation_conflict", f"{field} is invalid") from exc


def _date(value: str) -> str:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise _fail("missed_run_date_invalid", "missed run date must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _fail("missed_run_date_invalid", "missed run date is not Gregorian") from exc
    if parsed.isoformat() != value:
        raise _fail("missed_run_date_invalid", "missed run date is not canonical")
    return value


def _confined_directory(value: str | Path, *, workspace_root: str | Path) -> Path:
    workspace_input = Path(workspace_root)
    workspace_lexical = Path(os.path.abspath(workspace_input))
    if workspace_lexical.is_symlink():
        raise _fail("workspace_invalid", "workspace root cannot be a symlink")
    try:
        workspace = workspace_lexical.resolve(strict=True)
    except OSError as exc:
        raise _fail("workspace_invalid", "workspace root is unavailable") from exc
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace_lexical / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(workspace_lexical)
    except ValueError as exc:
        raise _fail("path_invalid", "input root is outside the workspace") from exc
    current = workspace_lexical
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise _fail("path_invalid", "input root contains a symlink")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise _fail("path_invalid", "input root is unavailable or escapes workspace") from exc
    if not resolved.is_dir():
        raise _fail("path_invalid", "input root must be a directory")
    return resolved


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise _fail("input_unreadable", f"{label} is unavailable") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise _fail("input_unreadable", f"{label} is not one regular file")
    if before.st_size > maximum_bytes:
        raise _fail("input_unreadable", f"{label} exceeds its byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise _fail("input_changed", f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise _fail("input_unreadable", f"{label} exceeds its byte cap")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or total != after.st_size:
            raise _fail("input_changed", f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _packaged_bytes(parts: Sequence[str], *, maximum_bytes: int, label: str) -> bytes:
    item = resources.files("yolozu.data")
    for component in parts:
        item = item.joinpath(component)
    try:
        payload = item.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise _fail("input_unreadable", f"packaged {label} is unavailable") from exc
    if len(payload) > maximum_bytes:
        raise _fail("input_unreadable", f"packaged {label} exceeds its byte cap")
    return payload


def _site_checksum_is_exact(report_bytes: bytes, checksum_bytes: bytes) -> bool:
    try:
        manifest = load_bounded_json_bytes(
            checksum_bytes, label="qualification checksums manifest"
        )
    except ValueError:
        return False
    expected = {
        "path": "qualification_report.json",
        "size_bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    return manifest == {
        "schema_version": 1,
        "files": [expected],
        "expected_paths": ["qualification_report.json"],
        "file_count": 1,
        "total_bytes": len(report_bytes),
    }


def _report_without_freshness(payload: Mapping[str, Any]) -> QualificationReport:
    completed_at = payload.get("completed_at")
    if not isinstance(completed_at, str):
        raise _fail("report_invalid", "qualification report completion is invalid")
    try:
        return validate_qualification_report(payload, as_of=completed_at)
    except ValueError as exc:
        raise _fail("report_invalid", "qualification report is invalid") from exc


@dataclass(frozen=True)
class _LoadedEvidence:
    records: tuple[EvidenceActivationRecord, ...]
    reports: dict[tuple[str, str], QualificationReport]
    trust_domain: str
    root: Path | None


def _load_evidence(
    *, evidence_root: Path | None, workspace_root: Path
) -> _LoadedEvidence:
    if evidence_root is None:
        raw_stream = _packaged_bytes(
            ("adaptive_routing", "evidence_activation.jsonl"),
            maximum_bytes=MAX_EVIDENCE_ACTIVATION_BYTES,
            label="evidence activation stream",
        )
        root = None
        trust = "yolozu_managed"
    else:
        root = _confined_directory(evidence_root, workspace_root=workspace_root)
        raw_stream = _read_regular(
            root / "evidence_activation.jsonl",
            maximum_bytes=MAX_EVIDENCE_ACTIVATION_BYTES,
            label="evidence activation stream",
        )
        trust = "unknown"
    try:
        raw_records = load_evidence_activation_jsonl_bytes(raw_stream)
    except ValueError as exc:
        raise _fail("activation_conflict", "activation stream is invalid") from exc
    if root is not None and raw_records:
        claims = {
            str(record.get("trust_domain"))
            for record in raw_records
            if isinstance(record, Mapping)
        }
        if len(claims) != 1 or next(iter(claims)) not in {
            "site_managed",
            "operator_asserted",
            "unknown",
        }:
            raise _fail("activation_conflict", "evidence trust claims conflict")
        trust = next(iter(claims))
    records: list[EvidenceActivationRecord] = []
    try:
        for raw in raw_records:
            records.append(
                validate_evidence_activation_record(
                    raw, source_trust_domain=trust
                )
            )
    except ValueError as exc:
        raise _fail("activation_conflict", "activation authority or record is invalid") from exc

    identities = sorted(
        {
            (record.to_dict()["report_id"], record.to_dict()["report_digest"])
            for record in records
        },
        key=lambda item: (item[0].encode("ascii"), item[1].encode("ascii")),
    )
    reports: dict[tuple[str, str], QualificationReport] = {}
    for report_id, report_digest in identities:
        if root is None:
            report_bytes = _packaged_bytes(
                (
                    "adaptive_routing",
                    "qualification_reports",
                    report_id,
                    "qualification_report.json",
                ),
                maximum_bytes=_MAX_REPORT_BYTES,
                label="qualification report",
            )
        else:
            report_dir = root / "qualification_reports" / report_id
            report_bytes = _read_regular(
                report_dir / "qualification_report.json",
                maximum_bytes=_MAX_REPORT_BYTES,
                label="qualification report",
            )
            checksum_bytes = _read_regular(
                report_dir / "checksums.json",
                maximum_bytes=_MAX_CHECKSUM_BYTES,
                label="qualification checksums manifest",
            )
        try:
            payload = load_bounded_json_bytes(report_bytes, label="QualificationReport")
        except ValueError as exc:
            raise _fail("report_invalid", "qualification report JSON is invalid") from exc
        if not isinstance(payload, Mapping):
            raise _fail("report_invalid", "qualification report must be an object")
        report = _report_without_freshness(payload)
        if report.report_id != report_id or report.report_digest != report_digest:
            raise _fail("activation_conflict", "activation/report identity mismatch")
        if root is not None and trust == "site_managed" and (
            not _site_checksum_is_exact(report_bytes, checksum_bytes)
            or not qualification_report_has_code_owned_issuer(report)
        ):
            raise _fail("report_invalid", "site report package is not exact and code-owned")
        reports[(report_id, report_digest)] = report
    return _LoadedEvidence(tuple(records), reports, trust, root)


def _row(
    *,
    state: str,
    reason_codes: Iterable[str],
    selection_key: str | None = None,
    report: QualificationReport | None = None,
    valid_until: str | None = None,
    days_remaining: int | None = None,
) -> dict[str, Any]:
    data = report.to_dict() if report is not None else {}
    return {
        "selection_key": selection_key,
        "report_id": data.get("report_id"),
        "report_digest": data.get("report_digest"),
        "bundle_spec_digest": data.get("bundle_spec_digest"),
        "artifact_set_digest": data.get("artifact_set_digest"),
        "valid_until": valid_until,
        "days_remaining": days_remaining,
        "state": state,
        "reason_codes": sorted(set(reason_codes), key=lambda item: item.encode("ascii")),
    }


def _active_reports(
    evidence: _LoadedEvidence, *, current: datetime
) -> list[tuple[str, QualificationReport, str]]:
    streams: dict[str, list[EvidenceActivationRecord]] = {}
    for record in evidence.records:
        data = record.to_dict()
        streams.setdefault(data["selection_key"], []).append(record)
    active_rows: list[tuple[str, QualificationReport, str]] = []
    for selection_key in sorted(streams, key=lambda item: item.encode("ascii")):
        previous = ZERO_DIGEST
        previous_time: datetime | None = None
        active: tuple[str, str] | None = None
        active_valid_until: str | None = None
        retired: set[tuple[str, str]] = set()
        pending: tuple[str, str] | None = None
        terminal = False
        for expected_sequence, record in enumerate(streams[selection_key], start=1):
            data = record.to_dict()
            if terminal:
                raise _fail("activation_conflict", "activation continues after revoke")
            if data["sequence"] != expected_sequence or data["previous_event_digest"] != previous:
                raise _fail("activation_conflict", "activation stream has a gap or fork")
            activated = _event_utc(data["activated_at"], field="activated_at")
            valid = _event_utc(data["valid_until"], field="valid_until")
            if activated > current:
                raise _fail("clock_invalid", "activation event is future-dated")
            if previous_time is not None and activated < previous_time:
                raise _fail("activation_conflict", "activation time reverses sequence")
            identity = (data["report_id"], data["report_digest"])
            report = evidence.reports.get(identity)
            if report is None:
                raise _fail("activation_conflict", "activation report is missing")
            report_data = report.to_dict()
            computed_key = compute_evidence_selection_key(
                bundle_spec_digest=report_data["bundle_spec_digest"],
                artifact_set_digest=report_data["artifact_set_digest"],
                environment_fingerprint=report_data["environment_fingerprint"],
                qualification_workload_fingerprint=report_data[
                    "qualification_workload_fingerprint"
                ],
                protocol_fingerprint=report_data["protocol_fingerprint"],
            )
            if computed_key != selection_key:
                raise _fail("activation_conflict", "activation selection key mismatch")
            completed = _event_utc(report_data["completed_at"], field="completed_at")
            report_valid = _event_utc(report_data["valid_until"], field="valid_until")
            if not completed <= activated < valid <= report_valid:
                raise _fail("activation_conflict", "activation validity is inconsistent")
            if data["state"] == "active":
                if report_data["status"] != "qualified" or active is not None:
                    raise _fail("activation_conflict", "active report state conflicts")
                if identity in retired or (pending is not None and pending != identity):
                    raise _fail("activation_conflict", "retired or wrong replacement reactivated")
                active = identity
                active_valid_until = data["valid_until"]
                pending = None
            elif data["state"] == "superseded":
                if active != identity:
                    raise _fail("activation_conflict", "supersession target is not active")
                retired.add(identity)
                active = None
                active_valid_until = None
                pending = (
                    data["replacement_report_id"],
                    data["replacement_report_digest"],
                )
            else:
                if active != identity:
                    raise _fail("activation_conflict", "revocation target is not active")
                retired.add(identity)
                active = None
                active_valid_until = None
                pending = None
                terminal = True
            previous = data["event_digest"]
            previous_time = activated
        if pending is not None:
            raise _fail("activation_conflict", "activation ends with dangling supersession")
        if active is not None and active_valid_until is not None:
            active_rows.append(
                (selection_key, evidence.reports[active], active_valid_until)
            )
    return active_rows


def _drift_reasons(
    report: QualificationReport, registry: LoadedAlgorithmBundleRegistry
) -> tuple[str | None, list[str]]:
    data = report.to_dict()
    bundle = registry.by_spec_digest().get(data["bundle_spec_digest"])
    if bundle is None:
        return "artifact_or_bundle_drift", ["bundle_missing"]
    bundle_data = bundle.to_dict()
    artifact_reasons: list[str] = []
    if bundle.artifact_set_digest != data["artifact_set_digest"]:
        artifact_reasons.append("artifact_set_changed")
    if bundle_data.get("model_revision") != data["source_runtime_provenance"][
        "model_revision"
    ]:
        artifact_reasons.append("model_revision_changed")
    lifecycle = registry.lifecycle.bundle_states.get(data["bundle_spec_digest"])
    if lifecycle is None:
        artifact_reasons.append("lifecycle_missing")
    else:
        if lifecycle["bundle_state"] != "enabled":
            artifact_reasons.append("bundle_disabled_or_revoked")
        if any(
            review["review_state"] != "approved"
            for review in lifecycle["artifact_license_reviews"]
        ):
            artifact_reasons.append("license_not_approved")
    if artifact_reasons:
        return "artifact_or_bundle_drift", artifact_reasons

    runtime = bundle_data.get("runtime")
    provenance = data["source_runtime_provenance"]
    runtime_reasons: list[str] = []
    if not isinstance(runtime, Mapping):
        runtime_reasons.append("runtime_binding_missing")
    else:
        for field in (
            "runtime_id",
            "runtime_version",
            "provider_id",
            "provider_version",
        ):
            if runtime.get(field) != provenance[field]:
                runtime_reasons.append(f"{field}_changed")
    if runtime_reasons:
        return "runtime_drift", runtime_reasons
    return None, []


def check_qualification_freshness(
    *,
    workspace_root: str | Path = ".",
    evidence_root: str | Path | None = None,
    registry_root: str | Path | None = None,
    as_of: str | datetime | None = None,
    missed_run_dates: Iterable[str] = (),
    registry: LoadedAlgorithmBundleRegistry | None = None,
) -> dict[str, Any]:
    """Return a bounded aggregate report without mutating governed state."""

    collected_at, current = _utc(as_of)
    missed = tuple(_date(value) for value in missed_run_dates)
    if len(missed) > 8 or len(set(missed)) != len(missed):
        raise _fail("missed_run_date_invalid", "missed run dates exceed bounds or repeat")
    today = current.date().isoformat()
    if any(value >= today for value in missed):
        raise _fail("missed_run_date_invalid", "missed run dates must precede this run")
    workspace = Path(workspace_root)
    try:
        evidence = _load_evidence(
            evidence_root=None if evidence_root is None else Path(evidence_root),
            workspace_root=workspace,
        )
    except QualificationFreshnessError as exc:
        state = "unknown" if exc.code in {"input_unreadable", "report_invalid"} else "conflict"
        rows = [_row(state=state, reason_codes=(exc.code,))]
        trust_domain = "unknown"
        source_scope = "site_confined_local_only" if evidence_root is not None else "repository_owned_public_ids_only"
    else:
        trust_domain = evidence.trust_domain
        source_scope = (
            "repository_owned_public_ids_only"
            if evidence.root is None
            else "site_confined_local_only"
        )
        try:
            loaded_registry = registry
            if loaded_registry is None:
                loaded_registry = load_algorithm_bundle_registry(
                    workspace_root=workspace if registry_root is not None else None,
                    custom_registry_root=(
                        None if registry_root is None else Path(registry_root)
                    ),
                )
            active = _active_reports(evidence, current=current)
            rows = []
            for selection_key, report, valid_until in active:
                valid = _event_utc(valid_until, field="valid_until")
                remaining_seconds = int((valid - current).total_seconds())
                days_remaining = max(0, remaining_seconds // 86400)
                drift_state, reasons = _drift_reasons(report, loaded_registry)
                if drift_state is not None:
                    state = drift_state
                elif remaining_seconds <= 0:
                    state = "expired"
                    reasons = ["evidence_expired"]
                    days_remaining = 0
                elif remaining_seconds <= 7 * 86400:
                    state = "due_7"
                    reasons = ["evidence_due_within_7_days"]
                elif remaining_seconds <= 14 * 86400:
                    state = "due_14"
                    reasons = ["evidence_due_within_14_days"]
                elif remaining_seconds <= 30 * 86400:
                    state = "due_30"
                    reasons = ["evidence_due_within_30_days"]
                else:
                    state = "ok"
                    reasons = []
                rows.append(
                    _row(
                        selection_key=selection_key,
                        report=report,
                        valid_until=valid_until,
                        days_remaining=days_remaining,
                        state=state,
                        reason_codes=reasons,
                    )
                )
        except (OSError, TypeError, ValueError, QualificationFreshnessError) as exc:
            code = getattr(exc, "code", "activation_conflict")
            state = "unknown" if code in {"clock_invalid", "input_unreadable", "report_invalid"} else "conflict"
            rows = [_row(state=state, reason_codes=(str(code),))]
    if len(rows) > _MAX_ROWS:
        rows = [_row(state="unknown", reason_codes=("row_limit_exceeded",))]
    counts = Counter(row["state"] for row in rows)
    report = {
        "kind": "qualification_freshness_report",
        "schema_version": 1,
        "collected_at": collected_at,
        "timezone": "UTC",
        "scheduled_timezone": "Asia/Tokyo",
        "source_trust_domain": trust_domain,
        "source_scope": source_scope,
        "upload_eligible": source_scope == "repository_owned_public_ids_only",
        "missed_expected_run_dates": list(sorted(missed)),
        "rows": rows,
        "summary": {
            "active_or_problem_rows": len(rows),
            "action_required_rows": sum(row["state"] != "ok" for row in rows),
            "state_counts": {
                state: counts.get(state, 0)
                for state in (
                    "ok",
                    "due_30",
                    "due_14",
                    "due_7",
                    "expired",
                    "runtime_drift",
                    "artifact_or_bundle_drift",
                    "conflict",
                    "unknown",
                )
            },
        },
        "retention": {
            "repository_artifact_days": 30,
            "site_output": "user_owned_local_only",
            "raw_performance_or_host_data": "not_collected",
        },
        "automation_caveat": (
            "This read-only warning does not qualify, extend, activate, promote, "
            "revoke, mutate Beads, or prove a performance regression."
        ),
    }
    if len(canonical_json_v1(report)) > _MAX_OUTPUT_BYTES:
        raise _fail("output_limit", "freshness report exceeds its byte cap")
    return report


def _relative_output(value: str | Path) -> str:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or raw.endswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} or part.startswith(".") for part in path.parts)
        or len(raw.encode("utf-8")) > 4096
    ):
        raise _fail("output_path_invalid", "output must be a visible relative directory")
    return path.as_posix()


def _ensure_parent(root: Path, relative: str) -> None:
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise _fail("output_path_invalid", "output parent is unsafe")
        else:
            current.mkdir()


def write_qualification_freshness_report(
    report: Mapping[str, Any], *, workspace_root: str | Path, output: str | Path
) -> Path:
    """Write only the explicit managed report destination."""

    workspace = _confined_directory(workspace_root, workspace_root=workspace_root)
    destination = _relative_output(output)
    encoded = canonical_json_v1(report)
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise _fail("output_limit", "freshness report exceeds its byte cap")
    _ensure_parent(workspace, destination)
    with ManagedOutputTransaction(
        root=workspace,
        destination=destination,
        declared_paths=("qualification_freshness_report.json",),
        limits=ManagedOutputLimits(
            max_files=2,
            max_file_bytes=_MAX_OUTPUT_BYTES,
            max_total_bytes=_MAX_OUTPUT_BYTES + _MAX_CHECKSUM_BYTES,
        ),
        force=True,
    ) as transaction:
        transaction.write_bytes("qualification_freshness_report.json", encoded)
        transaction.commit()
    return workspace / destination / "qualification_freshness_report.json"


def render_qualification_freshness_issue_body(report: Mapping[str, Any]) -> str:
    """Render at most 64 action rows containing only public control identifiers."""

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise _fail("report_invalid", "freshness rows are invalid")
    action_rows = [row for row in rows if isinstance(row, Mapping) and row.get("state") != "ok"]
    lines = [
        f"Collection time: {report.get('collected_at', 'unknown')}",
        f"Action rows: {len(action_rows)}",
        "",
    ]
    for row in action_rows[:64]:
        selection_key = row.get("selection_key")
        report_id = row.get("report_id")
        state = row.get("state")
        reasons = row.get("reason_codes")
        if selection_key is not None and (
            not isinstance(selection_key, str) or _SHA256_RE.fullmatch(selection_key) is None
        ):
            raise _fail("report_invalid", "issue selection key is invalid")
        if report_id is not None and (
            not isinstance(report_id, str) or _ID_RE.fullmatch(report_id) is None
        ):
            raise _fail("report_invalid", "issue report ID is invalid")
        if not isinstance(state, str) or not isinstance(reasons, list):
            raise _fail("report_invalid", "issue row is invalid")
        lines.append(
            "- selection_key={}; report_id={}; state={}; reasons={}".format(
                selection_key or "unknown",
                report_id or "unknown",
                state,
                ",".join(str(value) for value in reasons) or "none",
            )
        )
    if len(action_rows) > 64:
        lines.append(f"- additional_rows_omitted={len(action_rows) - 64}")
    lines.extend(
        [
            "",
            "This bounded public notice contains no paths, prompts, datasets, host facts, or raw telemetry.",
        ]
    )
    body = "\n".join(lines)
    if len(body.encode("utf-8")) > 64 * 1024:
        raise _fail("output_limit", "freshness issue body exceeds its byte cap")
    return body
