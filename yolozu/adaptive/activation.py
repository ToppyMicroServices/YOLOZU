"""Reviewed activation of exact adaptive-routing qualification evidence.

Qualification reports remain inert until this module appends a reviewed event to
the canonical per-selection-key stream.  Source trust is derived from the file
and workflow boundary; it is never accepted from a report field.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from .bundle_registry import LoadedAlgorithmBundleRegistry, load_algorithm_bundle_registry
from .bundles import ZERO_DIGEST
from .canonical import canonical_json_v1, canonical_sha256_v1
from .control_records import load_bounded_json_bytes
from .evidence import (
    MAX_EVIDENCE_ACTIVATION_BYTES,
    QualificationReport,
    compute_evidence_selection_key,
    load_evidence_activation_jsonl_bytes,
    project_evidence_activations,
    validate_evidence_activation_record,
    validate_qualification_report,
)
from .qualification import qualification_report_has_code_owned_issuer

__all__ = [
    "EvidenceActivationOutcome",
    "activate_qualification_evidence",
]


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_REPRODUCTION_BYTES = 4096
_REVIEWED_REPORT_ROOT = Path(
    "yolozu/data/adaptive_routing/qualification_reports"
)
_CANONICAL_STREAM = Path("yolozu/data/adaptive_routing/evidence_activation.jsonl")

FaultHook = Callable[[str], None]
TrustDomain = Literal[
    "yolozu_managed", "site_managed", "operator_asserted", "unknown"
]


@dataclass(frozen=True)
class _Gate:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class EvidenceActivationOutcome:
    """Machine-readable dry-run or mutation outcome."""

    status: Literal["dry_run_ready", "dry_run_blocked", "applied", "apply_failed"]
    operation: str
    approved: bool
    source_trust_domain: TrustDomain
    support_scope: Literal["public_qualified", "site_qualified", "none"]
    selection_key: str | None
    observed_head_digest: str
    observed_current_activation_id: str | None
    gates: tuple[_Gate, ...]
    planned_records: tuple[dict[str, Any], ...]
    applied_record_digests: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "evidence_activation_outcome",
            "status": self.status,
            "operation": self.operation,
            "approved": self.approved,
            "source_trust_domain": self.source_trust_domain,
            "support_scope": self.support_scope,
            "selection_key": self.selection_key,
            "observed_head_digest": self.observed_head_digest,
            "observed_current_activation_id": self.observed_current_activation_id,
            "gates": [gate.to_dict() for gate in self.gates],
            "planned_records": [dict(record) for record in self.planned_records],
            "applied_record_digests": list(self.applied_record_digests),
            "integrity_caveat": (
                "Hashes protect integrity after creation; they do not prove provenance "
                "against an adversarial local operator."
            ),
        }


def _gate(gates: list[_Gate], code: str, detail: str) -> None:
    if not any(item.code == code for item in gates):
        gates.append(_Gate(code, detail))


def _utc(value: str | datetime | None) -> tuple[str, datetime]:
    if value is None:
        current = datetime.now(timezone.utc).replace(microsecond=0)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ"), current
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("as_of must be timezone-aware UTC")
        current = value.astimezone(timezone.utc).replace(microsecond=0)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ"), current
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError("as_of must use exact RFC3339 UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("as_of is not a valid Gregorian UTC instant") from exc
    return value, parsed


def _confined_path(
    value: str | Path,
    *,
    workspace_root: Path,
    must_exist: bool,
    expected_file: bool,
) -> Path:
    workspace_lexical = Path(os.path.abspath(workspace_root))
    if workspace_lexical.is_symlink():
        raise ValueError("workspace root cannot be a symlink")
    workspace = workspace_lexical.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace root must be a directory")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace_lexical / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(workspace_lexical)
    except ValueError:
        # macOS exposes /var through /private/var.  Compare resolved paths below,
        # but retain strict component checks whenever both spellings share a root.
        relative = None
    if relative is not None:
        current = workspace_lexical
        parts = relative.parts if must_exist else relative.parts[:-1]
        for component in parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("path contains a symlink component")
    if must_exist:
        resolved = lexical.resolve(strict=True)
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("path resolves outside the workspace") from exc
        info = os.stat(resolved, follow_symlinks=False)
        if expected_file and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError("path must be one singly linked regular file")
        return resolved
    parent = lexical.parent.resolve(strict=True)
    try:
        parent.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("path parent resolves outside the workspace") from exc
    if not parent.is_dir():
        raise ValueError("path parent must be a directory")
    return parent / lexical.name


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be one singly linked regular file")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if total != after.st_size or identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _qualifier_checksum_is_exact(report_path: Path, report_bytes: bytes) -> bool:
    manifest_path = report_path.parent / "checksums.json"
    try:
        manifest_bytes = _read_regular(
            manifest_path, maximum_bytes=4 * 1024 * 1024, label="checksums.json"
        )
        manifest = load_bounded_json_bytes(
            manifest_bytes, label="qualification checksums manifest"
        )
    except (OSError, ValueError):
        return False
    expected_entry = {
        "path": "qualification_report.json",
        "size_bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    return manifest == {
        "schema_version": 1,
        "files": [expected_entry],
        "expected_paths": ["qualification_report.json"],
        "file_count": 1,
        "total_bytes": len(report_bytes),
    }


def _repository_material_is_exact(
    *, report_path: Path, report_bytes: bytes
) -> bool:
    required = (
        "protocol.json",
        "public_inputs.json",
        "qualification_report.json",
        "reproduce.txt",
    )
    files: dict[str, bytes] = {"qualification_report.json": report_bytes}
    try:
        for name in required:
            if name not in files:
                files[name] = _read_regular(
                    report_path.parent / name,
                    maximum_bytes=(
                        _MAX_REPRODUCTION_BYTES
                        if name == "reproduce.txt"
                        else _MAX_REPORT_BYTES
                    ),
                    label=name,
                )
        manifest_bytes = _read_regular(
            report_path.parent / "checksums.json",
            maximum_bytes=4 * 1024 * 1024,
            label="checksums.json",
        )
        manifest = load_bounded_json_bytes(
            manifest_bytes, label="repository qualification checksums manifest"
        )
        report = load_bounded_json_bytes(report_bytes, label="QualificationReport")
        protocol = load_bounded_json_bytes(files["protocol.json"], label="protocol")
        public_inputs = load_bounded_json_bytes(
            files["public_inputs.json"], label="public inputs"
        )
        reproduction_text = files["reproduce.txt"].decode("utf-8")
        public_inputs_digest = canonical_sha256_v1(
            public_inputs, own_digest_field="input_set_digest"
        )
        protocol_digest = canonical_sha256_v1(protocol)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    if (
        not isinstance(report, Mapping)
        or report.get("report_id") != report_path.parent.name
        or not isinstance(public_inputs, Mapping)
    ):
        return False
    if (
        public_inputs.get("schema_version") != 1
        or isinstance(public_inputs.get("schema_version"), bool)
        or public_inputs.get("privacy_class") != "public_non_sensitive"
        or set(public_inputs) != {
            "schema_version",
            "privacy_class",
            "input_set_id",
            "input_set_digest",
            "source_references",
        }
        or not isinstance(public_inputs.get("source_references"), list)
    ):
        return False
    if (
        not isinstance(public_inputs.get("input_set_id"), str)
        or _COMPONENT_RE.fullmatch(str(public_inputs["input_set_id"])) is None
        or not isinstance(public_inputs.get("input_set_digest"), str)
        or _SHA256_RE.fullmatch(str(public_inputs["input_set_digest"])) is None
        or public_inputs["input_set_digest"]
        != public_inputs_digest
        or not 1 <= len(public_inputs["source_references"]) <= 128
        or any(
            not isinstance(reference, str)
            or not reference
            or len(reference.encode("utf-8")) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in reference)
            for reference in public_inputs["source_references"]
        )
    ):
        return False
    if protocol_digest != report.get("protocol_fingerprint"):
        return False
    if (
        not reproduction_text.endswith("\n")
        or not reproduction_text.startswith("yolozu qualify-image-pipeline ")
        or any(
            ord(character) < 32 and character not in {"\n", "\t"}
            for character in reproduction_text
        )
    ):
        return False
    entries = [
        {
            "path": name,
            "size_bytes": len(files[name]),
            "sha256": hashlib.sha256(files[name]).hexdigest(),
        }
        for name in required
    ]
    return manifest == {
        "schema_version": 1,
        "files": entries,
        "expected_paths": list(required),
        "file_count": len(required),
        "total_bytes": sum(len(files[name]) for name in required),
    }


def _git_tracked_clean(root: Path, paths: Sequence[Path]) -> bool:
    relative: list[str] = []
    try:
        for path in paths:
            relative.append(path.relative_to(root).as_posix())
    except ValueError:
        return False
    try:
        for item in relative:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", item],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            if tracked.returncode != 0:
                return False
        clean = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return clean.returncode == 0


def _repository_retention_is_exact(
    *, report_path: Path, stream_path: Path, workspace: Path
) -> bool:
    try:
        relative_report = report_path.relative_to(workspace)
        relative_stream = stream_path.relative_to(workspace)
    except ValueError:
        return False
    if relative_stream != _CANONICAL_STREAM:
        return False
    try:
        retained = relative_report.relative_to(_REVIEWED_REPORT_ROOT)
    except ValueError:
        return False
    if (
        len(retained.parts) != 2
        or retained.parts[1] != "qualification_report.json"
        or _COMPONENT_RE.fullmatch(retained.parts[0]) is None
    ):
        return False
    return _git_tracked_clean(
        workspace,
        (
            report_path,
            report_path.parent / "checksums.json",
            report_path.parent / "protocol.json",
            report_path.parent / "public_inputs.json",
            report_path.parent / "reproduce.txt",
            stream_path,
        ),
    )


def _derive_source(
    *,
    report_path: Path,
    report_bytes: bytes,
    stream_path: Path,
    workspace: Path,
) -> tuple[TrustDomain, Literal["public_qualified", "site_qualified", "none"]]:
    repository_material = _repository_material_is_exact(
        report_path=report_path, report_bytes=report_bytes
    )
    if repository_material and _repository_retention_is_exact(
        report_path=report_path, stream_path=stream_path, workspace=workspace
    ):
        return "yolozu_managed", "public_qualified"
    checksum_ok = _qualifier_checksum_is_exact(report_path, report_bytes)
    if checksum_ok and stream_path != workspace / _CANONICAL_STREAM:
        return "site_managed", "site_qualified"
    return "operator_asserted", "none"


def _current_active_event(
    records: Sequence[Mapping[str, Any]], selection_key: str
) -> Mapping[str, Any] | None:
    current: Mapping[str, Any] | None = None
    for record in records:
        if record.get("selection_key") != selection_key:
            continue
        state = record.get("state")
        if state == "active":
            current = record
        elif state in {"superseded", "revoked"}:
            current = None
    return current


def _event(
    *,
    report: Mapping[str, Any],
    selection_key: str,
    sequence: int,
    previous: str,
    state: Literal["active", "superseded", "revoked"],
    activated_at: str,
    valid_until: str,
    reviewer_role_id: str,
    review_reference: Mapping[str, Any],
    trust_domain: TrustDomain,
    reason: str,
    replacement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issuer_claim = {
        "yolozu_managed": "repository_source",
        "site_managed": "site_source",
        "operator_asserted": "operator_source",
        "unknown": "unknown",
    }[trust_domain]
    value: dict[str, Any] = {
        "schema_version": 1,
        "stream_id": selection_key,
        "selection_key": selection_key,
        "sequence": sequence,
        "previous_event_digest": previous,
        "event_id": f"activation-{secrets.token_hex(16)}",
        "report_id": report["report_id"],
        "report_digest": report["report_digest"],
        "state": state,
        "replacement_report_id": None,
        "replacement_report_digest": None,
        "activated_at": activated_at,
        "valid_until": valid_until,
        "reviewer_role_id": reviewer_role_id,
        "review_reference": dict(review_reference),
        "issuer_claim": issuer_claim,
        "trust_domain": trust_domain,
        "reason": reason,
        "event_digest": ZERO_DIGEST,
    }
    if replacement is not None:
        value["replacement_report_id"] = replacement["report_id"]
        value["replacement_report_digest"] = replacement["report_digest"]
    value["event_digest"] = canonical_sha256_v1(
        value, own_digest_field="event_digest"
    )
    return value


def _atomic_replace_stream(
    *,
    path: Path,
    observed_bytes: bytes,
    replacement_bytes: bytes,
    fault_hook: FaultHook | None,
) -> None:
    if len(replacement_bytes) > MAX_EVIDENCE_ACTIVATION_BYTES:
        raise ValueError("evidence activation stream exceeds 64 MiB")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if os.name != "posix" or not nofollow or not directory_flag:
        raise ValueError("atomic activation requires POSIX no-follow primitives")
    parent_fd = os.open(path.parent, os.O_RDONLY | directory_flag | nofollow)
    temporary = f".{path.name}.stage.{secrets.token_hex(16)}"
    descriptor: int | None = None
    published = False
    try:
        try:
            before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        if before is not None and (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
        ):
            raise ValueError("activation stream must be one singly linked regular file")
        current = b"" if before is None else _read_regular(
            path, maximum_bytes=MAX_EVIDENCE_ACTIVATION_BYTES, label="activation stream"
        )
        if current != observed_bytes:
            raise ValueError("activation stream changed after dry-run validation")
        if fault_hook is not None:
            fault_hook("before_stage_open")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=parent_fd,
        )
        written = 0
        while written < len(replacement_bytes):
            count = os.write(descriptor, replacement_bytes[written:])
            if count <= 0:
                raise OSError("short activation-stream write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if fault_hook is not None:
            fault_hook("before_replace")
        try:
            latest = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            latest = None
        if (before is None) != (latest is None):
            raise ValueError("activation stream changed before commit")
        if before is not None and latest is not None and (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            latest.st_dev,
            latest.st_ino,
            latest.st_size,
            latest.st_mtime_ns,
        ):
            raise ValueError("activation stream identity changed before commit")
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        published = True
        os.fsync(parent_fd)
        if fault_hook is not None:
            fault_hook("after_replace")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            try:
                info = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                info = None
            if info is not None and stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def activate_qualification_evidence(
    *,
    operation: str,
    report_path: str | Path | None,
    report_id: str | None,
    report_digest: str | None,
    selection_key: str | None,
    stream_path: str | Path,
    workspace_root: str | Path,
    expected_head_digest: str | None,
    expected_current_activation_id: str | None,
    reviewer_role_id: str | None,
    reason: str | None,
    approve: bool = False,
    public_review_id: str | None = None,
    site_local_review_present: bool = False,
    supersede_activation_id: str | None = None,
    revoke_activation_id: str | None = None,
    prior_report_paths: Iterable[str | Path] = (),
    as_of: str | datetime | None = None,
    registry: LoadedAlgorithmBundleRegistry | None = None,
    fault_hook: FaultHook | None = None,
) -> EvidenceActivationOutcome:
    """Dry-run or apply one exact reviewed evidence activation transition."""

    gates: list[_Gate] = []
    planned: list[dict[str, Any]] = []
    observed_head = ZERO_DIGEST
    observed_current_id: str | None = None
    trust: TrustDomain = "operator_asserted"
    support_scope: Literal["public_qualified", "site_qualified", "none"] = "none"
    computed_key: str | None = None
    raw_stream = b""
    records: list[Mapping[str, Any]] = []
    reports: list[Mapping[str, Any]] = []
    validated_report: QualificationReport | None = None
    stream: Path | None = None

    if operation not in {"activate", "supersede", "revoke"}:
        _gate(gates, "operation_invalid", "operation must be activate, supersede, or revoke")
    try:
        now_text, now = _utc(as_of)
    except ValueError as exc:
        _gate(gates, "as_of_invalid", str(exc))
        now = datetime.now(timezone.utc).replace(microsecond=0)
        now_text = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        workspace = _confined_path(
            Path(workspace_root),
            workspace_root=Path(workspace_root),
            must_exist=True,
            expected_file=False,
        )
    except (OSError, ValueError) as exc:
        _gate(gates, "workspace_invalid", str(exc))
        workspace = Path(workspace_root).absolute()

    try:
        stream_candidate = Path(stream_path)
        stream_exists = (workspace / stream_candidate).exists() if not stream_candidate.is_absolute() else stream_candidate.exists()
        stream = _confined_path(
            stream_candidate,
            workspace_root=workspace,
            must_exist=stream_exists,
            expected_file=True,
        )
        if stream_exists:
            raw_stream = _read_regular(
                stream,
                maximum_bytes=MAX_EVIDENCE_ACTIVATION_BYTES,
                label="activation stream",
            )
            records = load_evidence_activation_jsonl_bytes(raw_stream)
    except (OSError, ValueError) as exc:
        _gate(gates, "activation_stream_invalid", str(exc))

    report_bytes: bytes | None = None
    report_file: Path | None = None
    if report_path is None:
        _gate(gates, "report_path_missing", "an exact qualification report path is required")
    else:
        try:
            report_file = _confined_path(
                report_path,
                workspace_root=workspace,
                must_exist=True,
                expected_file=True,
            )
            report_bytes = _read_regular(
                report_file, maximum_bytes=_MAX_REPORT_BYTES, label="qualification report"
            )
            payload = load_bounded_json_bytes(report_bytes, label="QualificationReport")
            validated_report = validate_qualification_report(payload, as_of=now)
            reports.append(validated_report.to_dict())
        except (OSError, ValueError) as exc:
            _gate(gates, "report_invalid", str(exc))

    for index, prior_path in enumerate(prior_report_paths):
        try:
            prior_file = _confined_path(
                prior_path,
                workspace_root=workspace,
                must_exist=True,
                expected_file=True,
            )
            prior_bytes = _read_regular(
                prior_file,
                maximum_bytes=_MAX_REPORT_BYTES,
                label=f"prior qualification report {index + 1}",
            )
            prior_payload = load_bounded_json_bytes(
                prior_bytes, label=f"prior QualificationReport {index + 1}"
            )
            reports.append(validate_qualification_report(prior_payload, as_of=now).to_dict())
        except (OSError, ValueError) as exc:
            _gate(gates, "prior_report_invalid", f"prior report {index + 1}: {exc}")

    if report_file is not None and report_bytes is not None and stream is not None:
        trust, support_scope = _derive_source(
            report_path=report_file,
            report_bytes=report_bytes,
            stream_path=stream,
            workspace=workspace,
        )
        if trust == "operator_asserted":
            _gate(
                gates,
                "source_not_managed",
                "arbitrary workspace JSON is operator_asserted and non-selectable",
            )
    if validated_report is not None:
        report = validated_report.to_dict()
        if report["status"] != "qualified":
            _gate(
                gates,
                "report_not_qualified",
                "smoke, hold, and failed reports cannot become active",
            )
        if not qualification_report_has_code_owned_issuer(validated_report):
            _gate(gates, "issuer_unknown", "report issuer/collector is not the code-owned qualifier workflow")
        if report_id is None:
            _gate(gates, "report_id_missing", "the exact report ID is required")
        elif report_id != report["report_id"]:
            _gate(gates, "report_id_mismatch", "supplied report ID does not match the report")
        if report_digest is None:
            _gate(gates, "report_digest_missing", "the exact report digest is required")
        elif report_digest != report["report_digest"]:
            _gate(gates, "report_digest_mismatch", "supplied report digest does not match the report")
        computed_key = compute_evidence_selection_key(
            bundle_spec_digest=report["bundle_spec_digest"],
            artifact_set_digest=report["artifact_set_digest"],
            environment_fingerprint=report["environment_fingerprint"],
            qualification_workload_fingerprint=report[
                "qualification_workload_fingerprint"
            ],
            protocol_fingerprint=report["protocol_fingerprint"],
        )
        if selection_key is None:
            _gate(gates, "selection_key_missing", "the exact selection key is required")
        elif selection_key != computed_key:
            _gate(gates, "selection_key_mismatch", "supplied selection key does not match the report")
    else:
        if report_id is None:
            _gate(gates, "report_id_missing", "the exact report ID is required")
        if report_digest is None:
            _gate(gates, "report_digest_missing", "the exact report digest is required")
        if selection_key is None:
            _gate(gates, "selection_key_missing", "the exact selection key is required")

    if expected_head_digest is None:
        _gate(gates, "expected_head_missing", "the expected stream head digest is required")
    elif _SHA256_RE.fullmatch(expected_head_digest) is None:
        _gate(gates, "expected_head_invalid", "expected head must be lowercase SHA-256")
    if expected_current_activation_id is None:
        _gate(gates, "expected_current_missing", "expected current activation ID is required; use none for zero-active")
    elif expected_current_activation_id != "none" and _COMPONENT_RE.fullmatch(expected_current_activation_id) is None:
        _gate(gates, "expected_current_invalid", "expected current activation ID is invalid")
    if reviewer_role_id is None:
        _gate(gates, "reviewer_role_missing", "a non-personal reviewer role ID is required")
    if reason is None:
        _gate(gates, "reason_missing", "a bounded review reason is required")
    elif not reason or len(reason.encode("utf-8")) > 512:
        _gate(gates, "reason_invalid", "reason must contain 1..512 UTF-8 bytes")

    review_reference: dict[str, Any]
    if trust == "yolozu_managed":
        if public_review_id is None:
            _gate(gates, "public_review_missing", "repository-managed activation requires a public review ID")
            review_reference = {"kind": "public_repository_id", "value": "missing"}
        else:
            review_reference = {"kind": "public_repository_id", "value": public_review_id}
        if site_local_review_present:
            _gate(gates, "review_reference_conflict", "repository review cannot also claim site-local review")
    else:
        review_reference = {
            "kind": "site_local_status",
            "status": "present" if site_local_review_present else "not_applicable",
        }
        if public_review_id is not None:
            _gate(gates, "review_reference_conflict", "site/workspace evidence cannot self-assign repository review")
        if trust == "site_managed" and not site_local_review_present:
            _gate(gates, "site_review_missing", "site-managed activation requires explicit local review status")

    loaded = registry
    if loaded is None:
        try:
            loaded = load_algorithm_bundle_registry()
        except (OSError, ValueError) as exc:
            _gate(gates, "registry_invalid", str(exc))
    if loaded is not None:
        if (
            loaded.registry_trust_domain != "yolozu_managed"
            or loaded.lifecycle_trust_domain != "yolozu_managed"
            or loaded.source_kind != "packaged_ssot"
        ):
            _gate(gates, "registry_untrusted", "activation requires the current canonical managed registry/lifecycle")
        for lifecycle_event in loaded.lifecycle.events:
            try:
                occurred = _utc(lifecycle_event.to_dict()["occurred_at"])[1]
                if occurred > now:
                    _gate(gates, "lifecycle_future", "current lifecycle contains a future-dated event")
            except (KeyError, ValueError) as exc:
                _gate(gates, "lifecycle_time_invalid", str(exc))
        if validated_report is not None:
            report = validated_report.to_dict()
            bundle = loaded.by_spec_digest().get(report["bundle_spec_digest"])
            if bundle is None:
                _gate(gates, "bundle_unknown", "report bundle is absent from the current canonical registry")
            elif bundle.artifact_set_digest != report["artifact_set_digest"]:
                _gate(gates, "artifact_set_mismatch", "report artifact set does not match the current bundle")
            state = loaded.lifecycle.bundle_states.get(report["bundle_spec_digest"])
            if state is None:
                _gate(gates, "lifecycle_missing", "bundle has no current lifecycle state")
            else:
                if state["bundle_state"] != "enabled":
                    _gate(gates, "bundle_blocked", "bundle is currently disabled or revoked")
                if any(
                    review["review_state"] != "approved"
                    for review in state["artifact_license_reviews"]
                ):
                    _gate(gates, "license_not_approved", "one or more current artifact license reviews are not approved")

    if computed_key is not None and not any(
        gate.code in {"activation_stream_invalid", "prior_report_invalid"} for gate in gates
    ):
        try:
            projection = project_evidence_activations(
                records,
                reports,
                source_trust_domain=trust,
                as_of=now,
            )
            observed_head = projection.head_by_selection_key.get(computed_key, ZERO_DIGEST)
            current_event = _current_active_event(records, computed_key)
            observed_current_id = None if current_event is None else str(current_event["event_id"])
        except ValueError as exc:
            _gate(gates, "activation_stream_conflict", str(exc))

    if expected_head_digest is not None and _SHA256_RE.fullmatch(expected_head_digest):
        if expected_head_digest != observed_head:
            _gate(gates, "stale_head", "expected stream head does not match the observed head")
    expected_current = None if expected_current_activation_id == "none" else expected_current_activation_id
    if expected_current_activation_id is not None and expected_current != observed_current_id:
        _gate(gates, "stale_current", "expected current activation ID does not match the observed current activation")

    if operation == "activate":
        if observed_current_id is not None:
            _gate(gates, "active_exists", "activate requires a zero-active selection key")
        if supersede_activation_id is not None or revoke_activation_id is not None:
            _gate(gates, "operation_target_conflict", "activate forbids supersede/revoke targets")
    elif operation == "supersede":
        if supersede_activation_id is None:
            _gate(gates, "supersede_target_missing", "supersede requires the exact current activation ID")
        elif supersede_activation_id != observed_current_id:
            _gate(gates, "supersede_target_mismatch", "supersede target is not the current activation")
        if revoke_activation_id is not None:
            _gate(gates, "operation_target_conflict", "supersede forbids a revoke target")
    elif operation == "revoke":
        if revoke_activation_id is None:
            _gate(gates, "revoke_target_missing", "revoke requires the exact current activation ID")
        elif revoke_activation_id != observed_current_id:
            _gate(gates, "revoke_target_mismatch", "revoke target is not the current activation")
        if supersede_activation_id is not None:
            _gate(gates, "operation_target_conflict", "revoke forbids a supersede target")

    if validated_report is not None and computed_key is not None and not gates:
        report = validated_report.to_dict()
        report_valid = _utc(report["valid_until"])[1]
        activation_valid = min(report_valid, now + timedelta(days=90))
        valid_text = activation_valid.strftime("%Y-%m-%dT%H:%M:%SZ")
        key_events = [record for record in records if record["selection_key"] == computed_key]
        sequence = len(key_events) + 1
        previous = observed_head
        if operation == "supersede":
            current = _current_active_event(records, computed_key)
            if current is None:
                _gate(gates, "current_missing", "supersede requires one current activation")
            else:
                retired_report = next(
                    (
                        item
                        for item in reports
                        if item["report_id"] == current["report_id"]
                        and item["report_digest"] == current["report_digest"]
                    ),
                    None,
                )
                if retired_report is None:
                    _gate(gates, "current_report_missing", "current report must be supplied for supersession")
                else:
                    first = _event(
                        report=retired_report,
                        selection_key=computed_key,
                        sequence=sequence,
                        previous=previous,
                        state="superseded",
                        activated_at=now_text,
                        valid_until=min(_utc(retired_report["valid_until"])[1], activation_valid).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        reviewer_role_id=str(reviewer_role_id),
                        review_reference=review_reference,
                        trust_domain=trust,
                        reason=str(reason),
                        replacement=report,
                    )
                    planned.append(first)
                    sequence += 1
                    previous = first["event_digest"]
        state: Literal["active", "revoked"] = "revoked" if operation == "revoke" else "active"
        target_report = report
        if operation == "revoke":
            current = _current_active_event(records, computed_key)
            if current is not None and (
                current["report_id"] != report["report_id"]
                or current["report_digest"] != report["report_digest"]
            ):
                _gate(gates, "revoke_report_mismatch", "revoke report is not the current active report")
        if not gates:
            planned.append(
                _event(
                    report=target_report,
                    selection_key=computed_key,
                    sequence=sequence,
                    previous=previous,
                    state=state,
                    activated_at=now_text,
                    valid_until=valid_text,
                    reviewer_role_id=str(reviewer_role_id),
                    review_reference=review_reference,
                    trust_domain=trust,
                    reason=str(reason),
                )
            )
            try:
                for item in planned:
                    validate_evidence_activation_record(
                        item, source_trust_domain=trust
                    )
                projected = project_evidence_activations(
                    [*records, *planned],
                    reports,
                    source_trust_domain=trust,
                    as_of=now,
                )
                if operation == "revoke":
                    if computed_key in projected.active_by_selection_key or projected.terminal_reason_by_selection_key.get(computed_key) != "evidence_revoked":
                        raise ValueError("revoke did not produce one valid terminal zero-active projection")
                elif computed_key not in projected.active_by_selection_key:
                    raise ValueError("activation did not produce one active report")
            except ValueError as exc:
                planned.clear()
                _gate(gates, "planned_transition_invalid", str(exc))

    if not approve:
        return EvidenceActivationOutcome(
            status="dry_run_ready" if not gates else "dry_run_blocked",
            operation=operation,
            approved=False,
            source_trust_domain=trust,
            support_scope=support_scope,
            selection_key=computed_key,
            observed_head_digest=observed_head,
            observed_current_activation_id=observed_current_id,
            gates=tuple(gates),
            planned_records=tuple(planned),
            applied_record_digests=(),
        )
    if gates or stream is None:
        return EvidenceActivationOutcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            source_trust_domain=trust,
            support_scope=support_scope,
            selection_key=computed_key,
            observed_head_digest=observed_head,
            observed_current_activation_id=observed_current_id,
            gates=tuple(gates),
            planned_records=tuple(planned),
            applied_record_digests=(),
        )

    appended = b"".join(canonical_json_v1(item) + b"\n" for item in planned)
    try:
        _atomic_replace_stream(
            path=stream,
            observed_bytes=raw_stream,
            replacement_bytes=raw_stream + appended,
            fault_hook=fault_hook,
        )
        readback = _read_regular(
            stream,
            maximum_bytes=MAX_EVIDENCE_ACTIVATION_BYTES,
            label="activation stream readback",
        )
        readback_records = load_evidence_activation_jsonl_bytes(readback)
        projection = project_evidence_activations(
            readback_records,
            reports,
            source_trust_domain=trust,
            as_of=now,
        )
        if projection.head_by_selection_key.get(computed_key) != planned[-1]["event_digest"]:
            raise ValueError("activation stream readback head mismatch")
        if operation == "revoke" and computed_key in projection.active_by_selection_key:
            raise ValueError("activation stream readback did not remain zero-active")
    except (OSError, ValueError) as exc:
        _gate(gates, "atomic_write_failed", str(exc))
        return EvidenceActivationOutcome(
            status="apply_failed",
            operation=operation,
            approved=True,
            source_trust_domain=trust,
            support_scope=support_scope,
            selection_key=computed_key,
            observed_head_digest=observed_head,
            observed_current_activation_id=observed_current_id,
            gates=tuple(gates),
            planned_records=tuple(planned),
            applied_record_digests=(),
        )
    return EvidenceActivationOutcome(
        status="applied",
        operation=operation,
        approved=True,
        source_trust_domain=trust,
        support_scope=support_scope,
        selection_key=computed_key,
        observed_head_digest=planned[-1]["event_digest"],
        observed_current_activation_id=(
            None if operation == "revoke" else str(planned[-1]["event_id"])
        ),
        gates=(),
        planned_records=tuple(planned),
        applied_record_digests=tuple(str(item["event_digest"]) for item in planned),
    )
