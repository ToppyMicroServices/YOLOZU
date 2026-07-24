#!/usr/bin/env python3
"""Normalize legacy Beads tombstones for import and restore them on export."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEGACY_TOMBSTONE_LABEL = "beads-sync-legacy-tombstone"
LEGACY_TOMBSTONE_METADATA_KEY = "beads_sync_legacy_tombstone"
MARKER_SCHEMA_VERSION = 1
PLACEHOLDER_VOLATILE_FIELDS = frozenset(
    {
        "_type",
        "closed_at",
        "comment_count",
        "created_at",
        "deleted_at",
        "deleted_by",
        "delete_reason",
        "dependencies",
        "dependency_count",
        "dependent_count",
        "labels",
        "original_type",
        "parent",
        "status",
        "updated_at",
    }
)
ISSUE_COMPARISON_IGNORED_FIELDS = frozenset(
    {
        "_type",
        "comment_count",
        "dependencies",
        "dependency_count",
        "dependent_count",
    }
)
ISSUE_TIMESTAMP_FIELDS = ("created_at", "updated_at", "closed_at")


class SnapshotError(ValueError):
    """Raised when a snapshot cannot be transformed without losing state."""


@dataclass(frozen=True)
class SnapshotRecord:
    record: dict[str, Any]
    raw: str
    line_number: int

    @property
    def issue_id(self) -> str:
        return str(self.record["id"])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_snapshot(path: Path) -> list[SnapshotRecord]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SnapshotError(f"cannot read snapshot {path}: {exc}") from exc

    records: list[SnapshotRecord] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            raise SnapshotError(f"{path}:{line_number}: blank JSONL line")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SnapshotError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise SnapshotError(f"{path}:{line_number}: record must be an object")
        issue_id = value.get("id")
        if not isinstance(issue_id, str) or not issue_id:
            raise SnapshotError(
                f"{path}:{line_number}: record requires a non-empty string id"
            )
        if issue_id in seen_ids:
            raise SnapshotError(f"{path}:{line_number}: duplicate id {issue_id}")
        seen_ids.add(issue_id)
        records.append(SnapshotRecord(record=value, raw=raw, line_number=line_number))

    if not records:
        raise SnapshotError(f"{path}: snapshot is empty")
    return records


def _write_snapshot(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{line}\n" for line in lines)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _record_map(records: Iterable[SnapshotRecord]) -> dict[str, SnapshotRecord]:
    return {item.issue_id: item for item in records}


def _dependency_values(item: SnapshotRecord) -> list[dict[str, Any]]:
    value = item.record.get("dependencies", [])
    if value is None:
        value = []
    if not isinstance(value, list) or not all(
        isinstance(dependency, dict) for dependency in value
    ):
        raise SnapshotError(
            f"record {item.issue_id}: dependencies must be an object array"
        )
    return value


def _dependency_key(dependency: dict[str, Any]) -> tuple[str, str, str]:
    values = tuple(
        dependency.get(field) for field in ("issue_id", "depends_on_id", "type")
    )
    if not all(isinstance(value, str) and value for value in values):
        raise SnapshotError(
            "dependency requires issue_id, depends_on_id, and type strings"
        )
    return values  # type: ignore[return-value]


def _dependency_fingerprints(
    records: Iterable[SnapshotRecord],
) -> tuple[int, str, str]:
    semantic_dependencies: list[str] = []
    complete_dependencies: list[str] = []
    for item in records:
        for dependency in _dependency_values(item):
            semantic_dependencies.append("\0".join(_dependency_key(dependency)))
            complete_dependencies.append(
                f"{item.issue_id}\0{_canonical_json(dependency)}"
            )
    return (
        len(semantic_dependencies),
        _sha256_text("\n".join(sorted(semantic_dependencies))),
        _sha256_text("\n".join(sorted(complete_dependencies))),
    )


def _fingerprints(records: list[SnapshotRecord]) -> dict[str, Any]:
    ids = sorted(item.issue_id for item in records)
    dependency_count, dependency_sha256, dependency_record_sha256 = (
        _dependency_fingerprints(records)
    )
    tombstones = sorted(
        f"{item.issue_id}\0{_sha256_text(item.raw)}"
        for item in records
        if item.record.get("status") == "tombstone"
    )
    raw_payload = "".join(f"{item.raw}\n" for item in records)
    return {
        "count": len(records),
        "id_sha256": _sha256_text("\n".join(ids)),
        "dependency_count": dependency_count,
        "dependency_sha256": dependency_sha256,
        "dependency_record_sha256": dependency_record_sha256,
        "tombstone_count": len(tombstones),
        "tombstone_sha256": _sha256_text("\n".join(tombstones)),
        "raw_sha256": _sha256_text(raw_payload),
    }


def _legacy_marker(item: SnapshotRecord) -> dict[str, Any] | None:
    metadata = item.record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    marker = metadata.get(LEGACY_TOMBSTONE_METADATA_KEY)
    if marker is None:
        return None
    if not isinstance(marker, dict):
        raise SnapshotError(f"record {item.issue_id}: invalid legacy tombstone marker")
    if marker.get("schema_version") != MARKER_SCHEMA_VERSION:
        raise SnapshotError(
            f"record {item.issue_id}: unsupported legacy tombstone marker version"
        )
    original_json = marker.get("original_json")
    if not isinstance(original_json, str) or not original_json:
        raise SnapshotError(f"record {item.issue_id}: marker requires original_json")
    expected_sha256 = marker.get("original_sha256")
    if expected_sha256 != _sha256_text(original_json):
        raise SnapshotError(
            f"record {item.issue_id}: legacy tombstone marker hash mismatch"
        )
    return marker


def _placeholder_stable_view(
    record: dict[str, Any],
    *,
    remove_marker: bool,
) -> dict[str, Any]:
    view = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in PLACEHOLDER_VOLATILE_FIELDS
    }
    metadata = view.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SnapshotError("placeholder metadata must be an object")
    if remove_marker:
        metadata.pop(LEGACY_TOMBSTONE_METADATA_KEY, None)
    if metadata:
        view["metadata"] = metadata
    else:
        view.pop("metadata", None)
    return view


def _timestamp_matches_imported_value(original: Any, current: Any) -> bool:
    original_time = _parse_timestamp(original)
    if original_time is None:
        return True
    current_time = _parse_timestamp(current)
    if current_time is None:
        return False
    return abs((current_time - original_time).total_seconds()) <= 1


def normalize_import(source: Path, output: Path) -> dict[str, Any]:
    source_records = _read_snapshot(source)
    normalized_lines: list[str] = []
    normalized_tombstones = 0

    for item in source_records:
        record = copy.deepcopy(item.record)
        if record.get("status") == "tombstone":
            metadata = record.get("metadata")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise SnapshotError(
                    f"record {item.issue_id}: metadata must be an object"
                )
            if LEGACY_TOMBSTONE_METADATA_KEY in metadata:
                raise SnapshotError(
                    f"record {item.issue_id}: marker key already exists"
                )

            labels = record.get("labels")
            if labels is None:
                labels = []
            if not isinstance(labels, list) or not all(
                isinstance(label, str) for label in labels
            ):
                raise SnapshotError(
                    f"record {item.issue_id}: labels must be a string array"
                )

            metadata[LEGACY_TOMBSTONE_METADATA_KEY] = {
                "schema_version": MARKER_SCHEMA_VERSION,
                "original_json": item.raw,
                "original_sha256": _sha256_text(item.raw),
            }
            record["metadata"] = metadata
            record["labels"] = list(dict.fromkeys([*labels, LEGACY_TOMBSTONE_LABEL]))
            record["status"] = "closed"
            if not record.get("close_reason"):
                record["close_reason"] = (
                    record.get("delete_reason")
                    or "Legacy tombstone retained for snapshot compatibility"
                )
            normalized_tombstones += 1

        normalized_lines.append(_compact_json(record))

    _write_snapshot(output, normalized_lines)
    normalized_records = _read_snapshot(output)
    if len(normalized_records) != len(source_records):
        raise SnapshotError("normalized snapshot record count changed")
    if {item.issue_id for item in normalized_records} != {
        item.issue_id for item in source_records
    }:
        raise SnapshotError("normalized snapshot issue ids changed")

    return {
        "operation": "normalize-import",
        "source": str(source),
        "output": str(output),
        "normalized_tombstones": normalized_tombstones,
        "source_fingerprints": _fingerprints(source_records),
        "output_fingerprints": _fingerprints(normalized_records),
    }


def _parse_marker_original(item: SnapshotRecord) -> SnapshotRecord:
    marker = _legacy_marker(item)
    if marker is None:
        raise SnapshotError(f"record {item.issue_id}: marker is missing")
    original_raw = str(marker["original_json"])
    try:
        original_record = json.loads(original_raw)
    except json.JSONDecodeError as exc:
        raise SnapshotError(
            f"record {item.issue_id}: marker original_json is invalid"
        ) from exc
    if not isinstance(original_record, dict):
        raise SnapshotError(
            f"record {item.issue_id}: marker original_json must be an object"
        )
    if original_record.get("id") != item.issue_id:
        raise SnapshotError(
            f"record {item.issue_id}: marker id does not match current id"
        )
    if original_record.get("status") != "tombstone":
        raise SnapshotError(
            f"record {item.issue_id}: marker is not an original tombstone"
        )
    labels = item.record.get("labels") or []
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise SnapshotError(f"record {item.issue_id}: labels must be a string array")

    expected_record = copy.deepcopy(original_record)
    expected_labels = expected_record.get("labels") or []
    if not isinstance(expected_labels, list) or not all(
        isinstance(label, str) for label in expected_labels
    ):
        raise SnapshotError(
            f"record {item.issue_id}: original labels must be a string array"
        )
    expected_labels = list(dict.fromkeys([*expected_labels, LEGACY_TOMBSTONE_LABEL]))
    expected_record["status"] = "closed"
    expected_record["labels"] = expected_labels
    if not expected_record.get("close_reason"):
        expected_record["close_reason"] = (
            expected_record.get("delete_reason")
            or "Legacy tombstone retained for snapshot compatibility"
        )

    local_dependency_keys = {
        _dependency_key(dependency) for dependency in _dependency_values(item)
    }
    original_item = SnapshotRecord(
        record=original_record,
        raw=original_raw,
        line_number=item.line_number,
    )
    original_dependency_keys = {
        _dependency_key(dependency) for dependency in _dependency_values(original_item)
    }
    placeholder_changed = (
        item.record.get("status") != "closed"
        or set(labels) != set(expected_labels)
        or not local_dependency_keys.issubset(original_dependency_keys)
        or _placeholder_stable_view(item.record, remove_marker=True)
        != _placeholder_stable_view(expected_record, remove_marker=False)
        or not _timestamp_matches_imported_value(
            original_record.get("created_at"),
            item.record.get("created_at"),
        )
        or not _timestamp_matches_imported_value(
            original_record.get("updated_at"),
            item.record.get("updated_at"),
        )
    )
    if placeholder_changed:
        raise SnapshotError(
            f"record {item.issue_id}: legacy tombstone placeholder was modified; "
            "remove the marker explicitly before intentional resurrection"
        )
    return original_item


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _strictly_newer(candidate: SnapshotRecord, baseline: SnapshotRecord) -> bool:
    candidate_time = _parse_timestamp(candidate.record.get("updated_at"))
    baseline_time = _parse_timestamp(baseline.record.get("updated_at"))
    if candidate_time is None or baseline_time is None:
        return False
    return candidate_time > baseline_time


def _issue_semantic_view(record: dict[str, Any]) -> dict[str, Any]:
    view = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in ISSUE_COMPARISON_IGNORED_FIELDS
    }
    labels = view.get("labels")
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        view["labels"] = sorted(set(labels))
    for field in ISSUE_TIMESTAMP_FIELDS:
        parsed = _parse_timestamp(view.get(field))
        if parsed is not None:
            view[field] = parsed.astimezone(timezone.utc).isoformat()
    return view


def _require_local_not_older(
    local_item: SnapshotRecord,
    baseline_item: SnapshotRecord,
) -> None:
    local_time = _parse_timestamp(local_item.record.get("updated_at"))
    baseline_time = _parse_timestamp(baseline_item.record.get("updated_at"))
    if baseline_time is None:
        return
    if local_time is None:
        raise SnapshotError(
            f"record {local_item.issue_id}: local updated_at is missing or invalid "
            "while the remote baseline has a timestamp; rerun "
            "refresh_beads_sync.sh before exporting"
        )
    if baseline_time > local_time:
        raise SnapshotError(
            f"record {local_item.issue_id}: remote baseline is newer than the "
            "local export; rerun refresh_beads_sync.sh before exporting"
        )
    if baseline_time == local_time and _issue_semantic_view(
        baseline_item.record
    ) != _issue_semantic_view(local_item.record):
        raise SnapshotError(
            f"record {local_item.issue_id}: remote baseline and local export have "
            "the same updated_at but divergent issue fields; resolve the tie with "
            "bd update before exporting"
        )


def _merge_baseline_dependencies(
    local_item: SnapshotRecord,
    baseline_item: SnapshotRecord,
) -> str:
    """Keep remote edges and append new local edges without duplicating a key."""
    baseline_dependencies = _dependency_values(baseline_item)
    local_dependencies = _dependency_values(local_item)
    merged = copy.deepcopy(baseline_dependencies)
    seen = {_dependency_key(dependency) for dependency in baseline_dependencies}
    for dependency in local_dependencies:
        key = _dependency_key(dependency)
        if key not in seen:
            merged.append(copy.deepcopy(dependency))
            seen.add(key)

    record = copy.deepcopy(local_item.record)
    if merged:
        record["dependencies"] = merged
    else:
        record.pop("dependencies", None)
    return _compact_json(record)


def restore_export(local: Path, baseline: Path, output: Path) -> dict[str, Any]:
    local_records = _read_snapshot(local)
    baseline_records = _read_snapshot(baseline)
    local_by_id = _record_map(local_records)
    baseline_by_id = _record_map(baseline_records)

    missing_ids = sorted(set(baseline_by_id) - set(local_by_id))
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        raise SnapshotError(
            "local export is missing remote snapshot ids: "
            f"{preview}{suffix} ({len(missing_ids)} missing)"
        )

    output_lines: list[str] = []
    emitted_ids: set[str] = set()
    restored_tombstones: list[str] = []
    superseded_tombstones: list[str] = []

    for baseline_item in baseline_records:
        local_item = local_by_id[baseline_item.issue_id]
        marker = _legacy_marker(local_item)
        if marker is not None:
            original_item = _parse_marker_original(local_item)
            if baseline_item.record.get("status") != "tombstone":
                raise SnapshotError(
                    f"record {local_item.issue_id}: placeholder conflicts with "
                    "a non-tombstone baseline"
                )
            if _canonical_json(original_item.record) != _canonical_json(
                baseline_item.record
            ):
                raise SnapshotError(
                    f"record {local_item.issue_id}: placeholder tombstone differs "
                    "from the current remote baseline"
                )
            output_lines.append(baseline_item.raw)
            restored_tombstones.append(local_item.issue_id)
        elif baseline_item.record.get("status") == "tombstone":
            if local_item.record.get("status") == "tombstone":
                if _strictly_newer(local_item, baseline_item):
                    output_lines.append(local_item.raw)
                else:
                    output_lines.append(baseline_item.raw)
                restored_tombstones.append(local_item.issue_id)
            elif _strictly_newer(local_item, baseline_item):
                output_lines.append(
                    _merge_baseline_dependencies(local_item, baseline_item)
                )
                superseded_tombstones.append(local_item.issue_id)
            else:
                raise SnapshotError(
                    f"record {local_item.issue_id}: remote tombstone would be "
                    "silently resurrected by a non-newer local row"
                )
        else:
            _require_local_not_older(local_item, baseline_item)
            output_lines.append(_merge_baseline_dependencies(local_item, baseline_item))

        emitted_ids.add(local_item.issue_id)

    for local_item in local_records:
        if local_item.issue_id in emitted_ids:
            continue
        if _legacy_marker(local_item) is not None:
            original_item = _parse_marker_original(local_item)
            output_lines.append(original_item.raw)
            restored_tombstones.append(local_item.issue_id)
        else:
            output_lines.append(local_item.raw)
        emitted_ids.add(local_item.issue_id)

    if emitted_ids != set(local_by_id):
        raise SnapshotError("not every local issue was emitted")

    _write_snapshot(output, output_lines)
    output_records = _read_snapshot(output)
    output_by_id = _record_map(output_records)
    if not set(baseline_by_id).issubset(output_by_id):
        raise SnapshotError("output dropped remote snapshot issue ids")

    return {
        "operation": "restore-export",
        "local": str(local),
        "baseline": str(baseline),
        "output": str(output),
        "restored_tombstones": sorted(restored_tombstones),
        "superseded_tombstones": sorted(superseded_tombstones),
        "remote_ids_preserved": True,
        "local_fingerprints": _fingerprints(local_records),
        "baseline_fingerprints": _fingerprints(baseline_records),
        "output_fingerprints": _fingerprints(output_records),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep legacy Beads tombstones importable in bd 1.1.0 without "
            "publishing closed placeholders."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser(
        "normalize-import",
        help="convert legacy tombstones to marked local closed placeholders",
    )
    normalize_parser.add_argument("--input", required=True, type=Path)
    normalize_parser.add_argument("--output", required=True, type=Path)

    restore_parser = subparsers.add_parser(
        "restore-export",
        help="restore marked placeholders to lossless tombstones before publishing",
    )
    restore_parser.add_argument("--local", required=True, type=Path)
    restore_parser.add_argument("--baseline", required=True, type=Path)
    restore_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "normalize-import":
            result = normalize_import(args.input, args.output)
        else:
            result = restore_export(args.local, args.baseline, args.output)
    except SnapshotError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
