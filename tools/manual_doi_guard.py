"""Pure helpers for idempotent manual DOI publication."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def normalize_manual_version(value: object) -> str:
    version = str(value or "").strip()
    return version[1:] if version.startswith("v") else version


def record_version(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return normalize_manual_version(metadata.get("version"))


def find_matching_record(
    records: Iterable[Mapping[str, Any]],
    version: object,
) -> Mapping[str, Any] | None:
    expected = normalize_manual_version(version)
    if not expected:
        return None
    for record in records:
        if record_version(record) == expected:
            return record
    return None


def latest_record_id(records: Iterable[Mapping[str, Any]]) -> str:
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if record_id:
            return record_id
    raise ValueError("Zenodo records lookup hit is missing id")
