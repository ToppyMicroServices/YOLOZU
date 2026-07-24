"""Pure helpers for idempotent manual DOI publication."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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


def collect_record_search_pages(
    fetch_page: Callable[[int, int], Mapping[str, Any]],
    *,
    page_size: int = 100,
) -> list[Mapping[str, Any]]:
    """Collect every record from a paginated Zenodo search response."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    records: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    expected_total: int | None = None
    page = 1

    while True:
        payload = fetch_page(page, page_size)
        hits_container = payload.get("hits")
        if not isinstance(hits_container, Mapping):
            raise ValueError("Zenodo records lookup response is missing hits")

        raw_hits = hits_container.get("hits")
        if not isinstance(raw_hits, list):
            raise ValueError("Zenodo records lookup response has invalid hits")

        page_records: list[Mapping[str, Any]] = []
        for record in raw_hits:
            if not isinstance(record, Mapping):
                raise ValueError("Zenodo records lookup returned an invalid record")
            record_id = str(record.get("id") or "").strip()
            if not record_id:
                raise ValueError("Zenodo records lookup hit is missing id")
            if record_id in seen_ids:
                raise ValueError(
                    "Zenodo records lookup repeated a record across pages; "
                    "refusing an incomplete idempotency check"
                )
            seen_ids.add(record_id)
            page_records.append(record)

        raw_total = hits_container.get("total")
        if isinstance(raw_total, Mapping):
            raw_total = raw_total.get("value")
        current_total: int | None = None
        if raw_total is not None:
            try:
                current_total = int(raw_total)
            except (TypeError, ValueError) as exc:
                raise ValueError("Zenodo records lookup response has invalid total") from exc
            if current_total < 0:
                raise ValueError("Zenodo records lookup response has negative total")

        if current_total is not None:
            if expected_total is None:
                expected_total = current_total
            elif current_total != expected_total:
                raise ValueError(
                    "Zenodo records lookup total changed during pagination; "
                    "refusing an inconsistent idempotency check"
                )

        if not page_records:
            if expected_total is not None and len(records) < expected_total:
                raise ValueError(
                    "Zenodo records lookup ended before every published record was returned"
                )
            break

        records.extend(page_records)
        if expected_total is not None:
            if len(records) > expected_total:
                raise ValueError("Zenodo records lookup returned more records than its total")
            if len(records) == expected_total:
                break

        if len(page_records) < page_size:
            if expected_total is not None and len(records) < expected_total:
                raise ValueError(
                    "Zenodo records lookup returned a short page before its total"
                )
            break
        page += 1

    return records


def latest_record_id(records: Iterable[Mapping[str, Any]]) -> str:
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if record_id:
            return record_id
    raise ValueError("Zenodo records lookup hit is missing id")
