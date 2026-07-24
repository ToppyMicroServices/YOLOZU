from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "beads_snapshot_compat.py"
SPEC = importlib.util.spec_from_file_location("beads_snapshot_compat", TOOL)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot load {TOOL}")
COMPAT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPAT
SPEC.loader.exec_module(COMPAT)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> list[str]:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return lines


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class BeadsSnapshotCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.baseline = self.base / "baseline.jsonl"
        self.normalized = self.base / "normalized.jsonl"
        self.restored = self.base / "restored.jsonl"
        self.records: list[dict[str, object]] = [
            {
                "id": "T42-parent",
                "title": "deleted parent",
                "status": "tombstone",
                "priority": 2,
                "issue_type": "epic",
                "created_at": "2026-01-01T00:00:00.123456Z",
                "updated_at": "2026-01-02T00:00:00.654321Z",
                "close_reason": "historical cleanup",
                "deleted_at": "2026-01-02T00:00:00.654321Z",
                "deleted_by": "fixture",
                "delete_reason": "historical cleanup",
                "original_type": "epic",
            },
            {
                "id": "T42-parent.1",
                "title": "retained closed descendant",
                "status": "closed",
                "priority": 2,
                "issue_type": "task",
                "created_at": "2026-01-01T01:00:00Z",
                "updated_at": "2026-01-03T00:00:00Z",
                "dependencies": [
                    {
                        "issue_id": "T42-parent.1",
                        "depends_on_id": "T42-parent",
                        "type": "parent-child",
                        "created_at": "2026-01-01T01:00:00Z",
                    },
                    {
                        "issue_id": "T42-parent.1",
                        "depends_on_id": "T42-parent",
                        "type": "blocks",
                        "created_at": "2026-01-01T01:00:01Z",
                    },
                ],
            },
            {
                "id": "T42-open",
                "title": "current issue",
                "status": "open",
                "priority": 1,
                "issue_type": "bug",
                "created_at": "2026-01-04T00:00:00Z",
                "updated_at": "2026-01-04T00:00:00Z",
            },
        ]
        self.baseline_lines = _write_jsonl(self.baseline, self.records)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_normalize_import_preserves_ids_dependencies_and_original_json(
        self,
    ) -> None:
        summary = COMPAT.normalize_import(self.baseline, self.normalized)
        normalized = _read_jsonl(self.normalized)
        by_id = {record["id"]: record for record in normalized}

        self.assertEqual(summary["normalized_tombstones"], 1)
        self.assertEqual(
            summary["source_fingerprints"]["id_sha256"],
            summary["output_fingerprints"]["id_sha256"],
        )
        self.assertEqual(
            summary["source_fingerprints"]["dependency_sha256"],
            summary["output_fingerprints"]["dependency_sha256"],
        )

        placeholder = by_id["T42-parent"]
        self.assertEqual(placeholder["status"], "closed")
        self.assertIn(
            COMPAT.LEGACY_TOMBSTONE_LABEL,
            placeholder["labels"],
        )
        marker = placeholder["metadata"][COMPAT.LEGACY_TOMBSTONE_METADATA_KEY]
        self.assertEqual(marker["original_json"], self.baseline_lines[0])
        self.assertEqual(
            marker["original_sha256"],
            COMPAT._sha256_text(self.baseline_lines[0]),
        )

    def test_restore_export_recovers_exact_tombstone_and_dependency_edges(
        self,
    ) -> None:
        COMPAT.normalize_import(self.baseline, self.normalized)
        local_records = _read_jsonl(self.normalized)
        local_records.append(
            {
                "id": "T42-new",
                "title": "new local issue",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "created_at": "2026-01-05T00:00:00Z",
                "updated_at": "2026-01-05T00:00:00Z",
            }
        )
        local = self.base / "local.jsonl"
        _write_jsonl(local, local_records)

        summary = COMPAT.restore_export(local, self.baseline, self.restored)
        restored_lines = self.restored.read_text(encoding="utf-8").splitlines()
        restored = _read_jsonl(self.restored)

        self.assertEqual(restored_lines[0], self.baseline_lines[0])
        self.assertEqual(
            {record["id"] for record in restored},
            {"T42-parent", "T42-parent.1", "T42-open", "T42-new"},
        )
        self.assertEqual(summary["restored_tombstones"], ["T42-parent"])
        self.assertTrue(summary["remote_ids_preserved"])
        self.assertEqual(
            summary["baseline_fingerprints"]["tombstone_sha256"],
            summary["output_fingerprints"]["tombstone_sha256"],
        )
        self.assertEqual(
            summary["baseline_fingerprints"]["dependency_sha256"],
            summary["output_fingerprints"]["dependency_sha256"],
        )

    def test_restore_export_rejects_missing_remote_id_before_output_change(
        self,
    ) -> None:
        local = self.base / "local.jsonl"
        _write_jsonl(local, [self.records[2]])
        self.restored.write_text("unchanged\n", encoding="utf-8")

        with self.assertRaisesRegex(
            COMPAT.SnapshotError,
            "missing remote snapshot ids",
        ):
            COMPAT.restore_export(local, self.baseline, self.restored)

        self.assertEqual(
            self.restored.read_text(encoding="utf-8"),
            "unchanged\n",
        )

    def test_restore_export_rejects_modified_placeholder(self) -> None:
        COMPAT.normalize_import(self.baseline, self.normalized)
        local_records = _read_jsonl(self.normalized)
        local_records[0]["status"] = "open"
        local = self.base / "local.jsonl"
        _write_jsonl(local, local_records)

        with self.assertRaisesRegex(
            COMPAT.SnapshotError,
            "placeholder was modified",
        ):
            COMPAT.restore_export(local, self.baseline, self.restored)

    def test_restore_export_rejects_placeholder_field_edits(self) -> None:
        mutations = {
            "title": lambda record: record.__setitem__("title", "edited title"),
            "notes": lambda record: record.__setitem__("notes", "edited notes"),
            "external_ref": lambda record: record.__setitem__(
                "external_ref",
                "gh-99",
            ),
            "updated_at": lambda record: record.__setitem__(
                "updated_at",
                "2026-01-10T00:00:00Z",
            ),
        }

        for field, mutate in mutations.items():
            with self.subTest(field=field):
                COMPAT.normalize_import(self.baseline, self.normalized)
                local_records = _read_jsonl(self.normalized)
                mutate(local_records[0])
                local = self.base / f"local-{field}.jsonl"
                _write_jsonl(local, local_records)

                with self.assertRaisesRegex(
                    COMPAT.SnapshotError,
                    "placeholder was modified",
                ):
                    COMPAT.restore_export(local, self.baseline, self.restored)

    def test_restore_export_rejects_newer_non_tombstone_baseline(self) -> None:
        local_records = _read_jsonl(self.baseline)
        local_records[2]["title"] = "stale local title"
        local_records[2]["status"] = "open"
        local_records[2]["updated_at"] = "2026-01-03T23:59:59Z"
        local = self.base / "local.jsonl"
        _write_jsonl(local, local_records)
        self.restored.write_text("unchanged\n", encoding="utf-8")

        with self.assertRaisesRegex(
            COMPAT.SnapshotError,
            "remote baseline is newer than the local export",
        ):
            COMPAT.restore_export(local, self.baseline, self.restored)

        self.assertEqual(
            self.restored.read_text(encoding="utf-8"),
            "unchanged\n",
        )


if __name__ == "__main__":
    unittest.main()
