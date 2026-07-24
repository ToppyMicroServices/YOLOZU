# Beads issue tracking in YOLOZU

YOLOZU uses `bd` for issue tracking. The supported local CLI is Beads 1.1.0.

## Data boundary

- The local Beads database is the live working store used by `bd`.
- `.beads/issues.jsonl` is an exported issue-level exchange snapshot. It is
  versioned on the `beads-sync` branch, but it is not a full database backup and
  does not contain Dolt history or non-issue tables.
- `.beads/interactions.jsonl` is a separate append-only audit log. It is not
  produced by `bd export`; reconcile entries by their stable `id` before
  publishing it from more than one machine.

## Essential commands

```bash
bd list --all --limit 0
bd ready --limit 0
bd show <issue-id>
bd update <issue-id> --claim
bd close <issue-id>
```

Refresh the local database from the exported remote snapshot:

```bash
bash refresh_beads_sync.sh
```

Export the current issue state:

```bash
bash export_beads_snapshot.sh /path/to/beads-sync/.beads/issues.jsonl
```

The complete two-machine and missing-worktree procedure is documented in
`docs/beads_github_workflow.md`.

## Import safety

Normal `bd import` uses timestamp-aware upsert behavior: newer remote rows can
update local rows, older rows are skipped, and same-timestamp local fields are
kept while labels, comments, and dependencies are merged. Inspect the JSON
summary printed by `refresh_beads_sync.sh`.

The published snapshot contains historical hierarchical tombstones that
`bd 1.1.0 import` cannot load into a fresh database when a non-tombstoned
retained descendant needs a deleted parent for its child counter. The refresh
helper therefore:

1. preserves each original tombstone JSON line in compatibility metadata;
2. imports it as a closed, labeled placeholder in the local operational view;
3. takes a temporary database backup and restores it if import fails.

These placeholders are not publishable state.
`export_beads_snapshot.sh` requires the pulled remote snapshot as its baseline,
refuses to drop any remote issue ID, restores the exact tombstone lines, and
atomically replaces only `.beads/issues.jsonl`. It does not touch
`.beads/interactions.jsonl`.

Do not use `bd import --allow-stale` for normal sharing. It intentionally permits
an older snapshot to overwrite newer local issue fields and is reserved for an
explicit restore.

For Beads CLI documentation, run `bd quickstart` or `bd <command> --help`.
