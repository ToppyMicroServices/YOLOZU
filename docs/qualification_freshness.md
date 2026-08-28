# Qualification freshness monitor

`yolozu check-qualification-freshness` is a read-only warning surface for
active adaptive-routing qualification evidence. It does not run qualification,
extend a validity deadline, activate evidence, change lifecycle state, promote a
bundle, or edit Beads.

The repository workflow runs every Monday at 00:30 UTC (09:30 Asia/Tokyo),
after the monitored-source scout. It has one non-cancelling concurrency group
and a ten-minute job limit. A successful public run uploads one aggregate
`qualification-freshness-YYYY-MM-DD` artifact for 30 days. The artifact contains
only repository-owned public bundle, report, evidence, and reason identifiers.
It never contains paths, prompts, datasets, host facts, raw timings, or other raw
telemetry.

For each active report, the monitor emits one of `ok`, `due_30`, `due_14`,
`due_7`, `expired`, `runtime_drift`, `artifact_or_bundle_drift`, `conflict`, or
`unknown`. Exact remaining time is computed from parsed UTC instants. At 31 days
the result is `ok`; exact 30, 14, and 7-day boundaries enter their matching due
state. A positive interval below one day reports zero whole days and remains
`due_7`; at or after `valid_until` it is `expired`.

Drift is limited to governed inputs. Immutable bundle/artifact changes, missing
or disabled/revoked lifecycle state, and a non-approved artifact license review
are `artifact_or_bundle_drift`. Runtime/provider version changes are
`runtime_drift`. A later channel or maturity assignment alone is not drift.
Unreadable records and invalid clocks remain `unknown`; a gap, fork, conflicting
activation, or dangling supersession is `conflict`. None of these states proves
that performance regressed.

The scheduled workflow reads back the previous expected artifact with
`actions: read`. If it is absent, the next report records that date through
`--missed-run-date` and does not backfill. Due, drift, conflict, or unknown rows
create or update one exact-title issue, `Qualification evidence freshness action
required`, with at most 64 public-ID rows. A later successful scheduled readback
with no action rows may close it.

For site-managed evidence, pass an explicit workspace-confined evidence root:

```bash
yolozu check-qualification-freshness \
  --evidence-root site/evidence \
  --output reports/qualification_freshness
```

That mode is local-only. It performs no network operation and cannot be uploaded
by the repository workflow. The output remains user-owned. When a row is
actionable, the site operator should create a narrowly scoped requalification
Bead manually and synchronize it through the repository-supported Beads export
workflow. The monitor itself never reads or mutates the Beads database.
