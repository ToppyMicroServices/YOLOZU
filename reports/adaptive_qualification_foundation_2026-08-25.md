# Adaptive image qualification foundation — 2026-08-25

Status: **Experimental foundation implemented; no real bundle qualification collected**.

This report records the repository capability boundary added under
`YOLOZU-ll2.81.1.9`. It is not a `QualificationReport`, activation record,
support claim, benchmark result, or adoption snapshot.

## Implemented boundary

- `yolozu qualify-image-pipeline` and a reusable Python API accept a canonical
  typed job, exact packaged bundle identity, bounded local input, and managed
  output destination.
- Inputs and model assets are opened with no-follow checks, read and decoded or
  hashed from the pinned descriptors, and kept open through the runner lifetime.
- Blocking runner work is placed in a POSIX child process group with phase and
  outer watchdogs. A timeout terminates and reaps the group.
- Batch measurement fixes cold start, 20 warm-ups, three 200-handoff repeats,
  reset-at-zero input coverage, exact nearest-rank percentiles, and exact
  count/duration throughput ratios.
- Soft-real-time measurement requires at least 600 seconds and retains every
  successful unsigned 64-bit latency up to the fixed one-million-sample cap.
- Output publication uses the bounded managed transaction and currently emits
  only `qualification_report.json` plus `checksums.json`.

## Evidence boundary

The packaged bundle registry and repository-owned runner/evaluator factory maps are empty.
Therefore the public command currently returns an actionable error before model
load and writes no dummy evidence. Normal CI uses injected clocks and runners;
those fixtures test protocol semantics and do not establish latency, throughput,
quality, hardware support, or human adoption.

No names, email addresses, IP addresses, filenames, prompt text, raw timing
events, private messages, user datasets, or private prediction hashes are written
to public qualification reports. Incomplete runner-tree or accelerator memory
coverage is recorded as `unknown` and cannot satisfy a hard memory gate.

## Remaining work

Real measurement begins only after a bundle, artifacts, lifecycle/license review,
and audited code-owned runner are registered. Selection, recommendation, evidence
activation, and adaptive image execution remain separate Beads tasks. A smoke run
can never be promoted to `qualified`.
