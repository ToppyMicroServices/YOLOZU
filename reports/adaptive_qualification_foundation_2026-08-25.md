# Adaptive image qualification foundation — 2026-08-25

Status: **Experimental foundation implemented; no real bundle qualification collected**.

This report records the repository capability boundary added under
`YOLOZU-ll2.81.1.9` and the reviewed activation boundary added under
`YOLOZU-ll2.81.1.10`, the pure selection boundary added under
`YOLOZU-ll2.81.1.11`, and the read-only MCP recommendation boundary added under
`YOLOZU-ll2.81.1.12`. It is not a `QualificationReport`, activation record,
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
- `yolozu activate-qualification-evidence` revalidates one exact report against
  the current canonical registry/lifecycle and its complete per-key stream. It
  defaults to dry-run and requires explicit `--approve` before an atomic append.
- Exact stale-head/current-activation checks guard activate, two-record
  supersede, and terminal revoke operations. A terminal revoke projects a valid
  zero-active state; gaps, forks, dangling supersession, duplicate active state,
  or later reactivation fail closed.
- Source trust is derived from the retained workflow. A code-owned local
  qualifier output plus explicit local review is only `site_managed` with
  `site_qualified` scope. Arbitrary JSON is `operator_asserted` and cannot be
  activated. Hashes protect post-creation integrity, not provenance against an
  adversarial local operator.
- `select_qualified_pipeline` consumes only validated in-memory registry,
  lifecycle, screening, support, artifact, evidence, environment, and workload
  observations. It applies the documented filter order, per-channel collapse,
  exact hard gates, and lexicographic ranking without filesystem, provider,
  model, runner, or network I/O.
- Unpointed and noncurrent registry entries remain visible as complete excluded
  evaluations. The selector does not invent support observations, choose the
  newest evidence, normalize scores across candidates, or substitute unknown
  metrics.
- The Experimental MCP-only `recommend_image_pipeline` service validates a
  structured job and bounded local input, derives privacy-safe environment and
  workload profiles, and returns a complete selected or abstained
  SelectionDecision. Non-I/O gates run before artifact resolution, so a prior
  rejection records `not_checked_due_to_prior_filter` without opening or hashing
  that candidate's assets.
- The recommendation service performs no inference, model import, download,
  write, network request, or natural-language parsing. It does not return absolute
  paths, raw probe output, per-file hashes, filenames, or prompt text.
- The Experimental MCP-only `process_images` service requires that complete
  selected decision, repeats current lifecycle/evidence/environment/workload/input/
  resolver/artifact/class-mapping checks, and defaults to a runner-free, no-write
  dry-run. Explicit execution is restricted to registered code-owned network-free
  routes and an exact bounded managed output tree.

## Current-state addendum

This report records the qualification foundation before baseline registration.
On 2026-08-26, the three existing model-zoo records were added as non-promoted
Candidate metadata with unbound execution. No adaptive runner, qualification
evidence, support claim, or executable route was added. See
[`adaptive_baseline_bundle_registry_2026-08-26.md`](adaptive_baseline_bundle_registry_2026-08-26.md).

## Evidence boundary

At the time of this foundation report, the packaged bundle registry and
repository-owned runner/evaluator factory maps were empty.
Therefore the public command currently returns an actionable error before model
load and writes no dummy evidence. Normal CI uses injected clocks and runners;
those fixtures test protocol semantics and do not establish latency, throughput,
quality, hardware support, or human adoption.

No names, email addresses, IP addresses, filenames, prompt text, raw timing
events, private messages, user datasets, or private prediction hashes are written
to public qualification reports. Incomplete runner-tree or accelerator memory
coverage is recorded as `unknown` and cannot satisfy a hard memory gate.

## Remaining work

Real measurement begins only after a complete runner-consumed artifact set and
audited code-owned runner are bound to an eligible bundle. Read-only recommendation
is exposed through MCP, but the Candidate-only registry and empty evidence store
mean the default call abstains and no report is active. The pinned processing
interface is implemented, but the runner maps remain empty, so no real adaptive
model can execute.
Model adapters remain separate Beads tasks. A smoke run can never be promoted to
`qualified`.
