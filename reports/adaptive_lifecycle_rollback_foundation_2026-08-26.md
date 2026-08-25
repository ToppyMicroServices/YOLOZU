# Adaptive lifecycle maintenance and rollback foundation — 2026-08-26

## Outcome

YOLOZU now has an Experimental, dry-run-first interface contract for reviewed
bundle lifecycle maintenance and explicit per-channel rollback. The implemented
operations are disable, enable, license review, terminal global revoke, and rollback
of one Experimental or Stable pointer to a previously assigned, currently eligible same-family target or literal
`none`.

This is governance infrastructure, not a model-availability result. No canonical
lifecycle event was appended. The three packaged baselines remain Candidate-only,
their execution bindings remain unbound, and the screening, support-profile, and
public evidence streams remain empty. No adaptive model can currently execute.

## Mutation boundary

- Omission of `--approve` always writes nothing and returns a bounded gate result.
- Approval requires the exact lifecycle head, immutable bundle and artifact-set
  identities, affected state or pointer identities, a non-personal repository role,
  and an approved public review.
- Global events retain the ordered artifact hashes and sizes. Existing specs,
  artifacts, reports, and earlier lifecycle bytes are never rewritten.
- Disable and enable change eligibility. License review requires the complete ordered
  artifact set. Global revoke is terminal and applies through all channel pointers.
- Rollback changes only one named channel. Target `none` records an explicit empty
  pointer. A non-`none` target must be different, same-family, globally eligible,
  Candidate-registered, screening-eligible where applicable, and backed by exactly
  one current repository-managed activation for every historical advertised profile.
  It must also have a prior public assignment to that exact channel; a never-assigned
  Candidate belongs to the separate reviewed promotion path. The rollback event
  audit-binds the exact historical target-assignment digest.
- Missing, extra, duplicate, stale, cross-environment, site/operator-asserted, revoked,
  expired, or otherwise mismatched evidence fails closed.

The operation does not infer a target from download, traffic, benchmark, or product
metrics. It does not select a newer dormant profile set, accept a caller-authored
profile subset, promote a bundle to Stable, or run automatically.

## Interface contract and retention

`BundleLifecycleRecord` remains an append-only event format. New reviewed maintenance
records add explicit operation, actor-role, review-status, immutable artifact-member,
and reproducibility fields without changing existing record digests. The canonical
`LifecycleRollbackBindings` JSON input contains one complete ordered managed
activation set. Source and packaged schemas and manifests are byte-identical.

An approved update reopens the lifecycle and support streams immediately before the
atomic append and validates the complete readback. An interrupted write or readback
failure never returns success. Immutable history is retained; no derived mutable
projection is published.

## Verification scope

The focused suite covers dry-run and approved global transitions, changed and
unchanged license review, terminal revoke across two pointers, explicit `none`, exact
last-known-good reassignment, stale heads and historical sets, family and artifact
mismatch, incomplete or untrusted evidence, environment mismatch, interrupted write,
readback failure, and CLI JSON output. These are controlled interface tests. They are
not public support, performance, adoption, or production-readiness evidence.
