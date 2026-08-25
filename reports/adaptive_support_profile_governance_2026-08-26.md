# Adaptive support-profile governance foundation — 2026-08-26

## Outcome

YOLOZU now has an Experimental reviewed operation for one complete ordered dormant
support-profile set. The operation is dry-run by default and can append only to the
canonical `yolozu/data/adaptive_routing/support_profiles.jsonl` SSOT after exact
head, current-set, proposal, and public-review checks.

This does not make a model available. The canonical stream remains empty because no
real measured proposal was reviewed in this change. No lifecycle pointer, evidence
activation, runner binding, model asset, maturity, or public support statement was
added.

## Implemented boundary

- `SupportProfileSetProposal` binds the exact family/channel and complete ordered
  1..32 profile IDs to the accompanying immutable `SupportProfileSpec` records.
- The review service rejects noncanonical input, stale heads/current sets, changed
  immutable profile IDs, incomplete/duplicate coverage, invalid measured gates, and
  detected private identifiers or local paths.
- Approval appends new definitions followed by exactly one set assignment through
  the shared bounded atomic control-stream helper, then validates the readback.
- Recommendation and execution share one loader-derived eligibility provider.
  Execution reopens the support/lifecycle SSOTs and checks the lifecycle-pinned
  historical observation before resolving a runner session.
- Newer dormant reviews do not rewrite an already advertised lifecycle snapshot.

## Current availability

The three packaged baselines remain Candidate-only with unbound execution. The
screening, support-profile, and public evidence streams remain empty. The default
recommendation therefore abstains, and no adaptive model can execute.

## Verification

Focused support-profile governance, bundle/lifecycle, selector, recommendation,
processing, and evidence-activation tests passed locally. Repository-wide required
manifest, documentation, packaging, and pre-push gates are recorded in the pull
request checks for this change.
