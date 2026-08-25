# Adaptive candidate-screening foundation — 2026-08-26

## Result

YOLOZU now has a non-executing `CandidateScreeningRecord` interface contract and
one canonical provider that projects reviewed records into the existing
`ScreeningEligibilityObservation` selector input. The implementation derives
`pass`, `hold`, or `reject`; callers cannot supply the outcome independently of
the recorded facts.

This is an interface and control-path result. No external candidate was screened,
downloaded, imported, executed, registered, qualified, selected, or promoted.
The repository-owned stream
`yolozu/data/adaptive_routing/candidate_screening.jsonl` is empty, so there is no
current candidate pass or new availability claim.

## Boundaries implemented

- Source provenance and integrity, code/weight/dataset licenses, weight source,
  local availability, task/output mapping, runtime/provider needs,
  compute/memory estimates, maintenance, and known supply-chain concerns remain
  separate mechanical facts. Human review remains separate.
- Mandatory unknown facts produce `hold`; a failed fact produces `reject`.
  A new runtime/provider surface also remains on hold.
- The decision key binds the exact source, immutable revision, and requested
  capability. Supersession requires the next sequence and exact predecessor
  identity. Projection follows sequence, not review time.
- The complete stream is bounded to 8,192 records and 64 MiB. Partial suffixes,
  malformed JSON, duplicate keys, gaps, forks, duplicate identities, and wrong
  predecessors fail closed. V1 has no truncation or favorable-suffix scan.
- Trust comes from the selected source path. The packaged stream may be
  `yolozu_managed`; an explicit workspace-confined stream is always
  `operator_asserted` and cannot satisfy a managed pass.
- Recommendation preflight maps absence, hold, reject, untrusted input, pass
  conflict, and revision mismatch explicitly before the pure selector. A later
  hold or reject immediately excludes a previously passed candidate.
- A screening record is not accepted as an AlgorithmBundle registry. A pass
  permits later isolated adapter and qualification work only.

## Verification performed

- `python3 -m unittest tests.test_adaptive_candidate_screening`: 9 tests passed.
- Related recommendation, routing, processing, selection, selector, and registry
  suites: 65 tests passed.
- Ruff checks over the new module, integrations, and tests passed.

These are structural and fixture checks. They are not performance, support,
security-vulnerability absence, adoption, or real-model evidence.
