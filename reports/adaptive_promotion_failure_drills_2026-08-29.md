# Adaptive promotion failure drills

Collected on 2026-08-29 (Asia/Tokyo).

The normal adaptive promotion test suite runs six deterministic offline fixtures.
They use repository-owned test inputs and require no network or GPU.

| Fixture | Observed failure | Selector result |
|---|---|---|
| Weight SHA-256 mismatch | Artifact resolver rejects `artifact model: SHA-256 mismatch` | Exact Stable last-known-good remains selected |
| Incompatible runtime version | Selector records `runtime_unavailable` | Exact Stable last-known-good remains selected |
| Qualification timeout | Forked runner records `phase_timeout` and is terminated | Candidate evidence is inactive; exact Stable last-known-good remains selected |
| p95 regression | Stable comparison records `stable p95_latency_ms regressed` | Candidate fails `p95_latency_gate_failed`; exact Stable last-known-good remains selected |
| Unknown weight license | Screening records `weight_license_unknown` | Candidate fails `license_not_approved`; exact Stable last-known-good remains selected |
| Predictions interface contract violation | Screening records `predictions_interface_mapping_failed` | No eligible bundle exists, so selection abstains |

Each failed fixture is also presented to the approved promotion path as a canonical
failed drill record. Promotion returns `apply_failed` with
`failure_drill_invalid`. Tests assert byte identity for the registry, lifecycle,
support-profile, screening, and evidence streams and confirm that the Stable pointer
does not change. A separate approved fixture reassigns only the Experimental pointer
to its exact eligible last-known-good target while preserving the Stable pointer and
all immutable inputs.

The fixture stream is canonical JSONL. Its checksum manifest covers every other
declared regular file and does not list itself. Tests assert the expected file set,
count, byte total, size, SHA-256, canonical bytes, and record order.

These drills prove the tested fail-closed interface behavior. They do not establish
real model qualification, runtime containment, OOM behavior on target hardware,
performance, support, adoption, or production readiness. No canonical promotion or
rollback event was added, and the packaged registry remains Candidate-only.
