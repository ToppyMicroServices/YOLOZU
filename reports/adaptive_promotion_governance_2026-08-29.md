# Adaptive image-pipeline promotion governance

Collected on 2026-08-29 (Asia/Tokyo).

This change implements the Experimental reviewed promotion control plane. It does
not promote a model. The canonical registry still contains three unbound Candidate
baselines, and the screening, support-profile, and evidence streams remain empty.

The new `yolozu promote-image-pipeline` command is dry-run by default and separates
Candidate-to-Experimental from Experimental-to-Stable. It requires exact lifecycle
and support heads, exact source and target pointers, the complete ordered canonical
profile set, one current repository-managed activation per profile, a non-personal
repository approval role, a public review reference, an explicit rollback target,
and `--approve` before one lifecycle append.

Stable review adds absolute preregistered gates, a passed bounded failure-drill
record with a separate automated-pass reference, and exact-profile current-Stable
comparison. Unknown or incomparable quality, throughput/FPS, p95, p99, process-tree
RSS, or accelerator-memory evidence fails closed. Fixtures exercise these interface
contracts; they are not performance, support, containment, or adoption evidence.

The append helper provides compare-before-commit and lifecycle readback for the one
stream. The service verifies that registry, support, screening, and evidence inputs
are unchanged. Documentation and manifests are synchronized in the same reviewed
commit, without claiming cross-file crash atomicity.
