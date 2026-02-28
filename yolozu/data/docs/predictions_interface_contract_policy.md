# Predictions Interface Contract Policy

This page defines the stable/soft boundaries for predictions interface contract evolution.

## Hard Invariants

- Entry payload must canonicalize to:
  - `schema_version` (entry-level) = `2`
  - non-empty `image` string
  - `detections` list (`[]` allowed)
- Numeric safety in canonicalization:
  - finite `score`, finite bbox components
  - `0 <= score <= 1`
  - `0 <= cx,cy <= 1`, `0 < w,h <= 1` when `bbox` exists
- Deterministic normalization:
  - stable detection sort `(-score, class_id, bbox)`
  - duplicate image entries rejected
- Schema lifecycle:
  - current entry schema version: `2`
  - minimum supported entry schema version: `1`
  - unknown future entry schema versions are rejected

## Soft Invariants

- Score/metric drift and performance drift are behavior checks and may run in `warn` mode before hardening.
- Backend numerical differences (torch/onnxruntime/tensorrt) are treated as soft drift unless explicitly hardened in CI policy.

## Migration Policy

- One-generation migration is supported:
  - entry `v1` (or missing `schema_version`) -> entry `v2`
- Migration is applied during canonicalization/load paths.
- Breaking changes beyond one generation require explicit migration extension and docs update.
