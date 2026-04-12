# Roadmap notes (Symmetry + Commonsense + Real-Time Constraints)

This document is kept for **historical context**. Active work should be tracked as Beads issues:

```bash
bd list
bd search "<keyword>"
```

Goal: Convert symmetry + commonsense + real-time constraints into an implementable, test-first plan with explicit gates.

Status (2025-02-14)
- Not stuck. Baseline and smoke-run use early-phase utilities; real metrics still require a full training/inference repo.
- Metric gates are checked against the historical baseline (`reports/baseline.json`), not a real model.

---

## Stage 0) Repo discovery and baselines
Deliverables
- [x] Map training/inference code paths, config loader, metrics, and tests.
- [x] Capture baseline metrics and fps for later comparison.

Tests (set up before implementation)
- [x] Smoke run for training and inference to ensure harness exists.

Gate
- [x] Baseline report saved (metrics + fps) and reproducible.

---

## Stage 1) Spec + configuration (source of truth)
Deliverables
- [x] Add a “Symmetry & Commonsense Constraints” checklist section in spec/docs.
- [x] Create `configs/runtime/symmetry.json` (class -> symmetry group, axis, order, notes).
- [x] Create `configs/runtime/constraints.yaml`:
  - tabletop plane parameters (if calibrated): `n, d`
  - depth range priors per class
  - upright constraints per class
  - enable/disable flags per constraint
- [x] Implement config validation and loader wiring.

Tests (set up before implementation)
- [x] Unit: config loads, validation errors for missing/invalid fields.
- [x] Unit: class lookup returns defaults when missing.

Gate
- [x] Config validation passes and is covered by unit tests.

---

## Stage 2) Symmetry-aware learning (must-have)
Deliverables
- [x] Implement symmetry-aware rotation loss:
  - `L_rot_sym = min_{S∈G} d(R_pred, R_gt · S)`
  - Support: `none`, `C2`, `C4`, `C∞(axis)` (yaw-invariant)
- [x] Update pose metrics to be symmetry-aware (ADD-S / symmetry-min geodesic).

Tests (set up before implementation)
- [x] Unit: symmetric object, equivalent `R_gt·S` yield identical loss/metric.
- [x] Unit: non-symmetric object reduces to standard loss/metric.
- [x] Unit: `C∞` yaw-invariance behaves as expected.

Gate
- [x] All symmetry unit tests pass; non-symmetric baselines unchanged.

---

## Stage 3) Symmetry-aware template verification (real-time inference)
Deliverables
- [x] Implement `score_tmp_sym = max_{S∈G} score(render(R_pred·S), obs)` (utility only).
  - Enumerate discrete `S` for `Cn`.
  - For `C∞`, use yaw-agnostic scoring or sample a small set of yaws.
- [x] Integrate into final scoring (utility only):
  - `S = w_det*score_det + w_tmp*score_tmp_sym - w_unc*(sigma_z + sigma_rot)`
- [x] Add low-FP gate (utility only): reject if `score_tmp_sym < τ`.

Tests (set up before implementation)
- [x] Unit: symmetry group enumeration matches expected counts/rotations.
- [x] Integration: for symmetric class, `score_tmp_sym` >= base score.
- [x] Integration: low-FP gate rejects known negatives.

Gate
- [x] FP rate decreases without unacceptable recall loss; fps within target.

---

## Stage 4) Commonsense constraints (train-time + inference)
Deliverables
- [x] Depth prior from CAD size + bbox + corrected intrinsics `K'` (utility only):
  - `z_prior(bbox, size, K')`
  - Regularizer `L_z_prior = |log z - log z_prior|` (low weight)
- [x] Table-plane constraint (if plane is known; utility only):
  - Train-time: `L_plane` (object base point near plane)
  - Inference: reject candidates below plane
- [x] Upright constraint per class (utility only):
  - Train-time: `L_upright` (penalty on roll/pitch outside allowed range)
  - Inference: optional gate or score penalty
- [x] Add ablations toggling each constraint individually.

Tests (set up before implementation)
- [x] Unit: depth prior monotonicity vs bbox size (larger bbox -> smaller z).
- [x] Unit: plane constraint rejects below-plane translations.
- [x] Unit: upright constraint penalizes roll/pitch out of range.
- [x] Integration: toggles enable/disable effects.

Gate
- [x] Each constraint improves its target metric in ablations without hurting fps.

---

## Stage 5) Handheld camera robustness
Deliverables
- [x] Ensure priors/constraints use corrected `K'` and `(u+Δu, v+Δv)`.
- [x] Add synthetic “handheld jitter” profiles in SIM:
  - intrinsics drift `δf, δc`
  - optional mild rolling shutter
  - extrinsics jitter

Tests (set up before implementation)
- [x] Integration: with jitter off, metrics match baseline.
- [x] Integration: with jitter on, degradation is within acceptable bounds.

Gate
- [x] Jittered scenarios show improved robustness vs baseline.

---

## Stage 6) Validation harness (automate)
Deliverables
- [x] Build a “scenario suite” covering:
  - symmetric vs non-symmetric objects
  - tabletop vs non-tabletop
  - extreme depth ranges
  - handheld intrinsics drift on/off
  - template gate on/off
- [x] Produce a single report with:
  - fps (E2E), mAP/Recall
  - depth error
  - pose error (symmetry-aware)
  - rejection rates

Tests (set up before implementation)
- [x] Integration: scenario suite runs end-to-end.
- [x] Integration: report schema is stable and parseable.

Gate
- [x] Suite runs in CI and report outputs are consistent.

---

## Stage 7) Performance + deployment safeguards
Deliverables
- [x] Keep symmetry/commonsense logic out of TensorRT graph where possible.
- [x] Ensure gating logic is optional and lightweight.

Tests (set up before implementation)
- [x] Benchmark: fps within target on reference hardware.

Gate
- [x] No regression in real-time constraints.

---

## Stage 8) Integration skeleton (dataset + adapter)
Deliverables
- [x] Add dataset manifest builder for YOLO-format COCO subset.
- [x] Add adapter interface and dummy adapter for wiring.
- [x] Add scenario runner CLI to exercise the adapter path.

Tests (set up before implementation)
- [x] Unit: manifest builder finds images and labels.
- [x] Integration: scenario runner outputs expected keys.

Gate
- [x] Adapter wiring passes smoke run on tiny COCO.

---

## Implementation order
1. Stage 0 (discovery, baselines)
2. Stage 1 (config + docs)
3. Stage 2 (symmetry loss/metrics)
4. Stage 3 (template verification + gate)
5. Stage 4 (commonsense constraints + ablations)
6. Stage 5 (handheld robustness)
7. Stage 6 (validation harness)
8. Stage 7 (perf + deployment)
9. Stage 8 (integration skeleton)

Notes
- If class-level symmetry metadata is reliable, do not add a symmetry head initially.
- Keep symmetry/commonsense logic out of TensorRT graph whenever possible.
