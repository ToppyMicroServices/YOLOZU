# TTT readiness audit — 2026-07-26

## Confirmed

- TTT remains a Research lane. A Stable parent command does not promote TTT
  methods, presets, checkpoints, or evidence.
- `ttt_job` and `ctta_job` require a repository-confined dataset, config,
  checkpoint, and output. They reject missing or incompatible inputs before a
  job is queued.
- Checkpoint preflight accepts only the canonical loader result
  `status=full` with `load.loaded=true`.
- The runnable YOLOZU method profiles explicitly distinguish implementation
  behavior from paper/reference fidelity. The current profiles do not establish
  efficacy.
- The checked-in public figure source is
  `evidence_kind=synthetic_fixture`, `promotion_eligible=false`, and contains no
  measured efficacy values.

## Observed

- A live CPU integration test creates a current tiny RT-DETR checkpoint, queues
  both TTT and CTTA through the public job layer, waits for terminal state, and
  verifies exit status, predictions, TTT reports, and persisted job state.
- The fail-closed compare tests cover missing, empty, partial, incompatible, and
  unloaded checkpoints before a successful plan is emitted.
- The historical bundled checkpoint audit matches 20 of 308 model-state
  tensors. Its model-parameter coverage is `0.03613405415670097`, so it is not
  current full-checkpoint evidence. The source is
  `reports/rtdetr_pose_coco128_gpu_matcher_historical.json`.

## Unknown / risk

- There is no current tracked or release-addressable measured bundle that runs
  clean baseline and every enabled method on the same deterministic target
  domains for at least three seeds.
- There is no complete evidence set for COCO AP50:95, worst-domain AP, clean
  retention, collapse/calibration, update ratio, latency, memory, and
  forward/backward counts.
- Sample-reset and continual-stream protocols have not yet been reproduced as
  separate comparable benchmark bundles.
- A current MIM-enabled, fully compatible checkpoint with complete provenance is
  not present.
- Therefore TTT effectiveness, a recommended default method, and comparative
  ranking remain unknown.

## Recommendation

Keep TTT opt-in and Research-only. Publish the runnable diagnostics, safety
guards, method-fidelity labels, and evidence requirements, but do not publish an
efficacy chart or promote a default method. Complete `YOLOZU-ll2.53` before
starting comparative candidate selection in `YOLOZU-ll2.54`.

## Change trigger

Revisit the efficacy conclusion only when a current-compatible,
provenance-bound bundle satisfies every acceptance field in
`YOLOZU-ll2.53`, is independently reproduced, and remains comparable under one
preregistered protocol. Revisit method expansion only after that bundle exists
and each candidate has a primary-source, license, RT-DETR applicability, and
compute-budget decision.

## Confidence

- Safety and execution-boundary conclusion: high.
- Current-efficacy conclusion (`not_established`): high.
- Relative method quality or production suitability: unavailable from current
  evidence.
