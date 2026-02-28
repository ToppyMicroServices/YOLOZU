# Reference Adapter Regression Policy

This page defines the test philosophy for `tools/run_reference_adapter_regression.py`.

## Invariants

### Hard invariants (contract)

- `predictions` interface contract schema is valid.
- Image keys are canonicalized and record/prediction mapping is preserved.
- `bbox` uses `cxcywh_norm` with finite values.
- `bbox` constraints: `0<=cx,cy<=1`, `0<w,h<=1`.
- `score` constraints: finite and `0<=score<=1`.
- Duplicate image entries are rejected.
- Detections are stable-sorted by `(-score, class_id, bbox)`.
- `class_id` is required in strict mode.

### Soft invariants (behavior)

- Score drift (`total_detections`, `score_sum`, `score_mean`, `bbox_checksum`) is tolerance-based.
- Performance drift is tolerance-based (`min_fps_ratio`, `absolute_floor_fps`).

Behavior gates are introduced as `warn` first, then promoted to `hard`.

## ReproPolicy

- `strict`: deterministic policy + seeded execution.
- `relaxed`: seeded execution without full deterministic enforcement.
- `off`: speed-first mode.

Every run writes reproducibility metadata (`run_meta`) including:

- seed / policy / backend / device / dtype
- versions (python/torch/onnxruntime/ultralytics/yolozu)
- config/checkpoint/weights hash
- dataset hash
- git SHA

## Baseline lifecycle

`baselines/reference_adapter/rtdetr_pose_smoke_val.json` stores not only results but also:

- gate policy
- protocol contract
- baseline metadata (`baseline_meta`)

Baseline updates must be intentional and reviewed in PR.
