# Fine-Tuning Lane Qualification Evidence (2026-07-29)

## Decision

- Protocol completion: `true`
- Promotion decision: `hold`
- Maturity: `experimental`
- Training quality: `not_established`

Exit code 0 means that every required attempt emitted machine-readable
evidence. It does not mean that every framework trained or that the lane passed
promotion.

## Reproduction

Source:

- Commit: `fe9b9bf612cc37c85b30588680743893560f409b`
- Tracked changes during execution: none
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Python: `3.14.6`
- Selected interpreter: `./.venv/bin/python`

Command:

```bash
./.venv/bin/python tools/qualify_finetune_lanes.py \
  --output-dir /tmp/yolozu_finetune_qualification_20260729_fe9b9bf \
  --python ./.venv/bin/python \
  --device cpu \
  --epochs 1 \
  --max-steps 4 \
  --batch-size 2 \
  --image-size 128
```

The output directory must not already exist. It is intentionally outside the
repository because the bounded run emitted about 946 MB of checkpoints and
supporting artifacts.

## Dataset and supervision provenance

- Dataset: `data/real_multitask_fewshot`
- Dataset tree SHA-256:
  `8f6840184d6400419dcbbaf0f2a34e3f23b94c89eb04d981dbb015988bbe11af`
- `prepare_summary.json` SHA-256:
  `44bbb9d7df60f1986b952aff06da1e985433d0e98349ad1d7fecc08401d56658`
- Model-inference-generated labels: `false`

| Stage | Supervision source | Strict-realism eligible |
|---|---|---:|
| bbox | `coco_instances_gt` | yes |
| segmentation | `coco_polygon_gt` | yes |
| keypoints | `bbox_derived_anchors` | no |
| depth | `bbox_scale_heuristic` | no |
| pose6d | `bbox_depth_intrinsics_heuristic` | no |

## Repository-local staged execution

All five stages returned success and recorded a matching requested checkpoint
handoff. The only reported validation metric was bbox mAP50-95. A before metric
was not emitted for any stage.

| Stage | Training | Task-native metric | After bbox mAP50-95 | Wall seconds | Peak child RSS |
|---|---:|---:|---:|---:|---:|
| bbox | yes | yes | 0.0 | 3.161 | 536,625,152 B |
| segmentation | yes | no | 0.0 | 2.706 | 538,836,992 B |
| keypoints | yes | no | 0.0 | 2.622 | 549,224,448 B |
| depth | yes | no | 0.0 | 2.602 | 549,486,592 B |
| pose6d | yes | no | 0.0 | 2.621 | 549,748,736 B |

Checkpoint bundle SHA-256 values:

```text
bbox          7ed5a5edfe6345aa75db7bc1c243c7e53bad6562ebbe2789a4d63f27391a43bf
segmentation  c95b7afa7e4a08bed1898d13d73b3c1331d2c935ea401335527ee11a1d3c2e45
keypoints     97f5011244388da60d9d6e3789b888ccba96f2bc9cdc941b1e1dde992026a6c4
depth         7bd8c1c5296ba1706a806105c436cfd3e3f74fe8db876a5b7e2166d100f72434
pose6d        9dd1591c3275ecebd0ed7522562a60c46fb24272dc2d5343b8632cd3294d800f
```

The real-stage report SHA-256 was
`f1012bc2a94a465128d73e57556f8a71834ee0f53ef94467a65e6700ea79c541`.

## External execution matrix

The matrix deliberately selected all five frameworks as non-dry. Config
projection is not counted as training.

| Framework | Actual training | Result |
|---|---:|---|
| RT-DETR | yes | completed |
| YOLOX | no | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |
| YOLOv | no | `E_DEP_ULTRALYTICS_MISSING` |
| MMDetection | no | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |
| Detectron2 | no | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |

The external-matrix process returned 1 because the selected
`--require-training-execution` condition was not satisfied by all frameworks.
Its report was still complete and had SHA-256
`e62ea66e1f69a605c38b5a07e817c6f545c3e73a800b5e4a304829c427058602`.

Additional advertised-provider attempts were also machine-readable:

| Provider | State | Failure code |
|---|---|---|
| MMPose | `requires_external_train_script` | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |
| MMSeg | `requires_external_train_script` | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |
| TAO | `runtime_failed` | `E_EXTERNAL_RUNTIME_MISSING` |
| HF DETR | `requires_external_train_script` | `E_EXTERNAL_TRAIN_SCRIPT_REQUIRED` |

Runtime probes found `torch` and `transformers`. They did not find
Ultralytics, YOLOX, MMEngine, MMDetection, MMPose, MMSeg, Detectron2, or the TAO
CLI in the selected environment.

## Artifact integrity

- Qualification summary SHA-256:
  `9403c0967699277958b95416ea83bab811ac497754a378440c83e5bdddd95589`
- Human-readable qualification report SHA-256:
  `f79dad1c4cb21cc0c0eb095f3027e3163410df0cf4fe85f5923151edf54981e0`
- Checksum manifest SHA-256:
  `ce8627ca54ac2a0cc3d31aa0f9f12b994a81f37dde3bd0b0ec7c362258d0c9b4`
- Files covered by `checksums.sha256`: 111
- Verification result: 111/111 matched

The 946 MB local bundle is not a release artifact. The tracked command, source
commit, dataset hashes, report hashes, and checkpoint hashes provide the
reproduction pointer without treating a bounded execution fixture as a
performance release.

## Promotion blockers

The current evidence does not establish fine-tuning quality:

1. Segmentation, keypoints, depth, and pose6d lack before/after task-native
   metrics in the staged runner.
2. Keypoint, depth, and pose6d fixture labels are heuristic.
3. The two-image validation split is an execution fixture, not efficacy
   evidence.
4. Only the repository-local RT-DETR external-matrix lane executed training in
   this environment.
5. Independent reproduction has not been recorded.

Reconsider promotion only after license-cleared non-heuristic labels,
task-native before/after evaluation, executed external runtimes, and an
independent reproduction are available.
