# Detection-native TTT local evidence — 2026-08-01

Status: **positive local diagnostic; not independently reproduced; efficacy not established**.

## Confirmed execution

- Model: fully compatible YOLO26n RT-DETR pose checkpoint, SHA-256
  `f07f27c5191e974798744408eb7b361317f6ca6880e53eb0612d997f2e8e506b`
- Config: `configs/yolo26_rtdetr_pose/yolo26n.json`, SHA-256
  `4ef6acb77205b6c8ad18a07786030109e48ca99f176b415406278e3ee724520f`
- Protocol: sample reset, 3 steps, normalization-only update, learning rate
  `5e-5`, foreground confidence `0.2`, top-k 20, score threshold `0.1`
- Evaluation: real `pycocotools` COCO AP, 10 images, CPU, input 320
- Clean content hash: `0280a79bdcf4813f5d14698757a83a293d53adef05e63711f0c878ea68536453`
- Shifted content hash: `d333513c7988734b46c9aca2135c535d17007d43b85e168be8b96efb25820122`

The loss excludes the final no-object class, selects foreground queries that
beat both the confidence threshold and no-object probability, and uses
same-query foreground-class KL, sigmoid-box consistency, and selected-query
entropy across a deterministic weak photometric view.

## Before / after

| Dataset | Baseline mAP50:95 | Adapted mAP50:95 | Delta | Baseline mAP50 | Adapted mAP50 |
|---|---:|---:|---:|---:|---:|
| clean | 0.0009900990 | 0.0011881188 | +0.0001980198 | 0.0033003300 | 0.0039603960 |
| shifted | 0.0003300330 | 0.0003960396 | +0.0000660066 | 0.0033003300 | 0.0039603960 |

All 30 adaptation steps completed without rollback, non-finite values, or a
guard stop. The later abstention rule defaults to a minimum of one selected
query; the recorded minimum was 12, so no recorded step would have abstained.
Mean selected queries per step were 14.47 on clean images and
15.40 on shifted images. Mean selected foreground confidence was 0.3523 and
0.3769, respectively.

## Reproduce locally

```bash
./.venv/bin/python tools/run_ttt_compare.py \
  --method detector_response \
  --data data/smoke \
  --weights /path/to/the-hash-matched-checkpoint.pt \
  --out reports/ttt_compare/detector_response \
  --image-size 320 --score-threshold 0.1 \
  --max-detections 100 --dataset-hash-mode content --force
```

The local raw clean and shifted compare JSON files had SHA-256
`656e1cfd349f69447f1facf8824f058dfa87a279b30f52a5205f9cae3a99bbd9`
and `0f17c0411b9c65fec1c6762fb23cea77382a049eb9ea6d20094b74cff57259e7`.
They are diagnostic outputs rather than checked-in SSOT artifacts.

## Boundary

The checkpoint and shifted fixture live in an ignored historical diagnostic
directory and are not a clean-checkout reproduction bundle. The result uses one
checkpoint, one small dataset, and one environment. It is a useful positive
signal for further preregistration, but it does not establish TTT efficacy or
promote the Research lane.
