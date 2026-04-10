# YOLO26 buckets (RT-DETR reference-trainer recipes)

This repo includes an Apache-2.0-friendly RT-DETR reference trainer (`rtdetr_pose/`) that can be used as the YOLO26
"bucketed" detector family for smoke / export / evaluation plumbing.

The bucket configs live under `configs/yolo26_rtdetr_pose/`:

| Bucket | Config |
|---|---|
| n | `configs/yolo26_rtdetr_pose/yolo26n.json` |
| s | `configs/yolo26_rtdetr_pose/yolo26s.json` |
| m | `configs/yolo26_rtdetr_pose/yolo26m.json` |
| l | `configs/yolo26_rtdetr_pose/yolo26l.json` |
| x | `configs/yolo26_rtdetr_pose/yolo26x.json` |

## Smoke run (CPU)

Fetch `data/coco128` and run the end-to-end chain for one or more buckets:

```bash
bash tools/fetch_coco128.sh
python3 tools/run_yolo26_smoke_rtdetr_pose.py --buckets n
python3 tools/run_yolo26_smoke_rtdetr_pose.py --buckets n,s,m,l,x --max-steps 1 --image-size 64
```

Artifacts are written to `runs/yolo26_smoke/<utc>/yolo26{bucket}/`:
- `checkpoint.pt`
- `model.onnx`
- `pred_yolo26{bucket}.json` (wrapped predictions payload)
- `eval_suite_dry_run.json` (schema + conversion check)

## Training notes (real runs)

These configs are intended as a starting point for "bucket sizing" only.
For real training runs, expect to tune at least:
- `--image-size` (larger for s/m/l/x)
- `--batch-size` / `--grad-accum`
- `--max-steps` or `--epochs`
- optimizer/scheduler flags (see `rtdetr_pose/tools/train_minimal.py --help`)

## Export/eval naming (protocol alignment)

For protocol-locked eval on COCO, the default naming convention is:
- predictions: `pred_yolo26{bucket}.json`
- suite: `eval_suite*.json`

`tools/eval_suite.py` uses `protocols/yolo26_eval.json` to pin:
- bbox format
- default split selection
