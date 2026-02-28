# Adapter strategy (inference backends → `predictions.json`)

YOLOZU’s main workflow is:

1) Run inference with your preferred backend
2) Export results into the **canonical** `predictions.json`
3) Validate + evaluate consistently

This page describes the **recommended adapter path** and priorities.

## Priorities (the “one thick road”)

1. **Ultralytics YOLO (v8/v11)**
   - Most common field format.
   - Fast path to high-quality `predictions.json` for detection/seg.

2. **MMDetection**
   - Strong research/production baseline; common in internal stacks.

3. **Detectron2**
   - Widely used for instance segmentation; predictable outputs.

4. **OpenCV DNN (ONNX)**
   - Deployment-friendly baseline for CPU / edge scenarios.

## Official reference adapter (for CI regression)

The official in-repo reference adapter is **`RTDETRPoseAdapter`**.
It is used to pin the adapter interface contract path
(`predict(records) -> entries`) to a reproducible real-image baseline.

Reference regression command:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --profile micro \
  --repro-policy relaxed \
  --runtime-lock requirements-ci.lock \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --diff-summary-out reports/reference_adapter_regression.diff_summary.json \
  --topk-examples-dir reports/reference_adapter_regression_topk \
  --topk-examples 3 \
  --output reports/reference_adapter_regression.json
```

Gates are explicit:
- schema drift (zero tolerance)
- consistency drift (record/image mapping, baseline identity, duplicate/finite checks)
- metric drift (aggregate + robust metrics: map50/map50_95, worst-k, recall@K, IoU quantiles)
- speed drift (minimum FPS floor + baseline ratio)
- optional backend parity drift (peer backend report)
- auto diff artifacts on hard failures (`diff_summary.json`, top-k overlays)

Reference adapter entry metadata is also checked as hard interface contract:
- `image_w/h`, `orig_w/h`, `model_input_w/h`
- `preprocess|preproc` parameters (`resize`, `pad`, `letterbox`, `dtype`, `color_order`)

Adoption policy:
- contract gates (`schema_drift`, `consistency_drift`) are hard
- behavior gates (`metric_drift`, `speed_drift`) start as warn and can be promoted to hard

Details: [reference_adapter_regression_policy.md](reference_adapter_regression_policy.md)

## Contract: what adapters must produce

Adapters should emit the canonical schema:
- `predictions.json` compliant with: [predictions_schema.md](predictions_schema.md)

The point is not “perfectly mirroring a framework’s internal objects”, but producing:
- Stable IDs / image keys
- Boxes/masks/keypoints in agreed coordinates
- Confidence scores
- Category mapping

## Recommended workflow

- Start with: [External inference backends](external_inference.md)
- Validate first:

```bash
yolozu validate predictions --predictions predictions.json
```

- Then evaluate (repo example dataset):

```bash
yolozu eval-coco --dataset data/coco128 --predictions predictions.json
```

## Related docs

- Import adapters (schema-centric): [import_adapters.md](import_adapters.md)
- Real model interface: [real_model_interface.md](real_model_interface.md)
- Evaluation protocol: [yolo26_eval_protocol.md](yolo26_eval_protocol.md)
