# Bring your own predictions: four checked quickstarts

Use these repo-checkout quickstarts when your model already lives in
Ultralytics, Detectron2, MMDetection, or YOLOX. Each route ends with the same
three artifacts:

1. a wrapped `predictions.json`;
2. a successful strict predictions interface contract check;
3. an `eval_report.json` written by `tools/eval_suite.py`.

The shared validation and evaluation lane is stable. The four inference
exporters remain external-runtime bridges: YOLOZU does not bundle their
frameworks, checkpoints, or framework-specific licenses.

## Confirmed boundary

- A real exporter command does not use `--dry-run`. It must execute inference
  for every selected image or exit nonzero without writing a new success-looking
  artifact.
- A successful real artifact records
  `meta.extra.execution_status="completed"`,
  `meta.extra.runtime_executed=true`, and a positive
  `meta.extra.inference_calls`.
- The CI-checked smoke blocks below intentionally use `--dry-run`. They check
  command syntax, wrapping, strict validation, report generation, and metadata
  handoff without claiming that a third-party runtime executed.
- The smoke report contains `null` metrics because it uses evaluation
  `--dry-run`. Only the real path computes COCO metrics, and that path requires
  `pycocotools`.

## One-time setup

Run from a YOLOZU source checkout. Install the evaluation dependency, then
prepare each framework in the environment where its own import and model load
already work:

```bash
python3 -m pip install -e ".[coco]"
python3 -m yolozu validate dataset /absolute/path/to/yolo-dataset --strict
```

The dataset must have matching `images/<split>/` and `labels/<split>/`
directories. The framework class indices must match the YOLO-format label
indices. Keep model files and datasets outside git.

Every real block below uses a fresh output directory, at most ten images, and a
ten-minute first-run diagnostic budget. Ten minutes is an operator stop
condition, not a performance claim: runtime depends on the model, framework,
hardware, and image sizes. If the command exceeds that budget, stop it and use
the failure route for that framework before attempting a larger run.

## Ultralytics

`tools/export_predictions_yolo_runtime.py` is the declared YOLO-runtime wrapper
around the optional Ultralytics bridge. Review the Ultralytics package and
model license for your use case before installing or running it.

Set `ULTRALYTICS_MODEL` to an existing local model file. This example chooses
the conventional post-NMS lane explicitly; do not copy that setting to an
end-to-end NMS-free model.

<!-- byop-real:ultralytics:start -->
```bash
export BYOP_DATASET="${BYOP_DATASET:-/absolute/path/to/yolo-dataset}"
export BYOP_SPLIT="${BYOP_SPLIT:-val}"
export BYOP_MAX_IMAGES="${BYOP_MAX_IMAGES:-10}"
export BYOP_DEVICE="${BYOP_DEVICE:-cpu}"
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop/ultralytics}"
export ULTRALYTICS_MODEL="${ULTRALYTICS_MODEL:-/absolute/path/to/model.pt}"
test -d "$BYOP_DATASET"
test -f "$ULTRALYTICS_MODEL"
mkdir -p "$BYOP_RUN_DIR"

python3 tools/export_predictions_yolo_runtime.py \
  --model "$ULTRALYTICS_MODEL" \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --max-images "$BYOP_MAX_IMAGES" \
  --device "$BYOP_DEVICE" \
  --image-size 640 \
  --conf 0.25 \
  --iou 0.45 \
  --max-det 300 \
  --no-end2end \
  --protocol nms_applied \
  --wrap \
  --strict \
  --output "$BYOP_RUN_DIR/predictions.json"

python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images "$BYOP_MAX_IMAGES" \
  --bbox-format cxcywh_norm \
  --strict \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-real:ultralytics:end -->

Expected files:

- `reports/byop/ultralytics/predictions.json`
- `reports/byop/ultralytics/eval_report.json`

Failure route:

- `ultralytics package is required`: activate the environment in which
  `python3 -c "import ultralytics"` succeeds.
- `source not found`: verify `BYOP_DATASET`, `BYOP_SPLIT`, and the
  `images/<split>/` directory.
- zero inference calls or an image-count mismatch: keep the failed output out
  of comparisons and inspect the model's input/source behavior.

### CI-checked Ultralytics schema smoke

This block does not load the model or execute Ultralytics. The test suite runs
this exact block with a per-source 120-second timeout.

<!-- byop-smoke:ultralytics:start -->
```bash
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop-smoke/ultralytics}"
mkdir -p "$BYOP_RUN_DIR"
python3 tools/export_predictions_yolo_runtime.py \
  --model /path/to/ultralytics_model.pt \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --device cpu \
  --image-size 640 \
  --conf 0.25 \
  --iou 0.45 \
  --max-det 300 \
  --no-end2end \
  --protocol nms_applied \
  --wrap \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/predictions.json"
python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset data/smoke \
  --split val \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images 2 \
  --bbox-format cxcywh_norm \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-smoke:ultralytics:end -->

## Detectron2

Prepare a Detectron2 environment that can import the config's registered
components. Set both file paths; a real run fails closed before inference if
either file is missing.

<!-- byop-real:detectron2:start -->
```bash
export BYOP_DATASET="${BYOP_DATASET:-/absolute/path/to/yolo-dataset}"
export BYOP_SPLIT="${BYOP_SPLIT:-val}"
export BYOP_MAX_IMAGES="${BYOP_MAX_IMAGES:-10}"
export BYOP_DEVICE="${BYOP_DEVICE:-cpu}"
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop/detectron2}"
export DETECTRON2_CONFIG="${DETECTRON2_CONFIG:-/absolute/path/to/detectron2_config.yaml}"
export DETECTRON2_WEIGHTS="${DETECTRON2_WEIGHTS:-/absolute/path/to/model_final.pth}"
test -d "$BYOP_DATASET"
test -f "$DETECTRON2_CONFIG"
test -f "$DETECTRON2_WEIGHTS"
mkdir -p "$BYOP_RUN_DIR"

python3 tools/export_predictions_detectron2.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --config "$DETECTRON2_CONFIG" \
  --weights "$DETECTRON2_WEIGHTS" \
  --max-images "$BYOP_MAX_IMAGES" \
  --device "$BYOP_DEVICE" \
  --score-thr 0.25 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --output "$BYOP_RUN_DIR/predictions.json"

python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images "$BYOP_MAX_IMAGES" \
  --bbox-format cxcywh_norm \
  --strict \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-real:detectron2:end -->

Expected files:

- `reports/byop/detectron2/predictions.json`
- `reports/byop/detectron2/eval_report.json`

Failure route:

- config merge or model initialization failure: run the config and checkpoint
  in the original Detectron2 project first, including any custom registrations.
- unreadable image: resolve the path recorded in the dataset manifest from the
  dataset root.
- completed inference but wrong classes: verify the contiguous Detectron2 class
  order against `labels/<split>/classes.json` before comparing metrics.

### CI-checked Detectron2 schema smoke

<!-- byop-smoke:detectron2:start -->
```bash
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop-smoke/detectron2}"
mkdir -p "$BYOP_RUN_DIR"
python3 tools/export_predictions_detectron2.py \
  --dataset data/smoke \
  --split val \
  --config /path/to/detectron2_config.yaml \
  --weights /path/to/model_final.pth \
  --max-images 2 \
  --device cpu \
  --score-thr 0.25 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/predictions.json"
python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset data/smoke \
  --split val \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images 2 \
  --bbox-format cxcywh_norm \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-smoke:detectron2:end -->

## MMDetection

Prepare an MMDetection environment that can initialize the supplied config and
checkpoint together. YOLOZU does not install MMDetection or resolve its
version-specific runtime dependencies.

<!-- byop-real:mmdetection:start -->
```bash
export BYOP_DATASET="${BYOP_DATASET:-/absolute/path/to/yolo-dataset}"
export BYOP_SPLIT="${BYOP_SPLIT:-val}"
export BYOP_MAX_IMAGES="${BYOP_MAX_IMAGES:-10}"
export BYOP_DEVICE="${BYOP_DEVICE:-cpu}"
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop/mmdetection}"
export MMDET_CONFIG="${MMDET_CONFIG:-/absolute/path/to/mmdet_config.py}"
export MMDET_CHECKPOINT="${MMDET_CHECKPOINT:-/absolute/path/to/checkpoint.pth}"
test -d "$BYOP_DATASET"
test -f "$MMDET_CONFIG"
test -f "$MMDET_CHECKPOINT"
mkdir -p "$BYOP_RUN_DIR"

python3 tools/export_predictions_mmdet.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --config "$MMDET_CONFIG" \
  --checkpoint "$MMDET_CHECKPOINT" \
  --max-images "$BYOP_MAX_IMAGES" \
  --device "$BYOP_DEVICE" \
  --score-thr 0.25 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --output "$BYOP_RUN_DIR/predictions.json"

python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images "$BYOP_MAX_IMAGES" \
  --bbox-format cxcywh_norm \
  --strict \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-real:mmdetection:end -->

Expected files:

- `reports/byop/mmdetection/predictions.json`
- `reports/byop/mmdetection/eval_report.json`

Failure route:

- runtime initialization failure: confirm the exact config/checkpoint pair with
  the original MMDetection environment before invoking the wrapper.
- unsupported prediction sample shape: inspect the MMDetection data sample
  produced by that model family; do not treat a failed conversion as empty
  detections.
- wrong class order: align the MMDetection dataset metadata with the YOLO label
  indices.

### CI-checked MMDetection schema smoke

<!-- byop-smoke:mmdetection:start -->
```bash
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop-smoke/mmdetection}"
mkdir -p "$BYOP_RUN_DIR"
python3 tools/export_predictions_mmdet.py \
  --dataset data/smoke \
  --split val \
  --config /path/to/mmdet_config.py \
  --checkpoint /path/to/checkpoint.pth \
  --max-images 2 \
  --device cpu \
  --score-thr 0.25 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/predictions.json"
python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset data/smoke \
  --split val \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images 2 \
  --bbox-format cxcywh_norm \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-smoke:mmdetection:end -->

## YOLOX

YOLOZU treats YOLOX as an external Apache-2.0 runtime and does not vendor it.
Set both the experiment and checkpoint paths. The exporter projects experiment
parameters before it runs inference and records their provenance.

<!-- byop-real:yolox:start -->
```bash
export BYOP_DATASET="${BYOP_DATASET:-/absolute/path/to/yolo-dataset}"
export BYOP_SPLIT="${BYOP_SPLIT:-val}"
export BYOP_MAX_IMAGES="${BYOP_MAX_IMAGES:-10}"
export BYOP_DEVICE="${BYOP_DEVICE:-cpu}"
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop/yolox}"
export YOLOX_EXP="${YOLOX_EXP:-/absolute/path/to/yolox_exp.py}"
export YOLOX_WEIGHTS="${YOLOX_WEIGHTS:-/absolute/path/to/yolox_checkpoint.pth}"
test -d "$BYOP_DATASET"
test -f "$YOLOX_EXP"
test -f "$YOLOX_WEIGHTS"
mkdir -p "$BYOP_RUN_DIR"

python3 tools/export_predictions_yolox.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --exp "$YOLOX_EXP" \
  --weights "$YOLOX_WEIGHTS" \
  --max-images "$BYOP_MAX_IMAGES" \
  --device "$BYOP_DEVICE" \
  --imgsz 640 \
  --score-thr 0.25 \
  --nms-thr 0.45 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --output "$BYOP_RUN_DIR/predictions.json"

python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset "$BYOP_DATASET" \
  --split "$BYOP_SPLIT" \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images "$BYOP_MAX_IMAGES" \
  --bbox-format cxcywh_norm \
  --strict \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-real:yolox:end -->

Expected files:

- `reports/byop/yolox/predictions.json`
- `reports/byop/yolox/eval_report.json`

Failure route:

- experiment projection failure: import the experiment file in the original
  YOLOX checkout and resolve its project-local imports.
- checkpoint load mismatch: verify the checkpoint against that exact experiment
  and architecture.
- inference or image-read failure: retain the stderr evidence and do not reuse a
  previous `predictions.json` as evidence for the failed run.

### CI-checked YOLOX schema smoke

<!-- byop-smoke:yolox:start -->
```bash
export BYOP_RUN_DIR="${BYOP_RUN_DIR:-reports/byop-smoke/yolox}"
mkdir -p "$BYOP_RUN_DIR"
python3 tools/export_predictions_yolox.py \
  --dataset data/smoke \
  --split val \
  --exp /path/to/yolox_exp.py \
  --weights /path/to/yolox_checkpoint.pth \
  --max-images 2 \
  --device cpu \
  --imgsz 640 \
  --score-thr 0.25 \
  --nms-thr 0.45 \
  --topk 300 \
  --protocol nms_applied \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/predictions.json"
python3 tools/validate_predictions.py "$BYOP_RUN_DIR/predictions.json" --strict
python3 tools/eval_suite.py \
  --dataset data/smoke \
  --split val \
  --predictions-glob "$BYOP_RUN_DIR/predictions.json" \
  --max-images 2 \
  --bbox-format cxcywh_norm \
  --strict \
  --dry-run \
  --output "$BYOP_RUN_DIR/eval_report.json"
```
<!-- byop-smoke:yolox:end -->

## Decide whether reports are actually comparable

All four routes produce the same report schema, but equal schema does not prove
equal evaluation conditions. Before comparing metrics:

1. confirm identical dataset content, split, image identities, and class order;
2. inspect each result's `export_settings` in `eval_report.json`;
3. align image size, score threshold, NMS/IoU policy, maximum detections,
   preprocessing, and bounding-box representation;
4. use the same named evaluation protocol only when the exporter settings
   satisfy that protocol's fixed conditions;
5. do not compare reports with different protocol hashes.

Detectron2 and MMDetection preprocessing is config-owned. The exporter records
the declared pipeline metadata but does not rewrite the framework config.
Therefore, changing a metadata flag is not a substitute for changing and
verifying the real preprocessing pipeline.

## Expected evidence checklist

For each real run, retain:

- the exact command and environment/package versions;
- `predictions.json`;
- `eval_report.json`;
- model/config/checkpoint paths and hashes from
  `meta.extra.model_provenance`;
- `execution_status`, `runtime_executed`, and `inference_calls`;
- dataset identity, split, class mapping, and the chosen evaluation protocol;
- stderr and the nonzero exit code for any failed run.

The repository tests execute the four schema-smoke blocks, strict validation,
and evaluation-report generation. Separate exporter tests use controlled fake
runtimes to exercise non-dry success and fail-closed behavior. Neither test
class is evidence that your real third-party runtime, model, dataset, or
hardware completed successfully.
