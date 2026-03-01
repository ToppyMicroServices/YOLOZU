# Ultralytics/DETR support (3-layer)

YOLOZU now exposes a fixed 3-layer support surface for Ultralytics/DETR workflows:

1. `trainer_runner`
- framework runtime entrypoint (`ultralytics.YOLO.train`, Transformers/accelerate script bridge)

2. `repo_impl`
- repository integration wrapper (`tools/support_ultralytics_detr.py`)

3. `export_deploy`
- ONNX export + optional TensorRT handoff

## Tool entrypoint

```bash
python3 tools/support_ultralytics_detr.py --help
```

Layer matrix:

```bash
python3 tools/support_ultralytics_detr.py layers --json
```

## Top 3 support paths

1. Ultralytics YOLO fine-tune (COCO/YOLO input via dataset conversion)

```bash
python3 tools/support_ultralytics_detr.py train-ultralytics \
  --model yolo11n.pt \
  --from internal \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/support_ultralytics_detr.train_ultralytics.json
```

2. Hugging Face DETR/RT-DETR entry (Transformers/Datasets bridge)

```bash
python3 tools/support_ultralytics_detr.py train-hf-detr \
  --model-id facebook/detr-resnet-50 \
  --from internal \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/support_ultralytics_detr.train_hf_detr.json
```

For non-dry execution, provide your external train script:

```bash
python3 tools/support_ultralytics_detr.py train-hf-detr \
  --model-id facebook/detr-resnet-50 \
  --dataset data/smoke \
  --split val \
  --train-script /abs/path/to/hf_detr_train.py \
  --output reports/support_ultralytics_detr.train_hf_detr.exec.json
```

3. ONNX export (optional TensorRT)

```bash
python3 tools/support_ultralytics_detr.py export-onnx \
  --provider ultralytics \
  --model yolo11n.pt \
  --output models/yolo11n.onnx \
  --dry-run \
  --report reports/support_ultralytics_detr.export_onnx.json
```

Optional TensorRT build handoff:

```bash
python3 tools/support_ultralytics_detr.py export-onnx \
  --provider ultralytics \
  --model yolo11n.pt \
  --output models/yolo11n.onnx \
  --trt-engine engines/yolo11n_fp16.plan \
  --trt-precision fp16 \
  --report reports/support_ultralytics_detr.export_onnx.trt.json
```

## Shared minimal adapters

- Dataset conversion (COCO/YOLO/Ultralytics -> internal wrapper):

```bash
python3 tools/support_ultralytics_detr.py dataset \
  --from auto \
  --dataset data/smoke \
  --split val \
  --output runs/support_ultralytics_detr/dataset \
  --report reports/support_ultralytics_detr.dataset.json
```

- Train template (CLI wrapper):
  - `train-ultralytics` and `train-hf-detr` reports include `template_train_command`.

- Export template (ONNX):
  - `export-onnx` report includes `template_export_command`.

- `predict -> YOLOZU interface contract` normalization:

```bash
python3 tools/support_ultralytics_detr.py predict-normalize \
  --input reports/raw_predictions.json \
  --output reports/predictions.normalized.json \
  --report reports/support_ultralytics_detr.predict_normalize.json
```

Ultralytics direct path:

```bash
python3 tools/support_ultralytics_detr.py predict-normalize \
  --ultralytics-model yolo11n.pt \
  --dataset data/smoke \
  --split val \
  --ultralytics-dry-run \
  --output reports/predictions.ultra.normalized.json \
  --report reports/support_ultralytics_detr.predict_normalize.ultra.json
```
