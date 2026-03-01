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
python3 tools/yolozu.py sud --help
```

Layer matrix:

```bash
python3 tools/support_ultralytics_detr.py ls -j
```

Preset shortcuts:
- default preset is `smoke` (or `YOLOZU_CLI_PRESET`)
- supported presets: `smoke`, `coco128`, `none`
- common env fallbacks: `YOLOZU_DATASET`, `YOLOZU_MODEL`, `YOLOZU_HF_MODEL_ID`, `YOLOZU_SPLIT`

## Top 3 support paths

1. Ultralytics YOLO fine-tune (COCO/YOLO input via dataset conversion)

```bash
python3 tools/support_ultralytics_detr.py tu \
  -P smoke \
  -n \
  -o reports/support_ultralytics_detr.train_ultralytics.json
```

2. Hugging Face DETR/RT-DETR entry (Transformers/Datasets bridge)

```bash
python3 tools/support_ultralytics_detr.py th \
  -P smoke \
  -n \
  -o reports/support_ultralytics_detr.train_hf_detr.json
```

For non-dry execution, provide your external train script:

```bash
python3 tools/support_ultralytics_detr.py th \
  -m facebook/detr-resnet-50 \
  -d data/smoke \
  -s val \
  -t /abs/path/to/hf_detr_train.py \
  -o reports/support_ultralytics_detr.train_hf_detr.exec.json
```

3. ONNX export (optional TensorRT)

```bash
python3 tools/support_ultralytics_detr.py eo \
  -P smoke \
  -o models/yolo11n.onnx \
  -n \
  -r reports/support_ultralytics_detr.export_onnx.json
```

Optional TensorRT build handoff:

```bash
python3 tools/support_ultralytics_detr.py eo \
  -p ultralytics \
  -m yolo11n.pt \
  -o models/yolo11n.onnx \
  -t engines/yolo11n_fp16.plan \
  -q fp16 \
  -r reports/support_ultralytics_detr.export_onnx.trt.json
```

## Shared minimal adapters

- Dataset conversion (COCO/YOLO/Ultralytics -> internal wrapper):

```bash
python3 tools/support_ultralytics_detr.py ds \
  -f auto \
  -d data/smoke \
  -s val \
  -o runs/support_ultralytics_detr/dataset \
  -r reports/support_ultralytics_detr.dataset.json
```

- Train template (CLI wrapper):
  - `train-ultralytics` and `train-hf-detr` reports include `template_train_command`.

- Export template (ONNX):
  - `export-onnx` report includes `template_export_command`.

- `predict -> YOLOZU interface contract` normalization:

```bash
python3 tools/support_ultralytics_detr.py pn \
  -i reports/raw_predictions.json \
  -o reports/predictions.normalized.json \
  -r reports/support_ultralytics_detr.predict_normalize.json
```

Ultralytics direct path:

```bash
python3 tools/support_ultralytics_detr.py pn \
  -m yolo11n.pt \
  -d data/smoke \
  -S val \
  -n \
  -o reports/predictions.ultra.normalized.json \
  -r reports/support_ultralytics_detr.predict_normalize.ultra.json
```
