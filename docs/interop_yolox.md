# YOLOX interop

Use this path if you want an Apache-2.0-friendly YOLO-style training/export lane.
YOLOZU does not vendor YOLOX itself. Instead, it standardizes dataset resolution,
reporting, and the predictions interface contract around an external YOLOX launcher.

## License boundary

- YOLOZU repository code stays Apache-2.0.
- YOLOX is treated as an external Apache-2.0 training/inference runtime.
- The optional Ultralytics bridge is separate and should be reviewed under its own license terms.

## 5-minute export/eval flow

```bash
yolozu migrate dataset \
  --from coco \
  --coco-root /path/to/coco \
  --split val2017 \
  --output data/coco_yolo_like \
  --mode manifest

python3 -m yolozu export \
  --backend yolox \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --exp /path/to/yolox_exp.py \
  --weights /path/to/yolox_ckpt.pth \
  --imgsz 640 \
  --score-thr 0.01 \
  --nms-iou 0.65 \
  --output reports/pred_yolox.json

python3 tools/validate_predictions.py reports/pred_yolox.json --strict
python3 tools/eval_coco.py \
  --dataset data/coco_yolo_like \
  --split val2017 \
  --predictions reports/pred_yolox.json \
  --protocol nms_applied \
  --classes data/coco_yolo_like/labels/val2017/classes.json \
  --output reports/eval_yolox.json
```

## Important compatibility points

- Exp parameters are captured in `meta.extra.export_settings.exp_params` when exp projection succeeds.
- YOLOX decode is treated as anchor-free grid decode with NMS; use `protocol=nms_applied` by default.
- Preprocess assumptions are stored in `export_settings.preprocessing` (letterbox/normalize/input color).
- `weights_sha256` is stored for reproducibility.

## Training lane (external launcher family)

The primary Apache-2.0-friendly route is:

```bash
python3 -m yolozu train \
  --external-backend yolox \
  configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_yolox.json
```

The equivalent repo-side bridge is:

```bash
python3 tools/support_external_training.py train-yolox \
  --dataset data/coco-yolo \
  --split val \
  --exp configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --dry-run \
  --output reports/support_external_training.train_yolox.json
```

This writes a machine-readable report that includes:

- resolved dataset root and split
- projected YOLOX train config
- the template training command
- `artifact_plan` with expected `predictions.json`, eval report, parity report, and next commands
- the runtime/license boundary for this lane

The dry-run does not execute external training. It validates the artifact plan, confirms the
Apache-2.0-friendly YOLOX runtime boundary, and records the export/eval/parity commands that
consume YOLOX-produced `predictions.json`:

```bash
python3 tools/export_predictions_yolox.py --dataset data/coco-yolo --split val --exp configs/examples/finetune_external/yolox_s_finetune_smoke.py --weights <path/to/yolox_ckpt.pth> --output runs/support_external_training/yolox/reports/yolox_predictions.json
python3 -m yolozu eval-coco --dataset data/coco-yolo --split val --predictions runs/support_external_training/yolox/reports/yolox_predictions.json --output runs/support_external_training/yolox/reports/yolox_eval.json
python3 -m yolozu parity --reference <reference_predictions.json> --candidate runs/support_external_training/yolox/reports/yolox_predictions.json --output runs/support_external_training/yolox/reports/yolox_parity.json
```

To execute a real external launcher, pass your local YOLOX train script:

```bash
python3 tools/support_external_training.py train-yolox \
  --dataset /path/to/coco-yolo \
  --split train \
  --exp /path/to/yolox_exp.py \
  --train-script /path/to/YOLOX/tools/train.py \
  --batch 16 \
  --weights /path/to/yolox_ckpt.pth \
  --output reports/support_external_training.train_yolox.exec.json
```

The helper injects these environment variables for the external launcher:

- `YOLOZU_DATASET_ROOT`
- `YOLOZU_SPLIT`
- `YOLOZU_BATCH_SIZE`
- `YOLOZU_MAX_EPOCHS`
- `YOLOZU_IMAGE_SIZE`

Optional bridges stay on the same top-level train surface:

```bash
python3 -m yolozu train \
  --external-backend ultralytics \
  yolo11n.pt \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_ultralytics.json

python3 -m yolozu train \
  --external-backend hf-detr \
  facebook/detr-resnet-50 \
  --dataset data/smoke \
  --split val \
  --dry-run \
  --output reports/train_external_hf_detr.json
```

These optional bridges keep the runtime/license boundary explicit while reusing the
same dataset resolution and machine-readable reporting surface.
