# CNN quantized LoRA test-time training

This Research command compares a pretrained CNN detector under four conditions:
floating weights, packed INT4 weights, floating weights with LoRA adaptation,
and packed INT4 weights with LoRA adaptation. It uses local checkpoints and
images and does not download models or datasets.

The first supported model is torchvision Mask R-CNN ResNet-50 FPN v2. The full
checkpoint is loaded strictly before its mask branch is disabled. This command
measures bounding-box detection; it does not evaluate or adapt segmentation.

The [2026-10-29 local result](../reports/cnn_qlora_ttt_evidence_2026-10-29.md)
records a completed four-condition comparison over three seeds. Packed storage,
adapter updates, and restore checks worked, but the tested settings produced no
AP gain, and INT4 reduced baseline AP. This remains an exploratory Research result.

## Run a bounded comparison

Supply the torchvision `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` checkpoint
and a YOLO-format dataset with the standard COCO 80-class mapping:

```bash
python3 tools/run_cnn_qlora_ttt.py \
  --checkpoint /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --dataset /path/to/coco-yolo --split val2017 \
  --output reports/cnn_qlora_ttt_seed11 --seed 11
```

Use `--help` for the bounded adaptation and evaluation counts, image size,
and device options. Each run needs a new output directory.
The reference implementation uses FP32 on CPU and CUDA. Report the actual compute dtype; a CPU FP32 comparison
must not be described as an FP16 benchmark.

For a fixed three-seed local study with four support images and twelve held-out
images, use separate output directories and keep all hyperparameters unchanged:

```bash
for seed in 11 22 33; do
  python3 tools/run_cnn_qlora_ttt.py \
    --checkpoint /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
    --dataset /path/to/coco-yolo --split val2017 \
    --support-images 4 --eval-images 12 --seed "$seed" \
    --output "reports/cnn_qlora_ttt/seed${seed}"
done
```

The defaults use rank 4, one pass over support images, learning rate `1e-4`,
brightness multiplier `0.55`, and image resize bounds `256/384`. These are
bounded starting settings, not a validated optimum. The comparison runs
sequentially and includes cold inference. Treat latency as a local diagnostic,
not an isolated hardware benchmark.

## What is quantized

The reference implementation packs two signed four-bit weight values into each
byte and stores one scale per output channel. Frozen convolutional and linear
weights are dequantized transiently for floating-point computation. LoRA
parameters remain floating point and receive gradients through the frozen base.
The layer report records actual packed storage and the selected adapter scope.

This is a quantized-base LoRA implementation. It does not reproduce the NF4,
double-quantization, or paged-optimizer recipe from the original QLoRA paper.
It also does not provide an optimized INT4 convolution kernel. Lower parameter
storage therefore does not establish lower peak memory or faster inference.

The existing trainer `--qlora` uses a separate experimental torchao integration.
It does not select this reference implementation or establish CNN INT4 coverage.

## Adaptation and evaluation

The runner selects disjoint adaptation and evaluation images before executing
the four conditions. Adaptation uses only shifted images and fixed predictions
from the model before updating. Confident teacher boxes become pseudo targets;
missing pseudo targets cause an explicit skipped update. No ground-truth label
is passed to adaptation.

The objective is detector self-training on pseudo boxes with a mild additional
brightness change. It is a baseline for this comparison, not a reproduction of
the original rotation-prediction TTT method or the existing Tent/SAR profiles.

Only adapter parameters are optimized. Normalization statistics and the base
remain fixed, and the report checks base invariance. The saved adapter is tied
to the base fingerprint and quantization recipe so incompatible restores fail.
Adapter reload is checked against predictions from the adapted model.

Evaluation uses ground-truth labels on held-out images, with clean and shifted
views reported separately. The imposed illumination change is synthetic; it is
not evidence of adaptation to real camera changes. Compare each adapted arm to
its matching non-adapted baseline, and also compare against the floating model.
Quantization loss and adaptation gain are separate effects.

## Restore a saved adapter

Rebuild the same local base checkpoint and layer recipe before loading an adapter.
The following matches the default INT4 arm. It performs no download:

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from yolozu.tta.quantized_lora import prepare_model, load_adapter_state_dict

model = maskrcnn_resnet50_fpn_v2(
    weights=None, weights_backbone=None, min_size=256, max_size=384,
    box_detections_per_img=100,
)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
model.roi_heads.mask_roi_pool = None
model.roi_heads.mask_head = None
model.roi_heads.mask_predictor = None
prepare_model(model, quantize=True, rank=4,
              adapter_include=("backbone.body.layer4",), quant_include=("*",))
load_adapter_state_dict(model, torch.load(adapter_path, map_location="cpu", weights_only=True))
model.eval()
```

Here `checkpoint_path` is the original local checkpoint and `adapter_path` is
`<output>/int4_lora_ttt/adapter.pt`. The runner verifies saved-byte reload in
the same model after clearing its adapters. That check is distinct from an
independent application or hardware deployment test.

## Reading the result

Read AP alongside detection counts, skipped updates, adapter reload parity,
parameter storage, runtime, and available process/device memory measurements.
CPU process peak RSS and CUDA peak allocation measure different quantities.
Do not treat cumulative process RSS as an isolated per-condition peak.

A run with no updates, failed reload, or changed base parameters cannot establish
adapter learning. A lower adaptation loss alone is not accuracy evidence.
An exploratory result on a small image subset cannot establish field efficacy.
Use a fixed protocol and independent sequences before making a product claim.

The intended follow-up is camera-specific adapter creation and restoration,
with visible before/after detections and resource measurements. Live video
capture and automatic production promotion are outside this command.
