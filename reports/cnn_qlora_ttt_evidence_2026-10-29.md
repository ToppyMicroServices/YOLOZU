# CNN quantized-LoRA TTT evidence — 2026-10-29

The four-arm experiment completed for seeds 11, 22 and 33. Frozen-base storage decreased from **175,086,688 to 22,434,712 bytes**. The LoRA tensors occupy **524,288 bytes**. Adapter parameters changed, and their saved state could be restored. **TTT produced no AP improvement in any seed**, and the quantized model had lower AP than the float model.

This is a local Research result. Efficacy remains `not_established` and promotion is ineligible. Full measurements and provenance are in [the JSON summary](cnn_qlora_ttt_evidence_2026-10-29.json).

## Experiment

The model is Torchvision Mask R-CNN ResNet50 FPN v2. The checkpoint was loaded strictly before its mask branch was removed for bounding-box prediction. Every arm used CPU FP32 compute. Quantized layers use packed symmetric INT4 weights with transient FP32 dequantization; this is neither the original NF4 QLoRA recipe nor an accelerated INT4 kernel implementation.

Sixteen COCO `val2017` images were selected in filename order: four for adaptation and twelve for evaluation. Support and evaluation are disjoint by image path and bytes. The same images were used for all three seeds; seeds change initialization and training randomness, not the dataset sample. An earlier two-support/four-evaluation probe used the same image pool before the split was expanded with unchanged hyperparameters. The reported experiment is exploratory, not an untouched or preregistered confirmatory test.

The shift multiplies image brightness by 0.55. Teacher pseudo labels are generated once from the shifted support images using each arm's pre-adaptation model. Student updates use an additional brightness multiplier of 0.9 without changing box geometry. No support ground-truth labels are used. Evaluation uses conventional zero-based COCO80 YOLO bounding-box labels and real `pycocotools.COCOeval`; these are subset metrics, not full COCO validation-set results.

LoRA rank is 4 on `backbone.body.layer4` (10 layers, 131,072 trainable parameters). The INT4 scope includes all supported Conv2d and Linear layers (72 layers). Adaptation runs one pass over four images with SGD learning rate 0.0001, pseudo-label confidence 0.7, top-k 20, two CPU threads, minimum image size 256 and maximum size 384.

## Accuracy

Values below are AP50:95 on a 0–100 scale. Adaptation deltas compare each TTT arm with its matching fixed-precision baseline.

| Seed | Arm | Clean AP | Shifted AP | Shifted adaptation delta |
|---|---|---:|---:|---:|
| 11 | `float_fixed` | 64.510027 | 57.722704 | baseline |
| 11 | `int4_fixed` | 53.244695 | 51.741571 | baseline |
| 11 | `float_lora_ttt` | 64.510027 | 57.722704 | 0.000000 |
| 11 | `int4_lora_ttt` | 53.244695 | 51.741571 | 0.000000 |
| 22 | `float_fixed` | 64.510027 | 57.722704 | baseline |
| 22 | `int4_fixed` | 53.244695 | 51.741571 | baseline |
| 22 | `float_lora_ttt` | 64.510027 | 57.722704 | 0.000000 |
| 22 | `int4_lora_ttt` | 53.244695 | 51.741571 | 0.000000 |
| 33 | `float_fixed` | 64.510027 | 57.722704 | baseline |
| 33 | `int4_fixed` | 53.244695 | 51.741571 | baseline |
| 33 | `float_lora_ttt` | 64.510027 | 57.722704 | 0.000000 |
| 33 | `int4_lora_ttt` | 53.244695 | 51.741571 | 0.000000 |

The float model lost 6.787323 AP points under the illumination shift. Quantization lowered clean AP by 11.265332 points and shifted AP by 5.981133 points relative to the float baseline. Neither TTT variant recovered AP under this configuration. The JSON preserves AP50, AP75, AR100 and COCOeval statistics for every seed and arm.

## Storage and adapter updates

| Quantity | Bytes |
|---|---:|
| Float frozen base | 175,086,688 |
| Packed INT4 frozen base | 22,434,712 |
| LoRA parameter tensors | 524,288 |
| Serialized float-base adapter envelope | 532,455 |
| Serialized INT4-base adapter envelope | 540,583 |

The base parameter/buffer storage reduction is 87.1865%. It excludes activation tensors, gradients, optimizer state, transient unpacked weights and runtime overhead. It does not establish a reduction in process peak memory or GPU VRAM. GPU peak VRAM was not measured.

Every adapted arm performed four optimizer updates. Base-parameter and buffer fingerprints remained unchanged, including normalization buffers. The following update norms are nonzero even though AP remained unchanged. Times include teacher pseudo-label generation and optimizer updates.

| Seed | Arm | Updates | Final adapter update norm | Adaptation seconds observed |
|---|---|---:|---:|---:|
| 11 | `float_lora_ttt` | 4 | 0.0000272739 | 14.845 |
| 11 | `int4_lora_ttt` | 4 | 0.0000259779 | 15.654 |
| 22 | `float_lora_ttt` | 4 | 0.0000246242 | 15.137 |
| 22 | `int4_lora_ttt` | 4 | 0.0000209982 | 15.562 |
| 33 | `float_lora_ttt` | 4 | 0.0000301505 | 15.356 |
| 33 | `int4_lora_ttt` | 4 | 0.0000323400 | 15.885 |

Timing was not isolated: a separate-process adapter-restoration check overlapped seed 22. These observations cannot support a latency advantage. Recorded CPU RSS is a process-lifetime high-water mark rather than an isolated per-arm measurement.

Each of the six adapter files passed same-process save/reset/reload prediction parity. Seed 11's INT4 adapter also produced exactly matching predictions for one image after loading into a fresh model in a separate process on the same host. That check does not establish independent-environment or full-set reproducibility.

## Provenance and checks

The checkpoint filename is `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth`, with SHA-256:

```text
73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e
```

The JSON summary records source hashes, all sixteen image filenames and hashes, twelve evaluation-label hashes, raw-report and prediction hashes, and adapter hashes. Raw model weights, images and adapter binaries are not committed with this report. Raw reports remain local, and their hashes identify the measured artifacts.

The focused test command passed 39 tests:

```bash
python -m unittest tests.test_quantized_lora tests.test_cnn_qlora_ttt tests.test_packaged_tools_manifest tests.test_manifest_docs_references tests.test_ttt_docs_implementation_alignment tests.test_manual_ttt_chapter_structure
```

Ruff checks on the five new Python files, declarative tool manifest validation and the new CLI help check also passed. These are local checks; no CI result is claimed here.

## Repeat the experiment

Provide the sixteen images named in the JSON summary under `images/val2017/`, with corresponding standard COCO80 YOLO labels under `labels/val2017/`. Set the dataset and checkpoint paths to local files. Use a new output directory; the runner rejects existing output paths. No implicit download occurs.

```bash
CNN_TTT_DATASET=/path/to/cnn_ttt_coco16
CNN_TTT_CHECKPOINT=/path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth
for experiment_seed in 11 22 33; do
  python tools/run_cnn_qlora_ttt.py \
    --dataset "$CNN_TTT_DATASET" --split val2017 \
    --checkpoint "$CNN_TTT_CHECKPOINT" \
    --output "reports/cnn_qlora_ttt_repeat/seed${experiment_seed}" \
    --support-images 4 --eval-images 12 --steps 1 --rank 4 \
    --learning-rate 0.0001 --pseudo-threshold 0.7 --pseudo-topk 20 \
    --min-size 256 --max-size 384 --brightness 0.55 \
    --seed "$experiment_seed" --device cpu --threads 2 \
    --adapter-include backbone.body.layer4 --quant-include '*'
done
```

The measured runtime was Python 3.14.10, PyTorch 2.12.0.dev20260330 and Torchvision 0.27.0.dev20260330. Exact outputs can depend on the runtime and hardware. This one-host, one-checkpoint, one-shift experiment over twelve evaluation images does not establish real-camera efficacy or a population-level accuracy estimate.
