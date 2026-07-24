# RT-DETR checkpoint compatibility

YOLOZU's public RT-DETR adapter, ONNX/TensorRT exporter, and backend suite use
one fail-closed loader:
`yolozu.inference.checkpoint_compatibility.load_checkpoint_compatible`.

## Default policy

A checkpoint is `status=full` only when every current model-state key has a
same-name tensor with the same shape and the checkpoint has no unexpected
tensor or non-tensor entries. The loader then calls `load_state_dict` with
`strict=True`.

The only supported legacy key normalization is removal of a uniform
`module.` or `_orig_mod.` prefix when doing so increases overlap with the
current model. No architecture-specific key guessing or shape coercion is
performed.

Name mismatches, shape mismatches, missing keys, unexpected keys, unsupported
checkpoint containers, and zero-match checkpoints fail before model mutation.
The default deserializer uses `torch.load(..., weights_only=True)`, so the
checkpoint is treated as untrusted tensor/state-dict input rather than a
general Python pickle. Archives that the restricted deserializer cannot read
also fail before model mutation.

Reported file paths are descriptive run context, not content identity.
Checkpoint/config SHA-256 values and the model-state signature are the
identity evidence; moving a file does not change that evidence.

The caller receives `CheckpointCompatibilityError.report` with:

- matched, missing, unexpected, and shape-mismatch key lists;
- model/checkpoint tensor-count coverage;
- model-state and model-parameter numel coverage;
- checkpoint SHA-256 and raw/wrapped state-dict format;
- model class, model-state signature, config identity, and config SHA-256;
- applied legacy normalization and final load status.

## Explicit partial loading

Partial loading is only for transfer-learning or diagnostic work. It must be
requested explicitly:

```bash
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --config /path/to/model_config.json \
  --checkpoint /path/to/transfer_checkpoint.pt \
  --allow-partial-checkpoint \
  --wrap \
  --output reports/transfer_predictions.json

python3 tools/export_trt.py \
  --config /path/to/model_config.json \
  --checkpoint /path/to/transfer_checkpoint.pt \
  --allow-partial-checkpoint \
  --onnx models/transfer.onnx \
  --onnx-meta reports/transfer.onnx.meta.json

python3 tools/rtdetr_pose_backend_suite.py \
  --config /path/to/model_config.json \
  --checkpoint /path/to/transfer_checkpoint.pt \
  --allow-partial-checkpoint \
  --backends torch \
  --output reports/transfer_backend_suite.json
```

The adapter API exposes the same choice as the keyword-only
`allow_partial_checkpoint=True` argument. A partial load records
`status=partial`, `allow_partial=true`, and the exact loaded key count. It must
not be cited as evidence that the complete checkpoint was evaluated.

## Stale-output policy

Before a non-dry run loads a checkpoint:

- `export_predictions.py` removes its existing predictions target and requested
  TTA/TTT log targets;
- `export_trt.py` removes ONNX/engine outputs and their metadata when those
  stages are requested;
- `rtdetr_pose_backend_suite.py` removes its existing report target.

An incompatible checkpoint therefore cannot leave an older prediction,
export, or parity artifact at the requested path looking like a successful
result.

`export_trt.py --skip-onnx` consumes an existing ONNX artifact and therefore
rejects `--checkpoint`; checkpoint provenance must come from the metadata of
the ONNX export that produced that artifact.

The backend suite rejects `--checkpoint` unless `torch` is included in
`--backends`; it never records an unchecked checkpoint path in place of a
compatibility report.

## Historical bundled checkpoint

`reports/rtdetr_pose_ckpt_coco128_gpu_matcher.pt` and its prediction/evaluation
artifacts are pinned to source commit
`72f0862f2487c7a23267820cc2dfc4818e46118b`. The recipe named
`reports/rtdetr_pose_minimal_config.json`, but that config was not committed.
The set is historical and is not current full-checkpoint evidence.

Against `rtdetr_pose/configs/base.json` at commit
`735e71d3123e278180836cd89767690c5f426248`, the audit found 20 matching model
state tensors out of 308, 182 missing names, 157 unexpected names, and 106
shape mismatches. Model-parameter numel coverage was about 3.61%. The pinned
hashes and complete summary are in
[`../reports/rtdetr_pose_coco128_gpu_matcher_historical.json`](../reports/rtdetr_pose_coco128_gpu_matcher_historical.json).

Use a checkpoint produced with a checked-in current config for inference,
export, or parity evidence. Keep the historical files only for provenance.
