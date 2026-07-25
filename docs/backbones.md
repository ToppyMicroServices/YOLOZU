# Backbones (`rtdetr_pose` adapter)

YOLOZU core interface contracts are model-family agnostic. This page documents the
adapter-scoped backbone interface contract for the in-repo `rtdetr_pose` training path.

## Output interface contract (required)

All backbones used by `rtdetr_pose` must return exactly:

- `features = [P3, P4, P5]`
- `P3`: `(B, C3, H/8,  W/8)`
- `P4`: `(B, C4, H/16, W/16)`
- `P5`: `(B, C5, H/32, W/32)`

Strides other than `[8,16,32]` are not supported directly.

## Projection interface contract

`BackboneProjector` applies independent `1x1 conv` per level:

- `C3 -> d_model`
- `C4 -> d_model`
- `C5 -> d_model`

This keeps transformer input shape stable even when swapping backbones.

## Neck/encoder parity (current behavior)

After projection, the `HybridEncoder` applies:

- FPN + PAN multi-scale fusion (`FPNPAN`)
- optional level embeddings
- optional transformer encoder layers

This is the active path used by `build_model()` and is no longer a per-scale
identity-only neck.

## Supported backbone names

- `cspresnet` (existing default)
- `tiny_cnn` (CPU smoke)
- `cspdarknet_s` (YOLO-like CSP backbone)
- `resnet50` (torchvision)
- `convnext_tiny` (torchvision)

## Config examples

Legacy compatible fields still work (`backbone_name`, `backbone_channels`, etc.), but new config is:

```yaml
model:
  backbone:
    name: cspdarknet_s
    norm: bn   # bn | syncbn | frozenbn | gn
    args:
      width_mult: 0.5
      depth_mult: 0.5
  projector:
    d_model: 256
```

Equivalent swap examples:

```yaml
model:
  backbone:
    name: resnet50
    norm: bn
    args: {}
  projector:
    d_model: 256
```

```yaml
model:
  backbone:
    name: convnext_tiny
    norm: bn
    args: {}
  projector:
    d_model: 256
```

## Adding a new backbone

1. Implement class inheriting `BaseBackbone`.
2. Return only `[P3,P4,P5]` with strides `[8,16,32]`.
3. Define `out_channels = [C3,C4,C5]`.
4. Register with `@register_backbone("your_name")` in `rtdetr_pose/rtdetr_pose/models/backbones/`.
5. Add/update tests:
- shape/stride interface contract
   - projector channel alignment
   - NaN-free forward
   - ONNX smoke export
- neck/encoder interface contract (`tests/test_rtdetr_backbone_neck_parity.py`)

## Norm notes (DDP)

- `bn`: standard training default.
- `syncbn`: useful for multi-GPU small per-rank batch.
- `frozenbn`: stable when BN stats are noisy or frozen-finetune mode is desired.
- `gn`: batch-size independent fallback.
