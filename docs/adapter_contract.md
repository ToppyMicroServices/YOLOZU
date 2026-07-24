# Adapter contract (v1)

Adapters are small wrappers that let an inference backend emit YOLOZU-compatible
prediction artifacts.

They power `tools/export_predictions.py --adapter <name>` (and any other tool that
calls `adapter.predict(records)`).

## Required behavior

An adapter must implement:

- `predict(records: list[dict]) -> list[dict]`

### Input: `records`

`records` are built from YOLO-format datasets via `yolozu.datasets.dataset.build_manifest(...)`.

Each record is a dict with at least:

- `image` (required): image path/key (string)

And may include (optional, best-effort):

- `labels`: list of YOLO labels (`{class_id, cx, cy, w, h, ...}`)
  - keypoints may be appended in YOLO pose-style and normalized by the loader
- `image_hw` / `image_size`: `[h, w]` hints
- pose: `pose` or `R_gt` / `t_gt`
- intrinsics: `intrinsics` or `K_gt` / `K`
- masks: `mask_path` / `mask` (plus `mask_format`, `mask_instances`, `mask_classes`,
  `mask_class_id`, `mask_class_map`)
- depth: `depth_path` / `depth` / `D_obj`

Adapters should ignore unknown keys (so sidecar schemas can evolve without breaking
older backends).

### Output: per-image entries

Each returned entry must be:

- `{ "image": <same key as input record["image"]>, "detections": [...] }`

Each detection must include:

- `class_id`: int
- `score`: float
- `bbox`: `{cx, cy, w, h}` (see `docs/predictions_schema.md` for bbox format rules)

## Optional behavior

- `predict(...)` may include extra keys per detection (mask/depth/pose/intrinsics),
  as long as required keys remain stable.

## TTT (Test-Time Training) hooks
Adapters may optionally support the Research TTT methods (`tent`, `mim`, `cotta`,
`eata`, and `sar`) by implementing:

- `supports_ttt() -> bool`
  - Returns `True` if the adapter supports TTT, `False` otherwise
  - Default: `False`

- `get_model() -> torch.nn.Module | None`
  - Returns the underlying PyTorch model for adaptation
  - Called by TTT integration to access model parameters
  - Default: `None`

- `build_loader(records, *, batch_size: int = 1) -> Iterable[torch.Tensor]`
  - Builds a data loader that yields batches of preprocessed tensors
  - Each batch should be ready for model forward pass (no additional preprocessing)
  - Used by TTT to create adaptation batches before inference
  - Default: raises `RuntimeError("this adapter does not support TTT")`

## Error / exception conventions (recommended)

Adapters should fail fast with clear messages. Recommended patterns:

- Missing optional dependency:
  - raise `RuntimeError("missing optional dependency: torch (python3 -m pip install -U 'yolozu[demo]' or python3 -m pip install -e '.[demo]')")`
- Missing config/checkpoint:
  - raise `FileNotFoundError(<path>)`
- Unsupported option/backend:
  - raise `RuntimeError("unsupported <thing>: ...")`
- Unsupported TTT:
  - raise `RuntimeError("this adapter does not support TTT")` (exact string, for consistency)
- Invalid input record:
  - raise `ValueError("record missing 'image'")` (or similar, with the key name)

### Example TTT adapter implementation

```python
class MyAdapter(ModelAdapter):
    def supports_ttt(self) -> bool:
        return True
    
    def get_model(self):
        self._ensure_backend()
        return self._model
    
    def build_loader(self, records, *, batch_size: int = 1):
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i+batch_size]
            tensors = [self._preprocess(r) for r in batch_records]
            yield torch.stack(tensors)
```

### Usage with TTT

When TTT is enabled (`--ttt`), the integration:
1. Calls `adapter.get_model()` to access the model
2. Calls `adapter.build_loader(records)` to create adaptation batches
3. Runs the selected TTT adaptation method (`tent`, `mim`, `cotta`, `eata`, or `sar`)
4. Calls `adapter.predict(records)` on the adapted model

See [docs/ttt_integration_plan.md](ttt_integration_plan.md) for more details.

## Stability
- `predict` signature and output schema are **stable**.
- New optional fields may be added without breaking old clients.
- The optional TTT hook signatures are backward-compatible as of v1. This
  adapter interface contract stability does not promote the Research TTT
  workflow.

## Versioning
Adapters should be compatible with predictions schema `v1`.

Starter files and onboarding checklist:
- `docs/adapter_templates.md`
- `examples/adapter_starters/`
