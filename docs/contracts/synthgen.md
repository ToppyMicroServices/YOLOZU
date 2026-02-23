# SynthGen data contract (YOLOZU intake)

`YOLOZU` keeps synthetic generation logic out of scope.  
This contract defines the stable intake boundary for external generators (for example `YOLOZU-synthgen`).

## Scope

- Producer: external synth generator repo.
- Consumer: YOLOZU dataset adapters, visualization, and evaluation tools.
- Versioning key: `schema_id` + `schema_version`.

## Required fields (v1)

Each sample record must provide:

- `image`: `uint8[H,W,3]` (RGB image)
- `depth_ndc`: `float32[H,W]` in `[0,1]`
- `inst_id`: `uint32[H,W]` (instance id map)
- `sem_id`: `uint16[H,W]` (semantic class map)
- `kpts2d`: `float32[N_inst,K,3]` (`u,v,vis`, where `vis ∈ {0,1,2}`)
- `prompt`: `str`
- `scene_spec`: JSON object (`str` or decoded object)
- `schema_id`: `str` (e.g. `animal_v1`, `mechanical_v1`)
- `schema_version`: `str`
- `asset_ids`: `list[str]`
- `inst_map`: JSON object (`str` or decoded object), mapping instance index ↔ `inst_id`

## Optional fields (v1+)

- `kpts3d_object`: `float32[N_inst,K,3]`
- `pose_obj2cam`: `float32[4,4]` or `float32[N_inst,4,4]`

## Shape and range rules

- `image`, `depth_ndc`, `inst_id`, and `sem_id` must share the same `H,W`.
- `depth_ndc` values must be normalized to `[0,1]`.
- `kpts2d[...,2]` (visibility) must be one of `0,1,2`.

## Runtime adapter policy

- Validation and coercion implementation: `yolozu/contracts/synthgen.py`
- Shard adapter: `yolozu/data/synthgen_shard_dataset.py`
- Stream adapter: `yolozu/data/synthgen_stream_dataset.py`

## Contract checks

Validation CLI:

```bash
python3 tools/validate_synthgen_contract.py --input /path/to/sample_or_jsonl --max-samples 100
```

Pytest gate:

```bash
python3 -m pytest -q tests/test_contract_synthgen.py
```
