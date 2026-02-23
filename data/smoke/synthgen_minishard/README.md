# SynthGen smoke mini-shard

This fixture is a deterministic, offline SynthGen intake sample used by tests and smoke flows.

Contents:

- `shards/train_000.jsonl` (2 records)
  - one `animal_v1`
  - one `mechanical_v1`
- `*_image.png`, `*_depth.npy`, `*_inst.npy`, `*_sem.npy`, `*_kpts.npy`
- `predictions_synthgen_smoke.json` (path-based prediction artifact)

Use it for:

- contract validation (`tools/validate_synthgen_contract.py`)
- overlay rendering (`tools/render_synthgen_overlay.py`)
- evaluation smoke (`tools/eval_synthgen.py`)
