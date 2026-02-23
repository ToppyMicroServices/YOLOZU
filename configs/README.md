# Config layout

This directory keeps operational config assets out of the repository root.

- `configs/runtime/`
  - Runtime control files used by utilities and experiments.
  - Current files:
    - `constraints.yaml`
    - `symmetry.json`
- `configs/examples/`
  - Source-checkout CLI examples.
  - Current files:
    - `train_setting.yaml`
    - `test_setting.yaml`
    - `synthgen/synthgen_animal_kpt.yaml`
    - `synthgen/synthgen_mechanical_kpt.yaml`
- `configs/tasks/`
  - Task presets that pin contract-level semantics (schema/task variants).
  - Current files:
    - `synthgen_animal_kpt.yaml`
    - `synthgen_mechanical_kpt.yaml`

Most tools accept an explicit `--config` path. For legacy compatibility,
`yolozu.config.default_runtime_config_path(...)` also checks the old root
location when needed.
