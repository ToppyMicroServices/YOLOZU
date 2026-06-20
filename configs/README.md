# Config layout

This directory keeps operational config assets out of the repository root.

- `configs/runtime/`
  - Runtime control files used by utilities and experiments.
  - Current files:
    - `constraints.yaml`
    - `symmetry.json`
- `configs/quickstart/`
  - Beginner/operator checklists that pin the command, expected output folders,
    and first PNG artifact for the most common smoke paths.
  - Current files:
    - `predict_images_dummy.yaml`
    - `instance_seg_demo.yaml`
- `configs/examples/`
  - Source-checkout CLI examples.
  - Current files:
    - `train_setting.yaml`
    - `train_rtdetr_stable.yaml`
    - `test_setting.yaml`
    - `finetune_external/yolox_s_finetune_smoke.py`
    - `finetune_external/ultralytics_yolov8n_finetune_smoke.yaml`
    - `finetune_external/mmdetection_finetune_smoke.py`
    - `finetune_external/detectron2_finetune_smoke.yaml`
    - `finetune_external/rtdetr_pose_finetune_smoke.yaml`
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
