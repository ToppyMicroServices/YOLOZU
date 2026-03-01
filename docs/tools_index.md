# Tools index (AI-friendly)

This repo treats `tools/` as a stable, scriptable interface layer on top of the lightweight `yolozu/` core.

Source of truth:
- Tool manifest: `tools/manifest.json`
- Manifest validator: `python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative`

## Unified CLI (recommended entrypoint)

For most day-to-day flows, start with:

- `python3 tools/yolozu.py doctor ...`
- `python3 tools/yolozu.py demo overview --output reports/demo_overview_report.json` (full demo map: bbox/segmentation/keypoints/depth/pose6d coverage + dependency checks + recommended commands)
- `python3 tools/yolozu.py completion --shell bash` (or `--shell zsh`) to print shell completion script for `yolozu`.
- `python3 tools/yolozu.py export --backend {dummy,torch,onnxrt,trt,executorch} ...`
  - Torch backend can use `--infer-batch-size`, `--torch-compile*`, `--torch-amp`, `--torch-channels-last`, `--torch-inference-mode` for lightweight inference acceleration.
  - TTA extensions: `--tta-mode {postprocess,model}`, `--tta-keypoint-swap-pairs`, `--tta-model-merge-iou`.
  - Non-torch score-only adaptation: `--ttt --ttt-lite-non-torch` (+ `--ttt-lite-*` knobs).
- `python3 tools/yolozu.py predict-images --input-dir /path/to/images ...`
- `python3 tools/yolozu.py eval-keypoints --dataset /path/to/yolo --predictions /path/to/predictions.json ...`
- `python3 tools/yolozu.py eval-instance-seg --dataset /path/to/yolo --predictions /path/to/instance_seg_predictions.json ...`
- `python3 tools/yolozu.py sweep --config docs/hpo_sweep_example.json ...`

## AI/MCP entrypoints

- MCP server (stdio): `python3 tools/run_mcp_server.py`
- MCP surface inspection: `python3 tools/run_mcp_server.py --print-tools`
- MCP settings check (manifest + generated reference parity): `python3 tools/check_mcp_settings.py --output reports/mcp_settings_check.json`
- Actions/OpenAPI server: `python3 tools/run_actions_api.py --host 127.0.0.1 --port 8080 --workers 1`

### MCP Lite (official) quickstart (copy-paste)

YOLOZU's official MCP surface is intentionally small and deterministic:
`doctor`, `generate_config`, `review_config`.

```bash
# 1) Start MCP server (stdio)
python3 tools/run_mcp_server.py

# 2) Inspect exposed tools as JSON (sanity check)
python3 tools/run_mcp_server.py --print-tools > reports/mcp_tools.json

# 3) Generate a deterministic sample config payload
python3 tools/run_mcp_server.py --sample-generate-config > reports/ai_generate_config.json

# 4) Review that config payload (deterministic output)
python3 tools/run_mcp_server.py --sample-review-config reports/ai_generate_config.json > reports/ai_review_config.json
```

## Dataset helpers

- `python3 tools/make_subset_dataset.py --dataset /path/to/yolo --n 500 --seed 0 --out reports/subset_dataset`
- `python3 scripts/prepare_ttt_domain_shift_target.py --dataset-root data/smoke --split val --out reports/domain_shift/smoke_gaussian_blur_s2 --corruption gaussian_blur --severity 2 --seed 2026 --force`
- Tiny COCO instances subset for demos (downloads 2 images + polygons JSON): `python3 scripts/download_coco_instances_tiny.py`
- `python3 tools/prepare_real_multitask_fewshot.py --out data/real_multitask_fewshot --train-images 6 --val-images 2 --strict-provenance --force`
- `python3 tools/validate_synthgen_contract.py --input /path/to/shard.jsonl --max-samples 200`

## Evaluation helpers

- Reference adapter regression gate (fixed baseline; interface contract hard + behavior warn):
  - full run: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --profile micro --repro-policy relaxed --runtime-lock requirements-ci.lock --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --diff-summary-out reports/reference_adapter_regression.diff_summary.json --topk-examples-dir reports/reference_adapter_regression_topk --topk-examples 3 --output reports/reference_adapter_regression.json`
  - interface-contract-only hard gate: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-ci.lock --enforce-runtime-lock --enforce-weights-hash --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --output reports/reference_adapter_regression_contract.json`
  - behavior-only warn gate: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --schema-gate-mode off --consistency-gate-mode off --score-gate-mode warn --perf-gate-mode warn --runtime-lock requirements-ci.lock --enforce-runtime-lock --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --output reports/reference_adapter_regression_behavior.json`
  - fixed real scenario smoke: `python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 1 --profile micro --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-ci.lock --baseline reports/reference_adapter_real_multitask_micro_baseline.json --write-baseline --output reports/reference_adapter_regression_real_scenario_baseline_write.json && python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 1 --profile micro --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-ci.lock --baseline reports/reference_adapter_real_multitask_micro_baseline.json --output reports/reference_adapter_regression_real_scenario.json`
  - policy docs: `docs/reference_adapter_regression_policy.md`
- External backend support audit (YOLOX/YOLOv8/Detectron2/MMDetection; optional non-dry checks): `python3 tools/audit_backend_support.py --dataset-root data/real_multitask_fewshot --split val --max-images 2 --output reports/backend_support_audit.json --require-non-dry --non-dry-backend yolox`
  - report includes `multitask_coverage` (training/inference/prediction/eval coverage for `bbox/segmentation/keypoints/depth/pose6d`) and enumerated `gaps`.
- External finetune smoke matrix (YOLOv/MMDetection/Detectron2/RT-DETR): `python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --output reports/external_finetune_smoke.json`
  - RT-DETR non-dry torch-missing path is explicit (`failure_code=E_DEP_TORCH_MISSING`).
  - MMDetection/Detectron2 with external train launchers can continue train-path audit even if projection deps are missing (`projection_error` + `train_path_audited=true`).
  - machine.dev/GPU example: `python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --non-dry-framework rtdetr --device cuda --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --require-training-execution --output reports/external_finetune_smoke.machine_dev.json`
- Keypoints (PCK + optional OKS mAP): `python3 tools/eval_keypoints.py --dataset /path/to/yolo --predictions reports/predictions.json --output reports/keypoints_eval.json`
  - Add `--oks` to compute COCO OKS mAP (requires `pycocotools`).
- Keypoints parity (backend output diffs): `python3 tools/check_keypoints_parity.py --reference reports/pred_ref.json --candidate reports/pred_cand.json --iou-thresh 0.99 --kp-atol 1e-4`
- Keypoints eval benchmark: `python3 tools/benchmark_keypoints_eval.py --dataset /path/to/yolo --predictions reports/predictions.json --max-images 50 --warmup 1 --iterations 5 --output reports/benchmark_keypoints_eval.json`
- SynthGen intake eval (kpts/seg/depth): `python3 tools/eval_synthgen.py --dataset-root /path/to/synthgen_dataset --predictions reports/synthgen_predictions.json --schema-id animal_v1 --output reports/synthgen_eval.json`
- SynthGen overlay renderer: `python3 tools/render_synthgen_overlay.py --dataset-root /path/to/synthgen_dataset --schema-id animal_v1 --sample-index 0 --output reports/synthgen_overlay.png`
- SynthGen smoke (interface contract + overlay + eval): `python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports`

## Continual learning (anti-forgetting)

- Train (runner that wires replay + checkpoint-based self-distillation):
  - `python3 rtdetr_pose/tools/train_continual.py --config configs/continual/rtdetr_pose_domain_inc_example.yaml`
  - Internally passes `--self-distill-from <prev_ckpt>` (plus optional replay / EWC / SI) into `rtdetr_pose/tools/train_minimal.py`.
- Evaluate forgetting / per-task summaries:
  - `python3 tools/eval_continual.py --run-json runs/continual/<run>/continual_run.json --device cpu --max-images 50`
  - Docs: `docs/continual_learning.md`

## Real-image multitask finetune demo

- `python3 tools/run_real_multitask_finetune_demo.py --dataset-root data/real_multitask_fewshot --out reports/real_multitask_finetune_demo --device cpu --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --strict-provenance --force`
- `python3 tools/run_real_multitask_finetune_demo.py --dataset-root data/real_multitask_fewshot --prepare --download-if-missing --allow-auto-download --accept-dataset-license --download-num-images 8 --out reports/real_multitask_finetune_demo --device cpu --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --strict-provenance --force`
- Stages: `bbox -> segmentation -> keypoints -> depth -> pose6d`

## Distillation helpers

- Prediction distillation (offline artifact blending; not continual-learning):
  - `python3 tools/distill_predictions.py --student reports/predictions_student.json --teacher reports/predictions_teacher.json --output reports/predictions_distilled.json`
  - Docs: `docs/distillation.md`

## Machine-readable tool registry

- Tool manifest: `tools/manifest.json`
- Manifest schema: `docs/schemas/tools_manifest.schema.json`
- Validator: `python3 tools/validate_tool_manifest.py`
- Declarative requirements: `docs/manifest_declarative_spec.md`
- Authoring workflow: `docs/manifest_authoring_workflow.md`

## Policy helpers

- License policy check: `python3 tools/check_license_policy.py`
- Dependency license report (best-effort): `python3 tools/report_dependency_licenses.py --output reports/dependency_licenses.json`

The manifest is intended for:
- AI agents that need to discover available CLI entrypoints + their I/O interface contracts
- humans who want a quick map of “what command do I run to do X?”

## Release helpers

- single-command release automation: `bash release.sh`
- dry-run preview: `bash release.sh --dry-run --allow-dirty --allow-non-main --output reports/release_report.dry_run.json`
- Release checklist: `docs/release_reliability_checklist.md`
- Manual DOI workflow details: `docs/manual_doi_release.md`

## Ultralytics/DETR helpers

- 3-layer support matrix: `python3 tools/support_ultralytics_detr.py ls -j`
- Ultralytics fine-tune wrapper (dry-run): `python3 tools/support_ultralytics_detr.py tu -P smoke -n -o reports/support_ultralytics_detr.train_ultralytics.json`
- HF DETR entry wrapper (dry-run): `python3 tools/support_ultralytics_detr.py th -P smoke -n -o reports/support_ultralytics_detr.train_hf_detr.json`
- ONNX export wrapper (dry-run): `python3 tools/support_ultralytics_detr.py eo -P smoke -o models/yolo11n.onnx -n -r reports/support_ultralytics_detr.export_onnx.json`
- Details: `docs/ultralytics_detr_support.md`

### AI-required manifest fields

For AI-safe automation, treat these fields as required per tool:

- `id`
- `summary`
- `inputs` (argument schema)
- `examples`
- `effects` (write side-effects)
- `requires` (network/GPU dependency hints)

### Determinism / safety defaults

When driving tools from an agent:

- prefer `--dry-run` when supported
- cap work with `--max-images` (e.g. 50)
- keep outputs in `reports/`
- assume `no-network` unless the tool explicitly requires network
- route execution through `python3 tools/yolozu.py registry run ...` for allowlist checks

## Interface Contracts (recommended)

Most flows in this repo pass data as JSON artifacts:
- `predictions_json`: per-image detections JSON (validate with `tools/validate_predictions.py`)
- `metrics_report_json`: stable report payloads (`yolozu.metrics_report.build_report`)
- `synthgen_sample_contract`: external synthetic sample intake interface contract (validate with `tools/validate_synthgen_contract.py`)
- JSON Schemas live under `docs/schemas/` (report/tool interface contracts) and `schemas/` (runtime data interface contracts such as SynthGen), and are referenced from `tools/manifest.json` contracts.

When adding a new tool, prefer:
1) reading inputs from file paths / flags
2) writing outputs to a deterministic path (default under `reports/`)
3) printing the output path to stdout

## CLI path behavior (consistency)

For user-facing tools (`eval_*`, parity checkers, baseline/tuning scripts), path handling is:

- CLI relative input paths: resolved from the current working directory first, then repo-root fallback.
- Config-file relative input paths (where supported): resolved from the config file directory first.
- Relative output paths: written under the current working directory.

This keeps local CUI runs predictable while preserving backwards compatibility with repo-root workflows.

## Adding a new tool (checklist)

- Add the script under `tools/` (thin CLI; keep logic in `yolozu/` when possible)
- Add an entry to `tools/manifest.json` with:
  - `id`, `entrypoint`, `runner`, `summary`
  - at least one runnable `examples[].command`
  - `contracts.{consumes,produces}` when applicable
- Run: `python3 tools/validate_tool_manifest.py`
