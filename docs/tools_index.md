# Tools index (AI-friendly)

This repo treats `tools/` as a stable, scriptable interface layer on top of the lightweight `yolozu/` core.

Source of truth:
- Tool manifest: `tools/manifest.json`
- Manifest validator: `python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative`

## Workflow maturity at a glance

- Stable: prediction validation/evaluation and the predictions interface contract
- Experimental: backend parity, benchmark orchestration, SynthGen handoff, macOS/MPS qualification paths
- Research: continual learning, TTT, Hessian refinement
- Read first: `docs/production_readiness.md`

## Canonical CLI (recommended entrypoint)

For most day-to-day flows, start with:

- `python3 -m yolozu doctor ...`
- `python3 -m yolozu guide --goal first-run` (beginner-safe route with visible PNG output)
- `python3 -m yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress` (writes images, masks, PNG overlays, and a JSON report; checklist: `configs/quickstart/instance_seg_demo.yaml`)
- `python3 -m yolozu demo overview --output reports/demo_overview_report.json` (full demo map: bbox/segmentation/keypoints/depth/pose6d coverage + dependency checks + visible quickstart command)
- `python3 -m yolozu completion --shell bash` (or `--shell zsh`) to print shell completion script for `yolozu`.
- `python3 -m yolozu export --backend {dummy,torch,onnxrt,trt,executorch} ...`
  - Torch backend can use `--infer-batch-size`, `--torch-compile*`, `--torch-amp`, `--torch-channels-last`, `--torch-inference-mode` for lightweight inference acceleration.
  - TTA extensions: `--tta-mode {postprocess,model}`, `--tta-keypoint-swap-pairs`, `--tta-model-merge-iou`.
  - Non-torch score-only adaptation: `--ttt --ttt-lite-non-torch` (+ `--ttt-lite-*` knobs).
- `python3 -m yolozu predict-images --input-dir /path/to/images ... --progress` (writes predictions JSON, PNG overlays, and optional HTML; checklist: `configs/quickstart/predict_images_dummy.yaml`)
- `python3 tools/eval_keypoints.py --dataset /path/to/yolo --predictions /path/to/predictions.json ...`
- `python3 -m yolozu eval-instance-seg --dataset /path/to/yolo --predictions /path/to/instance_seg_predictions.json ...`
- `python3 tools/hpo_sweep.py --config docs/hpo_sweep_example.json ...`
- `python3 -m yolozu train-orchestrate --spec reports/train_orchestration_spec.json --output reports/training_orchestration_report.json`
- `python3 -m yolozu train-orchestrate --spec reports/train_orchestration_spec.json --output reports/training_orchestration_report.json --registry-out reports/training_registry.jsonl --execute`

Compatibility note:
- `python3 tools/yolozu.py ...` remains available in a repo checkout as a legacy wrapper. It forwards canonical package commands such as `eval-coco`, `benchmark`, `validate`, `train`, and `train-orchestrate`, but `yolozu` / `python3 -m yolozu` is the single supported top-level CLI surface.

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

## TTT compare boilerplates

- Recommended short entrypoint: `bash scripts/ttt_compare.sh --boilerplate tent --dataset data/smoke --split val --checkpoint /path/to.ckpt --run-dir reports/ttt_compare/tent`
- Python equivalent: `python3 tools/run_ttt_compare.py --boilerplate tent --dataset data/smoke --split val --checkpoint /path/to.ckpt --run-dir reports/ttt_compare/tent`
- Available boilerplates: `tent`, `mim`, `cotta`, `eata`, `sar`
- Details: `docs/ttt_compare_boilerplates.md`, `docs/ttt_protocol.md`

## Evaluation helpers

- Reference adapter regression gate (fixed baseline; interface contract hard + behavior warn):
  - full run: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --profile micro --repro-policy relaxed --runtime-lock requirements-locks/requirements-ci.lock --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --diff-summary-out reports/reference_adapter_regression.diff_summary.json --topk-examples-dir reports/reference_adapter_regression_topk --topk-examples 3 --output reports/reference_adapter_regression.json`
  - interface-contract-only hard gate: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-locks/requirements-ci.lock --enforce-runtime-lock --enforce-weights-hash --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --output reports/reference_adapter_regression_contract.json`
  - behavior-only warn gate: `python3 tools/run_reference_adapter_regression.py --dataset data/smoke --split val --max-images 2 --schema-gate-mode off --consistency-gate-mode off --score-gate-mode warn --perf-gate-mode warn --runtime-lock requirements-locks/requirements-ci.lock --enforce-runtime-lock --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json --output reports/reference_adapter_regression_behavior.json`
  - fixed real scenario smoke: `python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 1 --profile micro --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-locks/requirements-ci.lock --baseline reports/reference_adapter_real_multitask_micro_baseline.json --write-baseline --output reports/reference_adapter_regression_real_scenario_baseline_write.json && python3 tools/run_reference_adapter_regression.py --dataset data/real_multitask_fewshot --split val --max-images 1 --profile micro --score-gate-mode off --perf-gate-mode off --runtime-lock requirements-locks/requirements-ci.lock --baseline reports/reference_adapter_real_multitask_micro_baseline.json --output reports/reference_adapter_regression_real_scenario.json`
  - policy docs: `docs/reference_adapter_regression_policy.md`
- External backend support audit (YOLOX/YOLOv8/Detectron2/MMDetection; optional non-dry checks): `python3 tools/audit_backend_support.py --dataset-root data/real_multitask_fewshot --split val --max-images 2 --output reports/backend_support_audit.json --require-non-dry --non-dry-backend yolox`
  - report includes `multitask_coverage` (training/inference/prediction/eval coverage for `bbox/segmentation/keypoints/depth/pose6d`) and enumerated `gaps`.
- External bridge dry-run DoD gate: `python3 -m unittest tests.test_support_external_training_tool`
  - covers YOLOX dry-run artifact plan, runtime/license boundary, next commands, expected outputs, and handoff files without executing external training.
- External finetune smoke matrix (YOLOv/MMDetection/Detectron2/RT-DETR): `python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --output reports/external_finetune_smoke.json`
  - RT-DETR non-dry torch-missing path is explicit (`failure_code=E_DEP_TORCH_MISSING`).
  - MMDetection/Detectron2 with external train launchers can continue train-path audit even if projection deps are missing (`projection_error` + `train_path_audited=true`).
  - machine.dev/GPU example: `python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --non-dry-framework rtdetr --device cuda --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --require-training-execution --output reports/external_finetune_smoke.machine_dev.json`
- Keypoints (PCK + optional OKS mAP): `python3 tools/eval_keypoints.py --dataset /path/to/yolo --predictions reports/predictions.json --output reports/keypoints_eval.json`
  - Add `--oks` to compute COCO OKS mAP (requires `pycocotools`).
- Keypoints parity (backend output diffs): `python3 tools/check_keypoints_parity.py --reference reports/pred_ref.json --candidate reports/pred_cand.json --iou-thresh 0.99 --kp-atol 1e-4`
- Keypoints eval benchmark: `python3 tools/benchmark_keypoints_eval.py --dataset /path/to/yolo --predictions reports/predictions.json --max-images 50 --warmup 1 --iterations 5 --output reports/benchmark_keypoints_eval.json`
- Keypoints backend benchmark/parity: `python3 tools/benchmark_model.py --task keypoints --model reports/keypoints_torch.json --onnx-model reports/keypoints_onnx.json --data /path/to/yolo_keypoints_dataset --format torch,onnx --latency-source artifact_eval --keypoints-parity-kp-atol 1e-4 --output reports/benchmark_keypoints_report.json`
  - artifact-backed real eval/parity lane for backend-specific predictions artifacts; evaluates with `tools/eval_keypoints.py` and compares normalized keypoints directly
- Benchmark support matrix: `docs/benchmark_support_matrix.md`
  - canonical per-format/per-task status for real, artifact-real, dry-run placeholder, and unsupported/skipped benchmark artifacts
- Manual CLI drift audit: `python3 tools/audit_manual_cli_drift.py --json`
  - checks manual chapter 04 against `python3 -m yolozu --help` and the legacy wrapper help surface
- SynthGen intake eval (kpts/seg/depth): `python3 tools/eval_synthgen.py --dataset-root /path/to/synthgen_dataset --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json --schema-id animal_v1 --output reports/synthgen_eval.json`
- Generic depth pair eval: `python3 tools/eval_depth.py --pred-depth /path/to/pred_depth.npy --gt-depth /path/to/gt_depth.npy --align median_scale --output reports/depth_eval.json`
  - writes `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `delta1/2/3`, plus valid-pixel counts
  - `--mask` is optional and `--align median_scale` is useful for relative monocular depth comparisons
- Generic 6DoF eval: `python3 tools/eval_pose.py --dataset /path/to/yolo_pose_dataset --predictions reports/predictions_pose.json --output reports/pose_eval.json`
  - writes `pose_success`, `rot_deg_mean`, `trans_l2_mean`, `depth_abs_mean`, and CAD-point metrics such as `add_mean` / `adds_mean` when sidecar metadata is present
- Depth backend benchmark/parity: `python3 tools/benchmark_model.py --task depth --model reports/depth_torch.npy --onnx-model reports/depth_onnx.npy --data data/reference/gt_depth.npy --format torch,onnx --latency-source artifact_eval --depth-align median_scale --output reports/benchmark_depth_report.json`
  - artifact-backed real eval/parity lane for backend-specific depth outputs; does not pretend YOLOZU executed the backend inference itself
- 6DoF backend benchmark/parity: `python3 tools/benchmark_model.py --task pose6d --model reports/pose_torch.json --onnx-model reports/pose_onnx.json --data /path/to/yolo_pose_dataset --format torch,onnx --latency-source artifact_eval --pose-parity-trans-atol 1e-4 --output reports/benchmark_pose6d_report.json`
  - artifact-backed real eval/parity lane for backend-specific predictions artifacts; evaluates with `tools/eval_pose.py` and compares pose fields directly
- SynthGen overlay renderer: `python3 tools/render_synthgen_overlay.py --dataset-root /path/to/synthgen_dataset --schema-id animal_v1 --sample-index 0 --output reports/synthgen_overlay.png`
- SynthGen smoke (interface contract + overlay + eval): `python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports`

## Continual learning (anti-forgetting)

- Train (runner that wires replay + checkpoint-based self-distillation):
  - `python3 rtdetr_pose/tools/train_continual.py --config configs/continual/rtdetr_pose_domain_inc_example.yaml`
  - Internally passes `--self-distill-from <prev_ckpt>` (plus optional replay / EWC / SI) into `rtdetr_pose/tools/train_minimal.py`.
- Evaluate forgetting / per-task summaries:
  - `python3 tools/eval_continual.py --run-json runs/continual/<run>/continual_run.json --device cpu --max-images 50`
  - On macOS, `--device mps` is supported when `yolozu doctor` reports `mps_available=true`.
  - Docs: `docs/continual_learning.md`
- Decide whether the candidate checkpoint should be promoted, reviewed, or held:
  - `python3 tools/continual_decide.py --eval-json runs/continual/<run>/continual_eval.json --run-json runs/continual/<run>/continual_run.json --max-forgetting 0.05 --min-new-task-score 0.40 --min-old-task-final 0.40 --min-reviewed-labels 20 --min-highconf-pseudo-labels 50 --min-total-curated-examples 60`
  - Device-agnostic: this step reads JSON artifacts only and does not require GPU or MPS.
  - `--curation-json` can inject reviewed-label and high-confidence pseudo-label counts as soft gates.
  - CI/batch pattern: `eval_continual.py -> continual_decide.py -> inspect continual_promotion_decision.json`

## Real-image multitask finetune demo

- `python3 tools/run_real_multitask_finetune_demo.py --dataset-root data/real_multitask_fewshot --out reports/real_multitask_finetune_demo --device cpu --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --strict-provenance --force`
- `python3 tools/run_real_multitask_finetune_demo.py --dataset-root data/real_multitask_fewshot --prepare --download-if-missing --allow-auto-download --accept-dataset-license --download-num-images 8 --out reports/real_multitask_finetune_demo --device cpu --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 --strict-provenance --force`
- Stages: `bbox -> segmentation -> keypoints -> depth -> pose6d`

## Distillation helpers

- Prediction distillation (offline artifact blending; not continual-learning):
  - `python3 tools/distill_predictions.py --student reports/predictions_student.json --teacher reports/predictions_teacher.json --config configs/examples/distill_predictions.yaml --output reports/predictions_distilled.json --output-report reports/distill_report.json`
  - Outputs: `reports/predictions_distilled.json`, `reports/distill_report.json`
  - Read first: `distill_report.json`
  - Docs: `docs/distillation.md`

## Machine-readable tool registry

- Tool manifest: `tools/manifest.json`
- Manifest schema: `docs/schemas/tools_manifest.schema.json`
- Validator: `python3 tools/validate_tool_manifest.py`
- Declarative requirements: `docs/manifest_declarative_spec.md`
- Authoring workflow: `docs/manifest_authoring_workflow.md`
- Every manifest entry carries `maturity = stable|experimental|research`; use `docs/production_readiness.md` as the prose source of truth for those labels.

## Policy helpers

- License policy check: `python3 tools/check_license_policy.py`
- Dependency license report (best-effort): `python3 tools/report_dependency_licenses.py --output reports/dependency_licenses.json`

The manifest is intended for:
- AI agents that need to discover available CLI entrypoints + their I/O interface contracts
- humans who want a quick map of “what command do I run to do X?”

## Release helpers

- single-command release automation: `bash release.sh`
- dry-run preview: `bash release.sh --dry-run --allow-dirty --allow-non-main --output reports/release_report.dry_run.json`
- manual publish recovery: `.github/workflows/publish.yml` `workflow_dispatch` with `expected_version=X.Y.Z` and optional `release_tag=vX.Y.Z`
- Release checklist: `docs/release_reliability_checklist.md`
- Release workflow details: `RELEASE.md`
- Manual DOI workflow details: `docs/manual_doi_release.md`

## External training helpers

- Training platform docs:
  - `docs/training_backend_interface.md`
  - `docs/training_capability_matrix.md`
  - `docs/training_orchestration.md`
- 3-layer support matrix: `python3 tools/support_external_training.py ls -j`
- Top-level train route (primary lane): `python3 -m yolozu train --external-backend yolox configs/examples/finetune_external/yolox_s_finetune_smoke.py --dataset data/smoke --split val --dry-run --output reports/train_external_yolox.json`
- Detectron2 external lane (`bbox` / instance `segmentation` / `keypoints` selected by config): `python3 -m yolozu train --external-backend detectron2 configs/examples/finetune_external/detectron2_finetune_smoke.yaml --dataset data/smoke --split val --task-family bbox --dry-run --output reports/train_external_detectron2_bbox.json`
- MMDetection external lane (`bbox` / instance `segmentation`): `python3 -m yolozu train --external-backend mmdetection configs/examples/finetune_external/mmdetection_finetune_smoke.py --dataset data/smoke --split val --task-family bbox --dry-run --output reports/train_external_mmdetection_bbox.json`
- MMPose external lane (`keypoints`): `python3 -m yolozu train --external-backend mmpose configs/examples/finetune_external/mmpose_finetune_smoke.py --dataset data/smoke --split val --dry-run --output reports/train_external_mmpose.json`
- MMSeg external lane (semantic `segmentation`): `python3 -m yolozu train --external-backend mmseg configs/examples/finetune_external/mmseg_finetune_smoke.py --dataset data/smoke --split val --dry-run --output reports/train_external_mmseg.json`
- Optional top-level Ultralytics bridge: `python3 -m yolozu train --external-backend ultralytics yolo11n.pt --dataset data/smoke --split val --dry-run --output reports/train_external_ultralytics.json`
- Optional top-level HF DETR bridge: `python3 -m yolozu train --external-backend hf-detr facebook/detr-resnet-50 --dataset data/smoke --split val --dry-run --output reports/train_external_hf_detr.json`
- Lightweight orchestration plan/execute: `python3 tools/orchestrate_train.py --spec reports/train_orchestration_spec.json --output reports/training_orchestration_report.json`
- Shared experiment registry append: `python3 tools/orchestrate_train.py --spec reports/train_orchestration_spec.json --output reports/training_orchestration_report.json --registry-out reports/training_registry.jsonl --execute`
- Apache-2.0-friendly YOLOX bridge (dry-run): `python3 tools/support_external_training.py train-yolox --dataset data/smoke --split val --exp configs/examples/finetune_external/yolox_s_finetune_smoke.py --dry-run --output reports/support_external_training.train_yolox.json`
- Detectron2 bridge (dry-run): `python3 tools/support_external_training.py train-detectron2 --config configs/examples/finetune_external/detectron2_finetune_smoke.yaml --dataset data/smoke --split val --task-family bbox --dry-run --output reports/support_external_training.train_detectron2.json`
- MMDetection bridge (dry-run): `python3 tools/support_external_training.py train-mmdetection --config configs/examples/finetune_external/mmdetection_finetune_smoke.py --dataset data/smoke --split val --task-family bbox --dry-run --output reports/support_external_training.train_mmdetection.json`
- MMPose bridge (dry-run): `python3 tools/support_external_training.py train-mmpose --config configs/examples/finetune_external/mmpose_finetune_smoke.py --dataset data/smoke --split val --dry-run --output reports/support_external_training.train_mmpose.json`
- MMSeg bridge (dry-run): `python3 tools/support_external_training.py train-mmseg --config configs/examples/finetune_external/mmseg_finetune_smoke.py --dataset data/smoke --split val --dry-run --output reports/support_external_training.train_mmseg.json`
- Optional Ultralytics bridge (dry-run): `python3 tools/support_external_training.py train-ultralytics --dataset data/smoke --split val --preset smoke --dry-run --output reports/support_external_training.train_ultralytics.json`
- HF DETR entry wrapper (dry-run): `python3 tools/support_external_training.py train-hf-detr -P smoke -n -o reports/support_external_training.train_hf_detr.json`
- ONNX export wrapper (dry-run): `python3 tools/support_external_training.py export-onnx -P smoke -o models/yolo11n.onnx -n -r reports/support_external_training.export_onnx.json`
- Legacy alias: `python3 tools/support_yolo_detr.py ...`
- Read first: `docs/interop_yolox.md`, `docs/training_inference_export.md`, `docs/license_policy.md`
- External lanes now write a standardized wrapper-owned bundle under `work_dir/`, including `reports/export_handoff.json`, `reports/eval_handoff.json`, `reports/parity_handoff.json`, and `reports/training_registry_entry.json`.

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
- route execution through `python3 -m yolozu registry run ...` for allowlist checks

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
