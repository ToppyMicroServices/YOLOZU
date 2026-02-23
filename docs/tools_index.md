# Tools index (AI-friendly)

This repo treats `tools/` as a stable, scriptable interface layer on top of the lightweight `yolozu/` core.

## Unified CLI (recommended entrypoint)

For most day-to-day flows, start with:

- `python3 tools/yolozu.py doctor ...`
- `python3 tools/yolozu.py export --backend {dummy,torch,onnxrt,trt} ...`
- `python3 tools/yolozu.py predict-images --input-dir /path/to/images ...`
- `python3 tools/yolozu.py eval-keypoints --dataset /path/to/yolo --predictions /path/to/predictions.json ...`
- `python3 tools/yolozu.py eval-instance-seg --dataset /path/to/yolo --predictions /path/to/instance_seg_predictions.json ...`
- `python3 tools/yolozu.py sweep --config docs/hpo_sweep_example.json ...`

## AI/MCP entrypoints

- MCP server (stdio): `python3 tools/run_mcp_server.py`
- MCP surface inspection: `python3 tools/run_mcp_server.py --print-tools`
- Actions/OpenAPI server: `python3 tools/run_actions_api.py --host 127.0.0.1 --port 8080 --workers 1`

## Dataset helpers

- `python3 tools/make_subset_dataset.py --dataset /path/to/yolo --n 500 --seed 0 --out reports/subset_dataset`
- `python3 tools/validate_synthgen_contract.py --input /path/to/shard.jsonl --max-samples 200`

## Evaluation helpers

- Keypoints (PCK + optional OKS mAP): `python3 tools/eval_keypoints.py --dataset /path/to/yolo --predictions reports/predictions.json --output reports/keypoints_eval.json`
  - Add `--oks` to compute COCO OKS mAP (requires `pycocotools`).
- Keypoints parity (backend output diffs): `python3 tools/check_keypoints_parity.py --reference reports/pred_ref.json --candidate reports/pred_cand.json --iou-thresh 0.99 --kp-atol 1e-4`
- Keypoints eval benchmark: `python3 tools/benchmark_keypoints_eval.py --dataset /path/to/yolo --predictions reports/predictions.json --max-images 50 --warmup 1 --iterations 5 --output reports/benchmark_keypoints_eval.json`
- SynthGen intake eval (kpts/seg/depth): `python3 tools/eval_synthgen.py --dataset-root /path/to/synthgen_dataset --predictions reports/synthgen_predictions.json --schema-id animal_v1 --output reports/synthgen_eval.json`
- SynthGen overlay renderer: `python3 tools/render_synthgen_overlay.py --dataset-root /path/to/synthgen_dataset --schema-id animal_v1 --sample-index 0 --output reports/synthgen_overlay.png`

## Continual learning (anti-forgetting)

- Train (runner that wires replay + checkpoint-based self-distillation):
  - `python3 rtdetr_pose/tools/train_continual.py --config configs/continual/rtdetr_pose_domain_inc_example.yaml`
  - Internally passes `--self-distill-from <prev_ckpt>` (plus optional replay / EWC / SI) into `rtdetr_pose/tools/train_minimal.py`.
- Evaluate forgetting / per-task summaries:
  - `python3 tools/eval_continual.py --run-json runs/continual/<run>/continual_run.json --device cpu --max-images 50`
  - Docs: `docs/continual_learning.md`

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
- AI agents that need to discover available CLI entrypoints + their I/O contracts
- humans who want a quick map of “what command do I run to do X?”

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

## Contracts (recommended)

Most flows in this repo pass data as JSON artifacts:
- `predictions_json`: per-image detections JSON (validate with `tools/validate_predictions.py`)
- `metrics_report_json`: stable report payloads (`yolozu.metrics_report.build_report`)
- `synthgen_sample_contract`: external synthetic sample intake contract (validate with `tools/validate_synthgen_contract.py`)
- JSON Schemas live under `docs/schemas/` and are referenced from `tools/manifest.json` contracts.

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
