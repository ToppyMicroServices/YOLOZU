# YOLOZU Docs

Evaluate existing predictions first.

Use this page as the shortest route from a wrapped `predictions.json` to a validated, comparable report.

## 1-Minute Demo

```bash
python3 -m pip install -U yolozu
yolozu doctor --proof
yolozu demo instance-seg --run-dir reports/quickstart_instance_seg --progress
```

Writes a report plus visible PNG overlays under `reports/quickstart_instance_seg/`.
Use `configs/quickstart/instance_seg_demo.yaml` as the checklist for expected files.
For an interactive-feeling route map in the terminal:

```bash
yolozu guide
yolozu guide --goal first-run
yolozu doctor --proof
```

## Read These First

- [`predictions_schema.md`](predictions_schema.md): the predictions interface contract
- [`python_api.md`](python_api.md): typed in-process validation/evaluation API and stable error policy
- [`install.md`](install.md): install, `doctor`, and environment setup
- [`cpu_only_dod.md`](cpu_only_dod.md): CPU-only proof/demo/validate/eval DoD path
- [`external_inference.md`](external_inference.md): evaluate predictions exported elsewhere
- [`byop_quickstarts.md`](byop_quickstarts.md): checked Ultralytics, Detectron2, MMDetection, and YOLOX paths
- [`case_studies/maskrcnn_eager_torchscript.md`](case_studies/maskrcnn_eager_torchscript.md): real eager and TorchScript outputs through one pinned evaluation lane

## Next 3 Routes

- Stable lane: evaluate precomputed predictions and keep the predictions interface contract stable
- Bridge lane: train/export flows that emit the same predictions interface contract
- Benchmark/Research lanes: backend parity, SynthGen handoff, and opt-in research lanes over already evaluated artifacts

## Future Adaptive Local Vision Work

The environment-qualified local image-processing program targets an Experimental lane and does not expand the current Stable surface. An AI client supplies a typed request; selection must use matching task, environment, workload, protocol, artifact, and license evidence or abstain.

- Human-readable projection: [`../reports/adaptive_vision_roadmap.md`](../reports/adaptive_vision_roadmap.md)
- Packaged machine-readable projection: [`../yolozu/data/manifest/adaptive_vision_roadmap.json`](../yolozu/data/manifest/adaptive_vision_roadmap.json)
- Live task source and refresh rule: [`roadmap.md`](roadmap.md)
- Current Experimental implementation boundary: `yolozu qualify-image-pipeline`
  emits only an unactivated report; `yolozu activate-qualification-evidence`
  defaults to a no-write gate report and mutates only with explicit review and
  `--approve`. Selection and adaptive execution are not yet available.

## Primary Focus

- Stable lane: evaluate precomputed predictions fairly across frameworks and runtimes
- Bridge lane: export and external training flows that emit the same predictions interface contract
- Benchmark lane: qualify backend parity after the stable evaluation path is working
- Research lane: opt-in workflows over already evaluated artifacts

## Capability Maturity

- Stable: prediction validation/evaluation, wrapped `predictions.json`, install/doctor flow, repo smoke/demo path
- Experimental: backend parity, benchmark orchestration, external training handoff, macOS/MPS evaluation paths
- Research: continual learning, self-distillation, TTT, Hessian refinement

## Production Readiness

- Production-ready today: prediction validation/evaluation and the predictions interface contract
- Needs qualification in your environment: backend parity, benchmark orchestration, SynthGen handoff, macOS/MPS paths
- Research-oriented: continual learning, self-distillation, TTT, Hessian refinement
- Details: [`production_readiness.md`](production_readiness.md)

## Quick route map

- If you already have predictions: go to [A) Evaluate from precomputed predictions](#a-evaluate-from-precomputed-predictions-no-inference-deps)
- If your model stays in another framework: use [Bring your own predictions quickstarts](byop_quickstarts.md)
- If you need the in-repo reference trainer: go to [B) Train → Export → Eval](#b-train--export--eval-rt-detr-reference-trainer)
- If you need an external training lane: use `yolozu train --external-backend yolox|detectron2|ultralytics|hf-detr ...` and then go to [Training / inference / export](training_inference_export.md#external-yolo-style-training-lane-yolox-primary-optional-bridges-second)
- If you need the current training scope boundary first: read [Current training support](training_inference_export.md#current-training-support)
- If you need the platform view of training: read [Training backend interface](training_backend_interface.md), [Training capability matrix](training_capability_matrix.md), and [Training orchestration](training_orchestration.md)
- If you need the latest real/external fine-tuning evidence: read the [2026-07-29 qualification report](../reports/finetune_lane_evidence_2026-07-29.md)
- If you are qualifying non-default paths: use [D) Bench/Parity](#d-benchparity-parity-check--benchmark-entry), [Research lanes](research_lanes.md), or [SynthGen handoff](synthgen_repo_integration.md)

## Offline repo smoke

The fastest safety check from repo root is:

```bash
bash scripts/smoke.sh
```

Expected report output:

- `reports/smoke_coco_eval_dry_run.json`

Optional deeper walkthrough (capability claims + deploy-path dry-runs + walkthrough report):

```bash
bash scripts/smoke.sh --profile deep
```

On a CUDA machine (single GPU), run the deep profile with the TTT probe on GPU:

```bash
bash scripts/smoke.sh --profile deep --torch-device cuda
```

Deep walkthrough report:
- `reports/smoke_walkthrough_report.json`

Supporting docs:

- Install: [`install.md`](install.md)
- Structured support and feedback: [`support.md`](support.md)
- Security / cryptography scope: [`security_crypto_scope.md`](security_crypto_scope.md)
- Repository governance / Scorecard posture: [`security_scorecard_governance.md`](security_scorecard_governance.md), [`repo_governance_audit.md`](repo_governance_audit.md)
- Production readiness: [`production_readiness.md`](production_readiness.md)
- Runtime version and qualification boundaries: [`versions.md`](versions.md)
- SSOT capability coverage audit: [`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md)
- TTT readiness and evidence boundary: [`../reports/ttt_readiness_audit_2026-07-26.md`](../reports/ttt_readiness_audit_2026-07-26.md)
- Non-TTT artifact research evidence: [`../reports/artifact_research_evidence_2026-07-28.md`](../reports/artifact_research_evidence_2026-07-28.md)
- Evaluation protocol template: [`evaluation_protocol_template.md`](evaluation_protocol_template.md)
- Reproducible runtime comparison: [`case_studies/maskrcnn_eager_torchscript.md`](case_studies/maskrcnn_eager_torchscript.md)
- Adoption measurement, feedback cadence, and dated baseline: [`adoption/README.md`](adoption/README.md)
- Consented design-partner observation kit: [`adoption/design_partner_observation_kit.md`](adoption/design_partner_observation_kit.md)
- Schema governance / browser coverage: [`schema_governance.md`](schema_governance.md)
- Searchable web docs: <https://www.toppymicros.com/yolozu/docs/>
- Web docs generation and publication plan: [`web_docs_plan.md`](web_docs_plan.md)
- Product readiness audit: [`../reports/yolozu_product_readiness_2026-07-26.md`](../reports/yolozu_product_readiness_2026-07-26.md)
- Generated CLI reference: [`generated/cli_reference.md`](generated/cli_reference.md)
- Dataset processing and round-trip matrix: [`dataset_processing_matrix.md`](dataset_processing_matrix.md)
- BOP T-LESS rigid-object 6DoF Research protocol: [`bop_tless_protocol.md`](bop_tless_protocol.md)
- Research lanes: [`research_lanes.md`](research_lanes.md)
- Research note template: [`research_note_template.md`](research_note_template.md)
- Learning features overview: [`learning_features.md`](learning_features.md)

---

## A) Evaluate from precomputed predictions (no inference deps)

Use this path when predictions are exported elsewhere and you only need validation/evaluation here.

Shortest command (strict predictions validation is included):

```bash
python3 -m yolozu eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_coco_eval_dry_run.json
```

Reference docs:
- [External inference backends](external_inference.md)
- [Predictions schema](predictions_schema.md)
- [Stable Python API](python_api.md)

## B) Train → Export → Eval (RT-DETR reference trainer)

Use this path when you want a train-like flow with smoke-safe local artifacts.

Shortest 3 commands:

```bash
python3 -m yolozu validate dataset data/smoke --strict
python3 -m yolozu export \
	--backend labels \
	--dataset data/smoke \
	--output runs/smoke/predictions_labels.json \
	--force
python3 -m yolozu eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions runs/smoke/predictions_labels.json \
	--dry-run \
	--output runs/smoke/coco_eval_dry_run.json
```

Reference docs:
- [Training / inference / export](training_inference_export.md)
- [RT-DETR checkpoint compatibility](checkpoint_compatibility.md)
- [Run contract](run_contract.md)

## C) Interface Contracts (predictions / adapter / TTT protocol)

Use this path to confirm JSON interface contracts and manifest consistency before bigger runs.

Shortest 3 commands:

```bash
python3 -m yolozu validate predictions \
	data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu validate dataset data/smoke --strict
python3 tools/validate_tool_manifest.py \
	--manifest tools/manifest.json \
	--require-declarative
```

Reference docs:
- [Predictions schema](predictions_schema.md)
- [Adapter interface contract](adapter_contract.md)
- [TTT protocol](ttt_protocol.md)
- [TTT before-after compare boilerplates](ttt_compare_boilerplates.md)

## D) Bench/Parity (qualification lane)

Use this path after the main validation/eval lane is already working and you are
qualifying backend parity or benchmark behavior. Some formats already support
artifact-backed real eval/parity lanes; others still report explicit
unsupported/skipped semantics rather than pretending to be fully implemented.
The canonical support-status table is
[Benchmark support matrix](benchmark_support_matrix.md).
It also records which backend flags apply after `--latency-source auto`
resolves. Artifact-backed evaluation consumes prepared files, so it accepts
only the defaults `--no-half --batch 1 --no-nms`.
Detect uses `auto` or `dataset_pass_wall_time` for backend execution and rejects
explicit `artifact_eval` before any output or backend write.

Shortest 3 commands:

```bash
python3 -m yolozu parity \
	--reference data/smoke/predictions/predictions_dummy.json \
	--candidate data/smoke/predictions/predictions_dummy.json \
	--bbox-format auto
python3 -m yolozu benchmark \
	--model runs/example/model.pt \
	--data data/coco8.yaml \
	--format all \
	--dry-run \
	--output reports/benchmark_report.json
python3 tools/benchmark_latency.py --help
```

When backend artifacts are available, the benchmark entry can orchestrate real
`torch` / `onnx` / `engine` / `torchscript` runs and a conditional OpenVINO
run:

```bash
python3 -m yolozu benchmark \
	--model runs/example/model.pt \
	--onnx-model exports/example.onnx \
	--engine-model exports/example.plan \
	--openvino-model exports/example.xml \
	--data data/coco8.yaml \
	--format torch,onnx,engine,openvino \
	--protocol nms_applied \
	--latency-source auto \
	--output reports/benchmark_report.json
```

Typical outputs:
- `reports/benchmark_report.json`
- `reports/export_settings_<format>.json`
- `reports/predictions_<format>.json`
- `reports/eval_<format>.json`
- `reports/parity_<format>.json`

`torchscript` is now a real detect benchmark lane when a local PyTorch runtime
and compatible TorchScript artifact are present. The declared decode path
expects `[x1,y1,x2,y2,score,class_id]` combined output rows.
For the detect command above, OpenVINO is accepted through `--openvino-model`;
its runtime and IR remain external, and missing prerequisites produce an
explicit skipped result. Artifact-backed OpenVINO task lanes consume prepared
files without checking or invoking that runtime.

The benchmark report now records:
- canonical task label
- requested task label
- metric family and expected metric keys
- whether the task is part of the mainstream benchmark surface or a YOLOZU-native extension

Reference docs:
- [TensorRT pipeline](tensorrt_pipeline.md)
- [RT-DETR checkpoint compatibility](checkpoint_compatibility.md)
- [Benchmark mode](benchmark_mode.md)
- [Benchmark support matrix](benchmark_support_matrix.md)
- [Backend runtime / license boundary matrix](benchmark_backend_runtime_matrix.md)
- [License policy and no-telemetry repository boundary](license_policy.md)
- [Benchmark latency](benchmark_latency.md)
- [Benchmark mode spec (parity target)](benchmark_mode_spec_parity_target.md)
- [Benchmark gap audit](benchmark_mode_gap_audit.md)
- [RunPod GPU split preflight](runpod_gpu_validation_split.md)

## E) LLM / MCP integrations

Use this path when integrating YOLOZU tools with MCP clients or Actions/OpenAPI routes.

Shortest 3 commands:

```bash
yolozu-mcp
python3 tools/run_actions_api.py
python3 tools/export_actions_openapi.py --output reports/actions_openapi.json
```

Reference docs:
- [LLM integrations](llm_integrations.md)
- [OpenAI MCP / Actions](openai_mcp_actions.md)
- [Copilot MCP integration](copilot_mcp_integration.md)
- [MCP extension architecture](mcp_extension_architecture.md)
- [AI-first guide (MCP scope)](ai_first.md)

## F) YOLO-style migration (v5/v8/11/26)

Use this path when you keep training/inference in a YOLO-family runtime and only want YOLOZU interface-contract/eval.

Shortest 3 commands:

```bash
python3 tools/import_yolo_data_yaml.py --data-yaml /path/to/data.yaml --split val --output data/yolo_wrapper --force
python3 tools/export_predictions_yolo_runtime.py --model yolo11n.pt --dataset data/yolo_wrapper --split val --protocol nms_applied --wrap --output reports/pred_yolo_runtime.json
python3 -m yolozu eval-coco --dataset data/yolo_wrapper --split val --predictions reports/pred_yolo_runtime.json --output reports/coco_eval_yolo_runtime.json
```

Protocol guidance:
- `nms_applied`: YOLOv5/YOLOv8/YOLO11 style post-NMS exports from a YOLO-family runtime.
- `e2e_nms_free`: YOLO26 / RT-DETR style NMS-free exports.

If predictions contain COCO `category_id`, pass `--classes data/yolo_wrapper/labels/<split>/classes.json` to `eval-coco`.

## G) Detectron2/MMDetection migration

Use this path when you keep Detectron2/MMDetection training/inference and only export results into YOLOZU interface contracts.

Shortest flow:

```bash
yolozu migrate dataset --from coco --coco-root /path/to/coco --split val2017 --output data/coco_yolo_like --mode manifest
yolozu export-dataset yolo --dataset data/coco_yolo_like --split val2017 --out-dir data/coco_yolo_export --force
yolozu export-dataset coco --dataset data/coco_yolo_like --split val2017 --out-dir data/coco_export --force
yolozu export-dataset kitti --dataset data/coco_yolo_like --split val2017 --out-dir data/coco_kitti_export --force
yolozu export-dataset segmentation --dataset reports/cityscapes_seg_wrapper --out-dir data/cityscapes_seg_export --image-mode symlink --force
python3 tools/export_predictions_detectron2.py --dataset data/coco_yolo_like --split val2017 --config /path/to/d2_config.yaml --weights /path/to/model_final.pth --protocol nms_applied --output reports/pred_detectron2.json
python3 tools/export_predictions_mmdet.py --dataset data/coco_yolo_like --split val2017 --config /path/to/mmdet_config.py --checkpoint /path/to/epoch_12.pth --protocol nms_applied --output reports/pred_mmdet.json
```

If you want a preflight that tells you what the dataset already is, use:

```bash
yolozu doctor import --dataset-from auto --dataset /path/to/dataset_root --split val2017 --output -
```

Auto-detect also recognizes semantic-segmentation roots/descriptors and COCO keypoints roots, so the same preflight works for bbox / keypoints / segmentation with one command.

Validation/eval:

```bash
python3 tools/validate_predictions.py reports/pred_detectron2.json --strict
python3 tools/eval_coco.py --dataset data/coco_yolo_like --split val2017 --predictions reports/pred_detectron2.json --protocol nms_applied --classes data/coco_yolo_like/labels/val2017/classes.json --output reports/coco_eval_detectron2.json
```

Reference: [Detectron2/MMDetection interop](interop_detectron2_mmdet.md)

External finetune smoke matrix:
- `python3 tools/run_external_finetune_smoke.py --dataset-root data/smoke --split train --output reports/external_finetune_smoke.json`
- one-command real/external qualification: `./.venv/bin/python tools/qualify_finetune_lanes.py --output-dir /tmp/yolozu-finetune-qualification`
- Guide: [External finetune smoke](external_finetune_smoke.md)

## H) OpenCV-DNN migration (CPU/CUDA/OpenVINO)

Use this path when OpenCV DNN is your runtime of record (C++/embedded/field inference) and you want YOLOZU validation/eval/parity.
Canonical CLI shown below (`yolozu ...` or `python3 -m yolozu ...`). The legacy `python3 tools/yolozu.py ...` wrapper still works in a repo checkout but is now compatibility-only.

Shortest flow:

```bash
python3 -m yolozu export --backend opencv-dnn --onnx /path/to/model.onnx --dataset /path/to/coco-yolo --split val2017 --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv.json
python3 tools/validate_predictions.py reports/pred_opencv.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_opencv.json --output reports/eval_opencv.json
```

Runtime switches:
- `--dnn-backend opencv|cuda|openvino`
- `--dnn-target cpu|cuda|cuda_fp16|opencl|opencl_fp16`

Reference: [OpenCV-DNN inference exporter](opencv_dnn_inference.md)

## I) YOLOX migration

Use this path when YOLOX is your training/inference stack and you want YOLOZU interface-contract validation + eval.
Canonical CLI shown below (`yolozu ...` or `python3 -m yolozu ...`).

```bash
python3 -m yolozu export --backend yolox --dataset /path/to/coco-yolo --split val2017 --exp /path/to/yolox_exp.py --weights /path/to/yolox_ckpt.pth --imgsz 640 --score-thr 0.01 --nms-iou 0.65 --output reports/pred_yolox.json
python3 tools/validate_predictions.py reports/pred_yolox.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_yolox.json --protocol nms_applied --classes /path/to/coco-yolo/labels/val2017/classes.json --output reports/eval_yolox.json
```

Reference: [YOLOX interop](interop_yolox.md)

## J) Model fetch (portable weights intake)

Use this path to download curated model artifacts with license gating, cache reuse, and metadata recording.

```bash
python3 -m yolozu list models
python3 -m yolozu fetch yolox-s-coco --out models --accept-license
cat models/yolox-s-coco/meta.json
```

The metadata interface contract at `models/<id>/meta.json` always includes:
- `source`
- `version`
- `license`
- `sha256`
- `created_at`

Reference: [Model fetch](model_fetch.md)

## K) SynthGen intake (external generator boundary)

Use this path when synthetic shards are produced outside YOLOZU and you only need intake/interface-contract/eval here.

```bash
./.venv/bin/python tools/smoke_synthgen.py --synthgen-repo ../YOLOZU-synthgen --output-dir /tmp/yolozu-synthgen-qualification
python3 tools/validate_synthgen_contract.py --input /path/to/synthgen_dataset/shards/train_000.jsonl --max-samples 200
python3 tools/render_synthgen_overlay.py --dataset-root /path/to/synthgen_dataset --schema-id animal_v1 --sample-index 0 --output reports/synthgen_overlay.png
python3 tools/eval_synthgen.py --dataset-root /path/to/synthgen_dataset --predictions /path/to/synthgen_dataset/shards/predictions_synthgen.json --schema-id animal_v1 --output reports/synthgen_eval.json
python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports
```

References:
- [SynthGen intake guide](synthgen_intake.md)
- [SynthGen interface contract](synthgen_contract.md)
- [YOLOZU-synthgen integration checklist](synthgen_repo_integration.md)

If your predictions artifact reuses shard-relative paths, keep it under `shards/` or rewrite those paths before running `eval_synthgen.py`.

## CI incidents

CI incident memo has moved to a dedicated page:

- [CI incidents memo](ci_incidents.md)
- [Release reliability checklist](release_reliability_checklist.md)
- [1.0 stability boundary](release_1_0_stability.md)
- [Manual PDF DOI release](manual_doi_release.md)
- [YOLO/DETR support (3-layer)](yolo_detr_support.md)
