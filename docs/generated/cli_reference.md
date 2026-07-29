# Generated CLI Reference

This file is generated from `python3 -m yolozu --help`, the benchmark parser, and `tools/manifest.json`.
Keep narrative docs short and link here for the full command surface.

## Top-level `yolozu --help`

```text
usage: yolozu [-h] [--version] {guide,doctor,dr,list,fetch,export,export-dataset,predict-images,eval-coco,calibrate,eval-long-tail,long-tail-recipe,benchmark,parity,predictions,validate,eval-instance-seg,onnxrt,resources,migrate,import,train,train-orchestrate,test,demo,registry,completion,comp} ...

positional arguments:
  {guide,doctor,dr,list,fetch,export,export-dataset,predict-images,eval-coco,calibrate,eval-long-tail,long-tail-recipe,benchmark,parity,predictions,validate,eval-instance-seg,onnxrt,resources,migrate,import,train,train-orchestrate,test,demo,registry,completion,comp}
    guide               Show beginner-friendly routes and copy-paste commands.
    doctor (dr)         Check the environment. Use --explain for beginner-friendly next actions.
    list                List registries and built-in catalogs.
    fetch               Download a model artifact from the built-in (or custom) model registry.
    export              Export predictions.json artifacts across the supported backend lanes.
    export-dataset      Export a YOLOZU dataset into YOLO, COCO, KITTI, or segmentation layout.
    predict-images      Run folder inference and write predictions JSON + overlays + HTML.
    eval-coco           Evaluate detections with COCOeval (optional extra: yolozu[coco]).
    calibrate           Apply post-hoc FRACAL calibration to bbox or instance-seg predictions JSON.
    eval-long-tail      Evaluate long-tail detection metrics in one standardized report.
    long-tail-recipe    Generate a decoupled long-tail training recipe with plugin-style rebalance config.
    benchmark           Ultralytics-parity benchmark entrypoint with real, artifact-backed, and explicit skipped results.
    parity              Compare two predictions JSON artifacts for backend parity.
    predictions         Predictions artifact utilities.
    validate            Validate artifacts (predictions JSON, instance-seg predictions).
    eval-instance-seg   Evaluate instance segmentation predictions (mask mAP over PNG masks).
    onnxrt              ONNXRuntime utilities (optional extra: yolozu[onnxrt]).
    resources           Access packaged configs/schemas/protocols.
    migrate             Migration helpers (dataset/config/predictions).
    import              Import adapters (read-only projection into canonical schema).
    train               Train with the RT-DETR pose reference trainer by default, or use --external-backend yolox|detectron2|mmdetection|mmpose|mmseg|tao|ultralytics|hf-detr for external training lanes.
    train-orchestrate   Plan or execute a small multi-backend training batch from one orchestration spec.
    test                Run scenario suite (dummy/precomputed adapters are CPU-only).
    demo                Run small self-contained demos (CPU-friendly).
    registry            AI-first tool registry: list/show/validate/run tools from the canonical manifest.
    completion (comp)   Print shell completion script (bash/zsh).

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

© 2026 ToppyMicroServices OÜ
Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia
Registry code: 16551297
Contact: develop@toppymicros.com
```

## `yolozu benchmark` option reference

```text
-h, --help
  show this help message and exit
-m, --model MODEL [required]
  Primary model/weights path recorded in the benchmark report.
--torch-model TORCH_MODEL
  Optional torch backend model override (typically .pt).
--onnx-model ONNX_MODEL
  Optional ONNX backend model override (typically .onnx).
--engine-model ENGINE_MODEL
  Optional TensorRT engine override (typically .engine or .plan).
--torchscript-model TORCHSCRIPT_MODEL
  Optional TorchScript backend model override (typically .torchscript, .ts, or .pt).
--openvino-model OPENVINO_MODEL
  OpenVINO-lane artifact override. Detect expects a compatible IR; artifact-backed tasks accept prepared task artifacts without checking or invoking the OpenVINO runtime.
-d, --data DATA [required]
  Dataset root or data.yaml path recorded in the benchmark report.
--depth-mask DEPTH_MASK
  Optional valid-pixel mask used for task=depth artifact evaluation.
--depth-align {none,median_scale}
  Depth artifact alignment mode for task=depth benchmark eval/parity (default: median_scale).
--depth-parity-mae-atol DEPTH_PARITY_MAE_ATOL
  Depth parity MAE threshold (default: 0.02).
--depth-parity-rmse-atol DEPTH_PARITY_RMSE_ATOL
  Depth parity RMSE threshold (default: 0.03).
--segmentation-parity-mismatch-atol SEGMENTATION_PARITY_MISMATCH_ATOL
  Segmentation parity mismatch-rate tolerance (default: 0.0, exact mask match).
--parity-reference-backend {auto,torch,onnx,engine,torchscript,openvino}
  Reference backend used when writing parity artifacts (default: auto prefers torch, then first eligible backend). OpenVINO detect requires a supplied IR and runtime; artifact-backed OpenVINO tasks use prepared artifacts without a runtime check.
--classification-parity-score-atol CLASSIFICATION_PARITY_SCORE_ATOL
  Classification parity class-score tolerance (default: 1e-4).
--obb-parity-iou-thresh OBB_PARITY_IOU_THRESH
  OBB parity rotated-IoU match threshold (default: 0.99).
--obb-parity-score-atol OBB_PARITY_SCORE_ATOL
  OBB parity confidence-score tolerance (default: 1e-4).
--keypoints-parity-iou-thresh KEYPOINTS_PARITY_IOU_THRESH
  Keypoints parity IoU threshold (default: 0.99).
--keypoints-parity-score-atol KEYPOINTS_PARITY_SCORE_ATOL
  Keypoints parity score tolerance (default: 1e-4).
--keypoints-parity-bbox-atol KEYPOINTS_PARITY_BBOX_ATOL
  Keypoints parity bbox tolerance (default: 1e-4).
--keypoints-parity-kp-atol KEYPOINTS_PARITY_KP_ATOL
  Keypoints parity keypoint tolerance in normalized coords (default: 1e-4).
--pose-parity-rot-deg-atol POSE_PARITY_ROT_DEG_ATOL
  6DoF parity rotation threshold in degrees (default: 1e-3).
--pose-parity-trans-atol POSE_PARITY_TRANS_ATOL
  6DoF parity translation L2 threshold in meters (default: 1e-4).
--pose-parity-depth-atol POSE_PARITY_DEPTH_ATOL
  6DoF parity depth threshold in meters (default: 1e-4).
-i, --imgsz IMGSZ
  Input image size (default: 640).
--half, --no-half
  Use FP16 in the torch detect backend. Must remain disabled when the effective latency source is artifact_eval.
--int8, --no-int8
  Record INT8 intent.
--device DEVICE
  Target device string (default: cpu).
--verbose
  Print per-format status lines.
-f, --format FORMAT
  Comma-separated Phase-1 formats or all.
--task {6dof,classification,classify,cls,depth,detect,detection,keypoints,obb,pose,pose-6d,pose6d,pose_6d,seg,segmentation}
  Benchmark task label. Canonical tasks: detect, segmentation, classification, obb, keypoints, depth, pose6d. Aliases: detection, seg, classify, cls, pose, 6dof.
--split SPLIT
  Dataset split label.
--protocol {yolo26,nms_applied,e2e_nms_free}
  Optional eval protocol passed through to eval_suite and torch exporter.
--max-images MAX_IMAGES
  Optional max image count recorded in the report.
--dry-run
  Validate wiring and dry-run artifacts without backend runs.
--strict
  Return exit code 2 if any requested format is skipped, fails, or is partial.
--repro-policy {strict,relaxed,off}
--runtime-lock RUNTIME_LOCK
  Runtime lock label recorded in run_meta.
--run-id RUN_ID
  Optional run id (default: UTC timestamp).
-o, --output OUTPUT
  Benchmark report JSON path.
--history HISTORY
  Optional JSONL history file path.
--predictions-output PREDICTIONS_OUTPUT
  Optional file/dir/template for predictions artifacts.
--eval-output EVAL_OUTPUT
  Optional file/dir/template for eval artifacts.
--parity-output PARITY_OUTPUT
  Optional file/dir/template for parity artifacts.
--batch BATCH
  Torch detect backend batch size (default: 1). Must remain 1 when the effective latency source is artifact_eval.
--dynamic, --no-dynamic
  Record dynamic-shape intent.
--nms, --no-nms
  Request NMS in the torch detect backend. Must remain disabled when the effective latency source is artifact_eval.
--simplify, --no-simplify
  Record ONNX simplify intent.
--opset OPSET
  Record ONNX opset (default: 17).
--workspace WORKSPACE
  Record TensorRT workspace in GiB (default: 4).
--fraction FRACTION
  Record dataset fraction knob (default: 1.0).
--latency-source {auto,synthetic_step,dataset_pass_wall_time,artifact_eval}
  Benchmark source selection. auto uses dataset_pass_wall_time for detect and artifact_eval for classification, obb, segmentation, keypoints, depth, and pose6d. Detect rejects explicit artifact_eval before writes because no prepared detection-artifact evaluation path is implemented. Supported artifact_eval tasks consume prepared artifacts, so --half, --batch values other than 1, and --nms are rejected. Non-dry-run artifact-backed tasks cannot use dataset_pass_wall_time; use auto or artifact_eval.
--iterations ITERATIONS
  Synthetic latency iterations (default: 50).
--warmup WARMUP
  Synthetic latency warmup iterations (default: 5).
--sleep-s SLEEP_S
  Finite, non-negative synthetic latency sleep per step (default: 0).
```

## Manifest Tool Registry

| Tool ID | Maturity | Entry point | Summary |
|---|---|---|---|
| adapter_parity_suite | experimental | tools/adapter_parity_suite.py | Run parity checks for multiple adapter outputs against a reference adapter predictions file. |
| announce_release | stable | tools/announce_release.py | Generate (and optionally post) release announcement bundle for LinkedIn/X/Reddit from GitHub release event payload. |
| audit_backend_support | experimental | tools/audit_backend_support.py | Audit YOLOX/YOLOv8/Detectron2/MMDetection exporters, with verified execution evidence for selected non-dry backends. |
| audit_docs_examples_drift | stable | tools/audit_docs_examples_drift.py | Audit README/docs examples against yolozu help, manual CLI drift, and manifest help drift gates. |
| audit_manual_cli_drift | stable | tools/audit_manual_cli_drift.py | Audit manual chapter 04 against canonical yolozu CLI help and the legacy wrapper passthrough help surface. |
| backend_parity_matrix | experimental | tools/backend_parity_matrix.py | Run one-command backend parity matrix checks across torch/onnxrt/trt/opencv_dnn/custom_cpp and export JSON+HTML reports. |
| benchmark_eata_stability | research | tools/benchmark_eata_stability.py | Compare EATA stability/efficiency diagnostics with a baseline; efficacy remains not established and the report cannot promote defaults. |
| benchmark_keypoints_eval | experimental | tools/benchmark_keypoints_eval.py | Benchmark keypoints evaluation runtime (PCK + optional OKS mAP) and write a stable JSON report. |
| benchmark_latency | experimental | tools/benchmark_latency.py | Latency/FPS benchmark harness producing stable JSON reports and optional JSONL history. |
| benchmark_model | experimental | tools/benchmark_model.py | Benchmark entrypoint with real torch/onnx/engine/torchscript and conditional OpenVINO detect orchestration when available; fail-closed rejection of detect artifact_eval; fail-closed, strict-JSON artifact-backed classification and OBB eval/parity with task-specific metrics and provenance; artifact-backed real eval/parity lanes for segmentation/keypoints/depth/pose6d on torch/onnx/engine/torchscript/openvino; task/source/format-aware flag and artifact interface contract validation; explicit task semantics, runtime/license boundary docs, stable artifacts, explicit skipped-format reporting, and a canonical support matrix. |
| benchmark_sar_robustness | research | tools/benchmark_sar_robustness.py | Compare local SAR/CoTTA/EATA diagnostics; efficacy remains not established and the report is not go/no-go evidence. |
| build_manifest | stable | tools/build_manifest.py | Build a dataset manifest for data/coco128 (writes reports/manifest.json). |
| build_trt_engine | experimental | tools/build_trt_engine.py | Build a TensorRT engine from ONNX using trtexec and write a reproducible meta JSON. |
| calibrate_scores | stable | tools/calibrate_scores.py | Temperature-scale detection scores to improve mAP proxy on a fixed subset (no NMS). |
| check_golden_compatibility | experimental | tools/check_golden_compatibility.py | Validate versioned golden compatibility assets and hash-pinned eval protocol snapshots. |
| check_keypoints_parity | experimental | tools/check_keypoints_parity.py | Compare two keypoints prediction JSONs and report mismatches (IoU/tolerance-based). |
| check_license_policy | stable | tools/check_license_policy.py | Enforce Apache-2.0-only constraints (denylist + no vendored GPL/AGPL license texts). |
| check_map_targets | stable | tools/check_map_targets.py | Compare eval_suite results against a target table and exit non-zero on failure. |
| check_mcp_settings | stable | tools/check_mcp_settings.py | Audit MCP settings by checking manifest alignment and generated MCP/Actions reference freshness. |
| check_predictions_parity | experimental | tools/check_predictions_parity.py | Compare two prediction JSONs and report mismatches (IoU/tolerance-based). |
| check_repo_governance | stable | tools/check_repo_governance.py | Audit repository governance posture from local workflow evidence and exported GitHub settings snapshots. |
| check_segmentation_parity | experimental | tools/check_segmentation_parity.py | Compare two segmentation predictions artifacts and report mask-level parity mismatches. |
| continual_decide | research | tools/continual_decide.py | Device-agnostic policy gate for continual-learning eval results that emits a promote/review/hold decision report with a research_report boundary. |
| convert_coco_instance_seg_predictions | stable | tools/convert_coco_instance_seg_predictions.py | Convert COCO instance segmentation predictions (polygons/RLE) into the YOLOZU instance-seg PNG-mask interface contract. |
| distill_predictions | research | tools/distill_predictions.py | Offline prediction distillation helper: blend teacher predictions into a student predictions JSON, emit a distilled artifact plus report, and document the workflow with beginner-facing mental models and YAML boilerplates, clearly separate from training-time self-distillation or TTT. |
| dod_cpu_smoke | stable | scripts/dod_cpu_smoke.sh | Run and time the CPU-only public DoD path: doctor proof, demo, validation, and eval dry-run. |
| download_bop_dataset | research | tools/download_bop_dataset.py | Safely download fixed-host BOP archives and record hashes, byte sizes, URLs, and license provenance. |
| download_coco_instances_tiny | stable | scripts/download_coco_instances_tiny.py | Download a tiny COCO instances (polygon) subset (2 images by default) for `yolozu demo instance-seg` without bundling images in git. |
| eval_coco | stable | tools/eval_coco.py | Strictly validate and evaluate predictions on a YOLO-format dataset using COCOeval, with concise path aliases, explicit repair, and bounded-subset accounting; --dry-run does not require pycocotools. |
| eval_continual | research | tools/eval_continual.py | Evaluate a continual learning run with real COCOeval, simple mAP proxy, or pose metrics; define FWT only from an explicit initial-checkpoint baseline; and write hash-bound JSON+HTML. |
| eval_cotta_drift | research | tools/eval_cotta_drift.py | Compare baseline-vs-CoTTA TTT reports and generate reproducible drift/stability evidence artifacts (JSON+Markdown). |
| eval_depth | experimental | tools/eval_depth.py | Evaluate one predicted depth map against one ground-truth map and report depth_error metrics. |
| eval_instance_segmentation | stable | tools/eval_instance_segmentation.py | Evaluate instance segmentation predictions (mask mAP) from binary PNG masks with optional HTML/overlays. |
| eval_keypoints | stable | tools/eval_keypoints.py | Evaluate keypoint predictions using PCK (bbox-normalized distance) with optional COCO OKS mAP, HTML, and overlays. |
| eval_pose | experimental | tools/eval_pose.py | Evaluate one pose-aware predictions artifact against dataset sidecars and report pose6d_error metrics, success rates, and optional CAD-point metrics. |
| eval_segmentation | stable | tools/eval_segmentation.py | Evaluate semantic segmentation predictions (mIoU / per-class IoU) with ignore_index support and optional HTML/overlays. |
| eval_suite | stable | tools/eval_suite.py | Evaluate prediction JSONs and preserve declared exporter settings in a single suite report. |
| eval_synthgen | experimental | tools/eval_synthgen.py | Evaluate SynthGen predictions (keypoints + segmentation + depth) against shard ground truth. |
| export_actions_openapi | stable | tools/export_actions_openapi.py | Export a static OpenAPI JSON schema for YOLOZU Actions API registration. |
| export_predictions | stable | tools/export_predictions.py | Repository wrapper for the packaged fail-closed predictions exporter with checkpoint compatibility evidence; installed yolozu export does not depend on repository-only tools, optional acceleration flags require backend/device qualification, TTA remains Experimental, TTT remains Research, and parent maturity does not promote either lane. |
| export_predictions_coco_keypoints | experimental | tools/export_predictions_coco_keypoints.py | Convert COCO-style keypoints results JSON into the YOLOZU predictions interface contract for downstream eval and parity. |
| export_predictions_detectron2 | experimental | tools/export_predictions_detectron2.py | Run fail-closed Detectron2 inference and export explicitly versioned predictions.json with protocol, execution-evidence, and provenance metadata. |
| export_predictions_executorch | experimental | tools/export_predictions_executorch.py | Decode declared ExecuTorch runtime output JSON into YOLOZU predictions JSON (dry-run supported for interface contract validation). |
| export_predictions_mmdet | experimental | tools/export_predictions_mmdet.py | Run fail-closed MMDetection inference and export explicitly versioned predictions.json with protocol, execution-evidence, and provenance metadata. |
| export_predictions_onnxrt | experimental | tools/export_predictions_onnxrt.py | Run ONNXRuntime inference and export YOLOZU predictions JSON (requires onnxruntime + numpy + opencv; see Rust ONNXRuntime template notes in external inference docs). |
| export_predictions_opencv_dnn | experimental | tools/export_predictions_opencv_dnn.py | Run OpenCV-DNN inference on an ONNX model and export YOLOZU predictions JSON (YOLOv8 84 or YOLOv5 85+obj raw heads supported). |
| export_predictions_opencv_dnn_rtdetr | experimental | tools/export_predictions_opencv_dnn_rtdetr.py | Run OpenCV-DNN inference on an RT-DETR ONNX model and export YOLOZU predictions JSON (no NMS), recording fixed preprocess/export_settings. |
| export_predictions_opencv_dnn_unified | experimental | tools/export_predictions_opencv_dnn_unified.py | Unified OpenCV-DNN exporter with preprocess/decode presets, IO dump, and backend/target controls. |
| export_predictions_openvino | experimental | tools/export_predictions_openvino.py | Run OpenVINO detection inference and export YOLOZU predictions JSON via the declared combined-output decode path. |
| export_predictions_torchscript | experimental | tools/export_predictions_torchscript.py | Run TorchScript detection inference and export YOLOZU predictions JSON via the declared combined-output decode path. |
| export_predictions_trt | experimental | tools/export_predictions_trt.py | Run TensorRT engine inference and export YOLOZU predictions JSON (requires tensorrt + CUDA bindings). |
| export_predictions_yolo_runtime | stable | tools/export_predictions_yolo_runtime.py | Run bounded external YOLO-runtime inference and export explicitly versioned predictions.json with cardinality-checked execution evidence. |
| export_predictions_yolov5 | experimental | tools/export_predictions_yolov5.py | Convert YOLOv5 outputs (save-txt or xyxy JSON) into YOLOZU predictions.json with recorded export_settings. |
| export_predictions_yolox | experimental | tools/export_predictions_yolox.py | Run fail-closed YOLOX inference (or dry-run) and export explicitly versioned predictions.json with execution evidence and exp/checkpoint provenance. |
| export_trt | experimental | tools/export_trt.py | Fail-closed PyTorch → ONNX → TensorRT export route with shared RT-DETR checkpoint compatibility evidence. |
| fetch_coco128 | stable | tools/fetch_coco128.sh | Fetch tiny COCO subset (YOLO-format) into data/coco128 (official COCO hosting). |
| fetch_model | stable | yolozu/cli.py | Download a model artifact with cache reuse, sha256 pinning, and explicit license/integrity gates. |
| fresh_install_journey | stable | scripts/fresh_install_journey.sh | Install YOLOZU from public PyPI in a clean environment and record the complete stable-lane journey. |
| gen_ci_dummy_dets_onnx | stable | tools/ci/gen_dummy_dets_onnx.py | Generate a tiny deterministic ONNX model for CI TensorRT/ONNXRuntime smoke parity checks. |
| gen_ci_smoke_dataset | stable | tools/ci/gen_smoke_dataset.py | Generate a minimal YOLO-format dataset for CI exporter and parity smoke tests. |
| generate_benchmark_support_matrix | stable | tools/generate_benchmark_support_matrix.py | Generate the canonical benchmark support matrix from support metadata. |
| generate_integration_tool_reference | stable | tools/generate_integration_tool_reference.py | Generate the MCP↔Actions interface contract reference from tool_runner + server wrappers and fail on drift in check mode. |
| generate_runtime_parity_case_study | experimental | tools/generate_runtime_parity_case_study.py | Generate and verify a real Mask R-CNN PyTorch eager versus TorchScript comparison through YOLOZU's stable evaluation lane. |
| generate_smoke_assets | stable | tools/generate_smoke_assets.py | Generate deterministic offline smoke assets under data/smoke from local data/coco128. |
| generate_web_docs | stable | tools/generate_web_docs.py | Safely generate searchable onboarding web docs from repository-local SSOT sources, validated URLs, schemas, and curated evidence. |
| gpu_validation_preflight | experimental | tools/gpu_validation_preflight.py | Generate a preflight report that splits the YOLOZU-zisn GPU sweep into local-executable checks and GPU-runtime checks. |
| hpo_sweep | stable | tools/hpo_sweep.py | Run a configurable parameter sweep (grid or list) and emit JSONL/CSV/MD results. |
| import_yolo_data_yaml | stable | tools/import_yolo_data_yaml.py | Import YOLO-style data.yaml into a YOLOZU dataset wrapper and classes mapping (classes.json/classes.txt). |
| list_models | stable | yolozu/cli.py | List fetchable model IDs from the built-in (or custom) model registry. |
| make_subset_dataset | stable | tools/make_subset_dataset.py | Create a deterministic, provenance-hashed YOLO subset while preserving referenced training sidecars and refusing unowned output replacement. |
| measure_trt_latency | experimental | tools/measure_trt_latency.py | Measure a TensorRT engine's latency/FPS and write a metrics report JSON. |
| normalize_predictions | stable | tools/normalize_predictions.py | Normalize prediction class ids (category_id↔class_id) and optionally wrap with meta. |
| orchestrate_train | experimental | tools/orchestrate_train.py | Lightweight experiment orchestration entrypoint for training: expand a multi-backend spec into planned commands or execute them, support top-level defaults such as dataset/split/resume_from, emit one orchestration report JSON, and optionally append executed runs to a shared JSONL registry with backend counts and registry_summary. |
| package_segmentation_predictions | experimental | tools/package_segmentation_predictions.py | Package a class-id mask directory into the YOLOZU segmentation predictions interface contract. |
| pre_push | stable | scripts/pre_push.sh | Run local pre-push gates (ruff, focused unit tests including manifest/generated-reference/SSOT coverage and external bridge dry-run DoD, offline smoke, and real-image preflight) to catch CI failures before pushing. |
| prepare_ade20k_seg | stable | tools/prepare_ade20k_seg.py | Prepare ADE20K semantic segmentation layout + dataset.json manifest. |
| prepare_bop_yolozu | research | tools/prepare_bop_yolozu.py | Convert BOP rigid-object pose data into an owned YOLOZU dataset with deterministic splits, metre-scaled CAD points, and provenance. |
| prepare_cityscapes_seg | stable | tools/prepare_cityscapes_seg.py | Prepare Cityscapes semantic segmentation layout + dataset.json manifest. |
| prepare_coco_instance_seg | stable | tools/prepare_coco_instance_seg.py | Convert official COCO instances JSON into YOLO-format labels + per-instance PNG masks + sidecar metadata for instance-seg eval. |
| prepare_coco_yolo | stable | tools/prepare_coco_yolo.py | Convert official COCO instances JSON into YOLO-format labels + (optional) copy images. |
| prepare_keypoints_dataset | stable | tools/prepare_keypoints_dataset.py | Prepare keypoints dataset in one command (auto-detect YOLO Pose or COCO keypoints) and emit YOLOZU-ready dataset wrapper. |
| prepare_real_multitask_fewshot | stable | tools/prepare_real_multitask_fewshot.py | Create a small real-image multitask few-shot dataset (bbox/seg/keypoints/depth/pose sidecars) from COCO instances, with optional tiny COCO auto-download and explicit label provenance metadata. |
| prepare_ttt_domain_shift_target | research | scripts/prepare_ttt_domain_shift_target.py | Prepare a deterministic, provenance-hashed domain-shift target and recipe while refusing source/output overlap, symlink outputs, protected roots, and unowned replacement. |
| prepare_voc_seg | stable | tools/prepare_voc_seg.py | Prepare Pascal VOC semantic segmentation layout + dataset.json manifest. |
| publish_benchmark_table | stable | tools/publish_benchmark_table.py | Generate official benchmark publication table (JSON+Markdown) from benchmark reports with run-id traceability. |
| qualify_artifact_research | research | tools/qualify_artifact_research.py | Qualify offline prediction distillation and Hessian refinement with three deterministic repetitions, stable COCO metrics, hashes, measured cost, rollback, and explicit hold gates. |
| qualify_finetune_lanes | experimental | tools/qualify_finetune_lanes.py | Qualify real-image and external fine-tuning execution in one command while failing closed on projection-only non-dry lanes and retaining Experimental maturity when labels or task-native metrics are insufficient. |
| qualify_sdft_continual | research | tools/qualify_sdft_continual.py | Qualify the repository's checkpoint-distillation continual lane against naive sequential fine-tuning across three fixed seeds, real COCOeval, identical data/order/budget, provenance hashes, and explicit hold gates. |
| refine_predictions_hessian | research | tools/refine_predictions_hessian.py | Refine pose-related prediction fields with an engine-external Newton/finite-diff Hessian stepper and optional research_report log boundary; the public CLI rollout is offsets-first and opt-in. |
| release | stable | release.sh | Single-command release automation: validate current metadata, atomically synchronize package version, dated CHANGELOG and CITATION metadata, and explicitly current-release-coupled source/packaged manifest examples; then auto-version, tag, publish a GitHub release, and hand off once to the PyPI and Zenodo workflows. Historical manifest examples require explicit evidence and are not auto-bumped. The publish workflow revalidates synchronized metadata before upload, then verifies that PyPI exposes the released wheel and sdist. The container workflow can reuse the same release_tag for GHCR publication and NGC mirroring under nvcr.io/yolozu/.... |
| release_tag | stable | tools/release_tag.py | Release/tag operation helper that fails closed unless package version, requested tag, dated CHANGELOG heading, CITATION version/date, and source/packaged manifests agree; after validation it can create/push tags and create GitHub draft/published releases with report output. Downstream publish automation can reuse the same release_tag for GHCR publication and NGC mirroring under nvcr.io/yolozu/.... |
| render_synthgen_overlay | experimental | tools/render_synthgen_overlay.py | Render semantic + instance + keypoint overlays from SynthGen shard samples. |
| render_ttt_manual_figures | research | tools/render_ttt_manual_figures.py | Render the six-file docs/manual TTT figure bundle atomically from validated synthetic-fixture or hash-bound measured sources. |
| report_dependency_licenses | stable | tools/report_dependency_licenses.py | Generate a best-effort dependency license report from installed Python packages (not legal advice). |
| rtdetr_pose_backend_suite | experimental | tools/rtdetr_pose_backend_suite.py | Fail-closed RT-DETR backend parity + benchmark suite with shared checkpoint compatibility evidence. |
| rtdetr_pose_train_continual | research | rtdetr_pose/tools/train_continual.py | Continual fine-tuning runner for rtdetr_pose with an explicit initial-checkpoint/FWT baseline, prior-stage checkpoint self-distillation, replay/LoRA/EWC/SI options, and per-task checkpoint, teacher, data-order, time, memory, and command provenance. |
| run_actions_api | stable | tools/run_actions_api.py | Run the GPT Actions API, including fail-closed typed TTT/CTTA export jobs with full checkpoint preflight. |
| run_external_finetune_smoke | experimental | tools/run_external_finetune_smoke.py | Run external finetune smoke matrix for YOLOX/Ultralytics/MMDetection/Detectron2/RT-DETR; non-dry projection-only lanes fail closed, while the report records dependency status, artifacts, hashes, runtime, source, and license boundaries. |
| run_mcp_server | stable | tools/run_mcp_server.py | Run the YOLOZU MCP stdio server, expose canonical live tool ids and filtered packaged discovery, and provide typed TTT/CTTA export jobs that reject arbitrary extra args and require status=full checkpoint preflight. |
| run_real_multitask_finetune_demo | experimental | tools/run_real_multitask_finetune_demo.py | Run staged real-image bbox/segmentation/keypoints/depth/pose6d training with strict provenance, checkpoint-handoff hashes, runtime/memory, and explicit task-native metric and heuristic-label boundaries. |
| run_reference_adapter_regression | experimental | tools/run_reference_adapter_regression.py | Run RT-DETR reference-adapter regression with hardened record I/O+preprocess interface contract checks, profile-aware micro/full flows, backend-aware behavior gates, optional parity checks, provenance/SBOM capture, and automatic diff-summary/top-k failure artifacts. |
| run_rtdetr_pose_backend_suite | experimental | tools/run_rtdetr_pose_backend_suite.py | End-to-end runner for rtdetr_pose: export (PyTorch→ONNX→TRT) + backend parity/benchmark suite. |
| run_scenarios | stable | tools/run_scenarios.py | Run the scenario suite pipeline (adapter + constraints utilities) and write a report JSON. |
| run_trt_pipeline | experimental | tools/run_trt_pipeline.py | Orchestrate the YOLO26 TensorRT pipeline (engine build → predictions export → parity → eval_suite → latency report). |
| run_ttt_compare | research | tools/run_ttt_compare.py | Run a fail-closed seeded TTT comparison with full checkpoint preflight, real COCO evaluation, calibration/collapse diagnostics, and generated adaptation-cost counters. |
| run_ttt_evidence_suite | research | tools/run_ttt_evidence_suite.py | Run and aggregate a fail-closed five-method clean/shifted TTT matrix for at least three seeds. |
| smoke | stable | scripts/smoke.sh | Run one-command offline smoke flow (doctor -> validate dataset -> validate predictions -> eval-coco dry-run -> synthgen intake smoke) with optional deep walkthrough checks. |
| smoke_synthgen | experimental | tools/smoke_synthgen.py | Run deterministic SynthGen intake smoke, or generate and qualify a fresh cross-repo handoff with strict QA and loader checks. |
| support_external_training | experimental | tools/support_external_training.py | External training bridge with a fixed 3-layer interface contract: Apache-2.0-friendly YOLOX-style training as the primary lane, Detectron2/MMDetection/MMPose/MMSeg/TAO external lanes for common vision tasks, plus optional Ultralytics and HF DETR bridges with explicit runtime/license boundaries. Missing external executables return machine-readable runtime failures instead of uncaught process-launch errors. Executed runs also write standardized resume/export/eval/parity handoff JSONs and an optional append-only registry entry. OpenCV DNN and ONNX Runtime remain export/inference runtimes rather than training backends. |
| support_yolo_detr | experimental | tools/support_yolo_detr.py | YOLO/DETR support utility with fixed 3-layer integration interface contract (trainer/repo/export), short-option + alias UX, preset defaults, shared dataset conversion, ONNX export template, and prediction normalization. |
| ttt_compare | research | scripts/ttt_compare.sh | Short shell entrypoint for the fail-closed TTT local diagnostic; requires a fully compatible checkpoint and never promotes efficacy. |
| tune_gate_weights | research | tools/tune_gate_weights.py | Offline grid-search for inference-time score-fusion weights (CPU-only, simple mAP proxy). |
| validate_instance_segmentation_predictions | stable | tools/validate_instance_segmentation_predictions.py | Validate YOLOZU instance segmentation predictions JSON (per-image instances; PNG masks). |
| validate_map_targets | stable | tools/validate_map_targets.py | Validate the mAP target table file (baselines/yolo26_targets.json). |
| validate_predictions | stable | tools/validate_predictions.py | Validate YOLOZU predictions JSON (permissive by default; strict optional) with compatible human output or an explicit bounded JSON result. |
| validate_run_meta | stable | tools/validate_run_meta.py | Validate the run_meta.json interface contract (git SHA, dependency lock, preprocess, hardware/runtime, command). |
| validate_segmentation_predictions | stable | tools/validate_segmentation_predictions.py | Validate YOLOZU segmentation predictions JSON (id→mask path mapping; meta optional). |
| validate_synthgen_contract | experimental | tools/validate_synthgen_contract.py | Validate SynthGen sample/shard interface-contract fields, dtypes, shapes, and ranges before training/eval. |
| validate_tool_manifest | stable | tools/validate_tool_manifest.py | Validate tools/manifest.json structure, references, and declarative metadata. |
| yolo26_pre_pr_quality | stable | scripts/pre_pr_quality.sh | Run pre-PR quality checklist for YOLO26 flow (smoke + lint + focused tests). |
| yolo26_quality_gate | stable | tools/yolo26_quality_gate.sh | Run YOLO26-focused quality gate (lint + focused tests) before eval/target checks. |
| yolozu | stable | yolozu/cli.py | Top-level YOLOZU dispatcher for validation, migration, dataset, training, and predictions/eval workflows, including a packaged torch/TTT exporter usable outside a source checkout; Stable applies to the parent/core lane and does not promote Experimental or Research subcommands and flags. |

## Smoke Coverage

- `tests/test_docs_examples_drift.py` checks documented shell examples against help/manifest flags.
- `tests/test_manual_cli_drift_audit.py` checks manual chapter 04 command references against top-level help.
- `tests/test_generated_cli_reference.py` fails when this generated reference drifts.
