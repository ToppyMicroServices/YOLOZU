# YOLOZU

YOLOZU is a framework-agnostic evaluation toolkit for vision models,
designed to address **catastrophic forgetting** and **domain shift**
through reproducible continual learning and test-time adaptation (TTT).

**YOLOZU exists to measure, compare, and control catastrophic forgetting
and inference-time adaptation in a reproducible way.**

YOLOZU treats **predictions—not models—as the stable interface contract**,
making continual learning and test-time training comparable,
restartable, and CI-friendly across frameworks and runtimes.

YOLOZU supports evaluation workflows for object detection, segmentation,
keypoint estimation, **monocular depth estimation**, and **6DoF pose estimation**,
while keeping training implementations optional and decoupled.

YOLOZU supports ONNX export and deployment across PyTorch, ONNX Runtime,
and TensorRT, with C++ and Rust inference templates.

YOLOZU is built for interface-contract-first, AI-first development:
every experiment produces versioned artifacts that can be
compared and regression-tested automatically in CI.

## Docs index

Use this page as the docs index for both humans and agents. All examples below use repository-real paths (`data/smoke`, `reports/...`) to reduce copy-paste mistakes.

## 0) Offline copy-paste smoke (single command)

The fastest safety check from repo root is:

```bash
bash scripts/smoke.sh
```

Expected report output:

- `reports/smoke_coco_eval_dry_run.json`

Install (pip + optional extras): [`docs/install.md`](install.md)

Support/legal: [`docs/support.md`](support.md)

---

## A) Evaluate from precomputed predictions (no inference deps)

Use this path when predictions are exported elsewhere and you only need validation/evaluation here.

Shortest 3 commands:

```bash
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 -m yolozu.cli validate predictions \
	data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_coco_eval_dry_run.json
```

Reference docs:
- [External inference backends](external_inference.md)
- [Predictions schema](predictions_schema.md)

## B) Train → Export → Eval (RT-DETR scaffold)

Use this path when you want a train-like flow with smoke-safe local artifacts.

Shortest 3 commands:

```bash
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 -m yolozu.cli export \
	--backend labels \
	--dataset data/smoke \
	--output runs/smoke/predictions_labels.json \
	--force
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions runs/smoke/predictions_labels.json \
	--dry-run \
	--output runs/smoke/coco_eval_dry_run.json
```

Reference docs:
- [Training / inference / export](training_inference_export.md)
- [Run contract](run_contract.md)

## C) Interface Contracts (predictions / adapter / TTT protocol)

Use this path to confirm JSON interface contracts and manifest consistency before bigger runs.

Shortest 3 commands:

```bash
python3 -m yolozu.cli validate predictions \
	data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu.cli validate dataset data/smoke --strict
python3 tools/validate_tool_manifest.py \
	--manifest tools/manifest.json \
	--require-declarative
```

Reference docs:
- [Predictions schema](predictions_schema.md)
- [Adapter contract](adapter_contract.md)
- [TTT protocol](ttt_protocol.md)

## D) Bench/Parity (parity check + benchmark entry)

Use this path for quick parity sanity checks and to discover benchmark CLI options.

Shortest 3 commands:

```bash
python3 -m yolozu.cli parity \
	--reference data/smoke/predictions/predictions_dummy.json \
	--candidate data/smoke/predictions/predictions_dummy.json
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_parity_eval_dry_run.json
python3 tools/benchmark_latency.py --help
```

Reference docs:
- [TensorRT pipeline](tensorrt_pipeline.md)
- [Benchmark latency](benchmark_latency.md)
- [RunPod GPU split preflight](runpod_gpu_validation_split.md)

## E) LLM / MCP integrations

Use this path when integrating YOLOZU tools with MCP clients or Actions/OpenAPI routes.

Shortest 3 commands:

```bash
python3 tools/run_mcp_server.py
python3 tools/run_actions_api.py
python3 tools/export_actions_openapi.py --output reports/actions_openapi.json
```

Reference docs:
- [LLM integrations](llm_integrations.md)
- [OpenAI MCP / Actions](openai_mcp_actions.md)
- [Copilot MCP integration](copilot_mcp_integration.md)
- [MCP extension architecture](mcp_extension_architecture.md)
- [AI-first guide (supported MCP scope)](ai_first.md)

## F) YOLO migration (v5/v8/11/26)

Use this path when you keep training/inference in YOLO tooling and only want YOLOZU interface-contract/eval.

Shortest 3 commands:

```bash
python3 tools/import_ultralytics_data_yaml.py --data-yaml /path/to/data.yaml --split val --output data/ultra_wrapper --force
python3 tools/export_predictions_ultralytics.py --model yolo11n.pt --dataset data/ultra_wrapper --split val --protocol nms_applied --wrap --output reports/pred_ultra.json
python3 -m yolozu.cli eval-coco --dataset data/ultra_wrapper --split val --predictions reports/pred_ultra.json --protocol nms_applied --output reports/coco_eval_ultra.json
```

Protocol guidance:
- `nms_applied`: YOLOv5/YOLOv8/YOLO11 style post-NMS exports.
- `e2e_nms_free`: YOLO26 / RT-DETR style NMS-free exports.

If predictions contain COCO `category_id`, pass `--classes data/ultra_wrapper/labels/<split>/classes.json` to `eval-coco`.

## G) Detectron2/MMDetection migration

Use this path when you keep Detectron2/MMDetection training/inference and only export results into YOLOZU interface contracts.

Shortest flow:

```bash
yolozu migrate dataset --from coco --coco-root /path/to/coco --split val2017 --output data/coco_yolo_like --mode manifest
python3 tools/export_predictions_detectron2.py --dataset data/coco_yolo_like --split val2017 --config /path/to/d2_config.yaml --weights /path/to/model_final.pth --protocol nms_applied --output reports/pred_detectron2.json
python3 tools/export_predictions_mmdet.py --dataset data/coco_yolo_like --split val2017 --config /path/to/mmdet_config.py --checkpoint /path/to/epoch_12.pth --protocol nms_applied --output reports/pred_mmdet.json
```

Validation/eval:

```bash
python3 tools/validate_predictions.py reports/pred_detectron2.json --strict
python3 tools/eval_coco.py --dataset data/coco_yolo_like --split val2017 --predictions reports/pred_detectron2.json --protocol nms_applied --classes data/coco_yolo_like/labels/val2017/classes.json --output reports/coco_eval_detectron2.json
```

Reference: [Detectron2/MMDetection interop](interop_detectron2_mmdet.md)

## H) OpenCV-DNN migration (CPU/CUDA/OpenVINO)

Use this path when OpenCV DNN is your runtime of record (C++/embedded/field inference) and you want YOLOZU validation/eval/parity.
Repo wrapper shown below (`python3 tools/yolozu.py ...`); pip equivalent is `yolozu export ...`.

Shortest flow:

```bash
python3 tools/yolozu.py export --backend opencv-dnn --onnx /path/to/model.onnx --dataset /path/to/coco-yolo --split val2017 --imgsz 640 --decode auto --preprocess yolo_letterbox_640 --dump-io reports/opencv_dump_io.json --output reports/pred_opencv.json
python3 tools/validate_predictions.py reports/pred_opencv.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_opencv.json --output reports/eval_opencv.json
```

Runtime switches:
- `--dnn-backend opencv|cuda|openvino`
- `--dnn-target cpu|cuda|cuda_fp16|opencl|opencl_fp16`

Reference: [OpenCV-DNN inference exporter](opencv_dnn_inference.md)

## I) YOLOX migration

Use this path when YOLOX is your training/inference stack and you want YOLOZU interface-contract validation + eval.
Repo wrapper shown below (`python3 tools/yolozu.py ...`); pip equivalent is `yolozu export ...`.

```bash
python3 tools/yolozu.py export --backend yolox --dataset /path/to/coco-yolo --split val2017 --exp /path/to/yolox_exp.py --weights /path/to/yolox_ckpt.pth --imgsz 640 --score-thr 0.01 --nms-iou 0.65 --output reports/pred_yolox.json
python3 tools/validate_predictions.py reports/pred_yolox.json --strict
python3 tools/eval_coco.py --dataset /path/to/coco-yolo --split val2017 --predictions reports/pred_yolox.json --protocol nms_applied --classes /path/to/coco-yolo/labels/val2017/classes.json --output reports/eval_yolox.json
```

Reference: [YOLOX interop](interop_yolox.md)

## J) Model fetch (portable weights intake)

Use this path to download curated model artifacts with license gating, cache reuse, and metadata recording.

```bash
python3 tools/yolozu.py list models
python3 tools/yolozu.py fetch yolox-s-coco --out models --accept-license
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
python3 tools/validate_synthgen_contract.py --input /path/to/synthgen_dataset/shards/train_000.jsonl --max-samples 200
python3 tools/render_synthgen_overlay.py --dataset-root /path/to/synthgen_dataset --schema-id animal_v1 --sample-index 0 --output reports/synthgen_overlay.png
python3 tools/eval_synthgen.py --dataset-root /path/to/synthgen_dataset --predictions reports/synthgen_predictions.json --schema-id animal_v1 --output reports/synthgen_eval.json
python3 tools/smoke_synthgen.py --dataset-root data/smoke/synthgen_minishard --output-dir reports
```

References:
- [SynthGen intake guide](synthgen_intake.md)
- [SynthGen interface contract](synthgen_contract.md)

## CI incidents

CI incident memo has moved to a dedicated page:

- [CI incidents memo](ci_incidents.md)
- [Release reliability checklist](release_reliability_checklist.md)
- [1.0 stability boundary](release_1_0_stability.md)
- [Manual PDF DOI release](manual_doi_release.md)
