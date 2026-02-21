# YOLOZU docs

Use this page as the docs index for both humans and agents.
All examples below use repository-real paths (`data/smoke`, `reports/...`) to reduce copy-paste mistakes.

## 0) Offline copy-paste smoke (single command)

The fastest safety check from repo root is:

```bash
bash scripts/smoke.sh
```

Expected report output:

- `reports/smoke_coco_eval_dry_run.json`

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

## C) Contracts (predictions / adapter / TTT protocol)

Use this path to confirm JSON contracts and manifest consistency before bigger runs.

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

## CI incidents

CI incident memo has moved to a dedicated page:

- [CI incidents memo](ci_incidents.md)
