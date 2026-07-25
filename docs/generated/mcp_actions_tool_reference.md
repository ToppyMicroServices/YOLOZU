# MCP/Actions tool reference (generated)

Source of truth: `yolozu.integrations.tool_runner`, `yolozu.integrations.mcp_server`, `yolozu.integrations.actions_api`.

## Public surface sets

| Surface | Tool ids | Availability |
|---|---|---|
| `mcp_live` | `ai_tools`, `generate_config`, `review_config`, `doctor`, `validate_predictions`, `validate_dataset`, `eval_coco`, `predict_images`, `parity_check`, `calibrate_predictions`, `eval_instance_seg`, `eval_long_tail`, `run_scenarios`, `convert_dataset`, `train_job`, `export_predictions_job`, `export_onnx_job`, `test_job`, `ttt_job`, `ctta_job`, `jobs_list`, `jobs_status`, `jobs_cancel`, `runs_list`, `runs_describe` | requires optional mcp dependency |
| `guaranteed_ai_safe` | `doctor`, `generate_config`, `review_config`, `validate_predictions` | deterministic lightweight guarantee |
| `config_review` | `generate_config`, `review_config` | in-process config generation and review |
| `actions_public` | `doctor`, `validate_predictions`, `validate_dataset`, `eval_coco`, `predict_images`, `parity_check`, `calibrate_predictions`, `eval_instance_seg`, `eval_long_tail`, `run_scenarios`, `convert_dataset`, `train_job`, `export_predictions_job`, `test_job`, `ttt_job`, `ctta_job`, `jobs_list`, `jobs_status`, `jobs_cancel`, `runs_list`, `runs_describe` | requires optional actions dependencies |

The `mcp_live` set is the exact list returned by the installed MCP SDK's live tool-list API; registration is discovery evidence, not an execution guarantee. Only `guaranteed_ai_safe` carries the deterministic lightweight guarantee; `actions_public` lists canonical operations shared with the Actions API.

## MCP/Actions parity

| Canonical | Category | MCP tool | Actions endpoint | Request model | Parity |
|---|---|---|---|---|---|
| `doctor` | sync | `doctor` | `POST /doctor` | `DoctorRequest` | ✅ |
| `validate_predictions` | sync | `validate_predictions` | `POST /validate/predictions` | `ValidatePredictionsRequest` | ✅ |
| `validate_dataset` | sync | `validate_dataset` | `POST /validate/dataset` | `ValidateDatasetRequest` | ✅ |
| `eval_coco` | sync | `eval_coco` | `POST /eval/coco` | `EvalCocoRequest` | ✅ |
| `predict_images` | sync | `predict_images` | `POST /predict/images` | `PredictImagesRequest` | ✅ |
| `parity_check` | sync | `parity_check` | `POST /parity/check` | `ParityCheckRequest` | ✅ |
| `calibrate_predictions` | sync | `calibrate_predictions` | `POST /calibrate/predictions` | `CalibratePredictionsRequest` | ✅ |
| `eval_instance_seg` | sync | `eval_instance_seg` | `POST /eval/instance-seg` | `EvalInstanceSegRequest` | ✅ |
| `eval_long_tail` | sync | `eval_long_tail` | `POST /eval/long-tail` | `EvalLongTailRequest` | ✅ |
| `run_scenarios` | sync | `run_scenarios` | `POST /run/scenarios` | `RunScenariosRequest` | ✅ |
| `convert_dataset` | sync | `convert_dataset` | `POST /convert/dataset` | `ConvertDatasetRequest` | ✅ |
| `train_job` | job | `train_job` | `POST /jobs/train` | `TrainJobRequest` | ✅ |
| `export_predictions_job` | job | `export_predictions_job` | `POST /jobs/export-predictions` | `ExportPredictionsJobRequest` | ✅ |
| `test_job` | job | `test_job` | `POST /jobs/test` | `TestJobRequest` | ✅ |
| `ttt_job` | job | `ttt_job` | `POST /jobs/ttt` | `TTTJobRequest` | ✅ |
| `ctta_job` | job | `ctta_job` | `POST /jobs/ctta` | `CTTAJobRequest` | ✅ |
| `jobs_list` | control | `jobs_list` | `GET /jobs` | `-` | ✅ |
| `jobs_status` | control | `jobs_status` | `GET /jobs/{job_id}` | `-` | ✅ |
| `jobs_cancel` | control | `jobs_cancel` | `POST /jobs/{job_id}/cancel` | `-` | ✅ |
| `runs_list` | control | `runs_list` | `GET /runs` | `-` | ✅ |
| `runs_describe` | control | `runs_describe` | `GET /runs/{run_id}` | `-` | ✅ |

## Parameters

### `doctor`

`tool_runner` params:
- `output` (default: `reports/doctor.json`)

### `validate_predictions`

`tool_runner` params:
- `path` (required)
- `strict` (default: `True`)

### `validate_dataset`

`tool_runner` params:
- `dataset` (required)
- `split` (default: `None`)
- `strict` (default: `True`)
- `mode` (default: `fail`)

### `eval_coco`

`tool_runner` params:
- `dataset` (required)
- `predictions` (required)
- `split` (default: `None`)
- `dry_run` (default: `True`)
- `output` (default: `reports/mcp_coco_eval.json`)
- `max_images` (default: `None`)
- `repair` (default: `False`)

### `predict_images`

`tool_runner` params:
- `input_dir` (required)
- `backend` (default: `dummy`)
- `output` (default: `reports/mcp_predict_images.json`)
- `max_images` (default: `None`)
- `dry_run` (default: `True`)
- `strict` (default: `True`)
- `force` (default: `True`)

### `parity_check`

`tool_runner` params:
- `reference` (required)
- `candidate` (required)
- `iou_thresh` (default: `0.5`)
- `score_atol` (default: `1e-06`)
- `bbox_atol` (default: `0.0001`)
- `max_images` (default: `None`)
- `image_size` (default: `None`)

### `calibrate_predictions`

`tool_runner` params:
- `dataset` (required)
- `predictions` (required)
- `method` (default: `fracal`)
- `split` (default: `None`)
- `task` (default: `auto`)
- `output` (default: `reports/mcp_calibrated_predictions.json`)
- `output_report` (default: `reports/mcp_calibration_report.json`)
- `max_images` (default: `None`)
- `force` (default: `True`)

### `eval_instance_seg`

`tool_runner` params:
- `dataset` (required)
- `predictions` (required)
- `split` (default: `None`)
- `output` (default: `reports/mcp_instance_seg_eval.json`)
- `max_images` (default: `None`)
- `min_score` (default: `None`)
- `allow_rgb_masks` (default: `False`)

### `eval_long_tail`

`tool_runner` params:
- `dataset` (required)
- `predictions` (required)
- `split` (default: `None`)
- `output` (default: `reports/mcp_long_tail_eval.json`)
- `max_images` (default: `None`)
- `max_detections` (default: `None`)

### `run_scenarios`

`tool_runner` params:
- `config` (required)
- `extra_args` (default: `None`)

### `convert_dataset`

`tool_runner` params:
- `from_format` (required)
- `output` (required)
- `data` (default: `None`)
- `args_yaml` (default: `None`)
- `split` (default: `None`)
- `task` (default: `None`)
- `coco_root` (default: `None`)
- `instances_json` (default: `None`)
- `mode` (default: `manifest`)
- `include_crowd` (default: `False`)
- `force` (default: `True`)

### `train_job`

`tool_runner` params:
- `train_config` (required)
- `run_id` (default: `None`)
- `resume` (default: `None`)

### `export_predictions_job`

`tool_runner` params:
- `dataset` (required)
- `output` (required)
- `split` (default: `None`)
- `force` (default: `True`)
- aliases: `export_onnx_job`, `POST /jobs/export-onnx`

### `test_job`

`tool_runner` params:
- `test_config` (required)
- `extra_args` (default: `None`)

### `ttt_job`

`tool_runner` params:
- `dataset` (required)
- `checkpoint` (required)
- `output` (default: `None`)
- `config` (default: `builtin:base`)
- `split` (default: `val`)
- `report` (default: `None`)
- `method` (default: `tent`)
- `preset` (default: `None`)
- `steps` (default: `1`)
- `reset` (default: `sample`)
- `device` (default: `cpu`)
- `max_images` (default: `1`)
- `force` (default: `True`)

### `ctta_job`

`tool_runner` params:
- `dataset` (required)
- `checkpoint` (required)
- `output` (default: `None`)
- `config` (default: `builtin:base`)
- `split` (default: `val`)
- `report` (default: `None`)
- `method` (default: `cotta`)
- `preset` (default: `None`)
- `steps` (default: `1`)
- `reset` (default: `stream`)
- `device` (default: `cpu`)
- `max_images` (default: `1`)
- `force` (default: `True`)

### `jobs_list`

`tool_runner` params:
- (none)

### `jobs_status`

`tool_runner` params:
- `job_id` (required)

### `jobs_cancel`

`tool_runner` params:
- `job_id` (required)

### `runs_list`

`tool_runner` params:
- `limit` (default: `20`)

### `runs_describe`

`tool_runner` params:
- `run_id` (required)
