# YOLOZU-zisn split: local vs GPU-required

This runbook splits the original RunPod validation sweep (`YOLOZU-zisn`) into:

- local-preparable scope (CPU/macOS/Linux)
- GPU-runtime execution scope (Linux + NVIDIA)

The goal is to finish all non-hardware work locally, then run only the hardware-dependent checks on GPU hosts.

## One-command preflight

Generate a machine-readable split report:

```bash
python3 tools/gpu_validation_preflight.py --output reports/gpu_validation_preflight.json
```

The report contains:

- per-step local checks
- per-step GPU checks
- status (`ready_local`, `ready_for_gpu`, `needs_gpu_runtime`, `blocked_local`)
- summary buckets (`local_executable_steps`, `gpu_execution_required_steps`, `needs_gpu_runtime_steps`)
- command templates + DoD checklist

Manifest id: `gpu_validation_preflight`

Use `--strict` to fail when local prerequisites are missing:

```bash
python3 tools/gpu_validation_preflight.py --strict
```

Quick inspect:

```bash
python3 - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path("reports/gpu_validation_preflight.json").read_text())
print("local:", obj["summary"]["local_executable_steps"])
print("gpu required:", obj["summary"]["gpu_execution_required_steps"])
print("needs gpu runtime:", obj["summary"]["needs_gpu_runtime_steps"])
PY
```

## Sweep breakdown

1. RT-DETR pose training (AMP/DDP/resume)
- local: verify scripts/config and `torchrun` wiring
- GPU required: 2 GPU DDP execution and artifact generation

2. ONNX export + ORT parity
- local: export + parity on CPU path
- GPU preferred: ORT CUDA provider parity

3. TensorRT build/export/eval/latency
- local: command-level dry-run and wiring checks
- GPU required: engine build + TRT exporter + latency/eval

4. Safe TTT/CTTA stability (Tent/CoTTA/EATA/SAR)
- local: CLI/config plumbing and dry-run checks
- GPU required: stability and rollback behavior under real runtime

5. OpenCV-DNN parity
- local: CPU baseline path
- GPU required: CUDA backend/target parity

6. Doctor matrix capture
- local: baseline doctor payload
- GPU preferred: full runtime matrix capture (driver/CUDA/providers/etc.)

## Recommended execution order

1. Run `gpu_validation_preflight.py` locally and resolve all `blocked_local`.
2. Run GPU jobs in this order:
- training (DDP) -> ONNX/ORT parity -> TRT build/export/eval/latency -> OpenCV-DNN CUDA -> TTT/CTTA stability -> final doctor capture.
3. Archive artifacts under one run folder and attach them to issue closure.

## GitHub Actions path (machine runner)

If you are not using RunPod directly, use the machine-runner workflow:

- Workflow: `.github/workflows/gpu_zisn_pipeline.yml`
- Trigger: `workflow_dispatch`
- Stage input:
  - `zisn1`: RT-DETR pose train/ONNX + ORT CUDA parity
  - `zisn2`: TensorRT build/export/parity/eval/latency
  - `zisn3`: OpenCV-DNN CPU/CUDA parity capture + Safe TTT/CTTA stability logs + doctor
  - `all`: run all stages in order and publish consolidated logs

Published logs branch:

- `ci-logs/gpu-zisn` with `ci_logs/ci_gpu_zisn/dod_summary.json`

Notes:

- `zisn1` attempts 2-GPU DDP; when only one GPU is available, it records `single_gpu_fallback` in `ddp_status.json`.
- `zisn2` requires `NGC_API_KEY` for `nvcr.io/nvidia/tensorrt:24.08-py3`. If absent, the stage exits with `skip_reason.txt`.
- `zisn3` records OpenCV CUDA capability in `opencv_cuda_status.json`; CUDA parity artifacts are required only when status is `ok`.
- `CI_LOGS_PUSH_TOKEN` is optional. If unset, the workflow still uploads artifacts to GitHub Actions but skips updating the `ci-logs/gpu-zisn` branch.
