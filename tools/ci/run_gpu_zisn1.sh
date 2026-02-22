#!/usr/bin/env bash
set -euo pipefail

cd /workspace
nvidia-smi
python3 --version
git config --global --add safe.directory /workspace || true

python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir -r requirements.txt onnx onnxruntime-gpu

out_dir="reports/ci_zisn1"
dataset_dir="${out_dir}/dataset"
mkdir -p "${out_dir}"

python3 tools/ci/gen_smoke_dataset.py --out "${dataset_dir}" --split train --image-stem 000001 --hw 64x64
python3 tools/ci/gen_smoke_dataset.py --out "${dataset_dir}" --split val --image-stem 000002 --hw 64x64

gpu_count="$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "detected_gpu_count=${gpu_count}"

if [[ "${gpu_count}" -ge 2 ]]; then
  ddp_mode="ddp_2gpu"
  export TORCH_DISTRIBUTED_DEFAULT_FIND_UNUSED_PARAMETERS=1
  torchrun --nproc_per_node=2 rtdetr_pose/tools/train_minimal.py \
    --config rtdetr_pose/configs/base.json \
    --dataset-root "${dataset_dir}" --split train --val-split val \
    --device cuda --amp fp16 --ddp \
    --batch-size 2 --val-batch-size 2 \
    --epochs 2 --max-steps 3 \
    --log-every 1 --val-every 1 --checkpoint-every 2 \
    --run-dir "${out_dir}/train" \
    --export-onnx --onnx-out "${out_dir}/model.onnx" \
    --onnx-meta-out "${out_dir}/model.onnx.meta.json" \
    --parity-json-out "${out_dir}/parity_train_onnxrt_cpu.json" \
    --parity-policy warn
else
  ddp_mode="single_gpu_fallback"
  python3 rtdetr_pose/tools/train_minimal.py \
    --config rtdetr_pose/configs/base.json \
    --dataset-root "${dataset_dir}" --split train --val-split val \
    --device cuda --amp fp16 \
    --batch-size 2 --val-batch-size 2 \
    --epochs 2 --max-steps 3 \
    --log-every 1 --val-every 1 --checkpoint-every 2 \
    --run-dir "${out_dir}/train" \
    --export-onnx --onnx-out "${out_dir}/model.onnx" \
    --onnx-meta-out "${out_dir}/model.onnx.meta.json" \
    --parity-json-out "${out_dir}/parity_train_onnxrt_cpu.json" \
    --parity-policy warn
fi

export DDP_MODE="${ddp_mode}"
export GPU_COUNT="${gpu_count}"
python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path("reports/ci_zisn1/ddp_status.json")
payload = {
    "ddp_mode": os.environ.get("DDP_MODE"),
    "gpu_count": int(os.environ.get("GPU_COUNT", "0") or 0),
    "checkpoint_bundle_exists": Path("reports/ci_zisn1/train/checkpoint_bundle.pt").exists(),
    "checkpoint_exists": Path("reports/ci_zisn1/train/checkpoint.pt").exists(),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

resume_from="${out_dir}/train/checkpoint_bundle.pt"
if [[ -f "${resume_from}" ]]; then
  python3 rtdetr_pose/tools/train_minimal.py \
    --config rtdetr_pose/configs/base.json \
    --dataset-root "${dataset_dir}" --split train --val-split val \
    --device cuda --amp fp16 \
    --batch-size 2 --val-batch-size 2 \
    --epochs 1 --max-steps 1 --log-every 1 --val-every 1 \
    --resume-from "${resume_from}" \
    --run-dir "${out_dir}/resume" \
    --no-export-onnx
fi

python3 tools/rtdetr_pose_backend_suite.py \
  --config rtdetr_pose/configs/base.json \
  --checkpoint "${out_dir}/train/checkpoint.pt" \
  --onnx "${out_dir}/model.onnx" \
  --backends torch,onnxrt \
  --device cuda \
  --image-size 64 \
  --batch 1 \
  --samples 1 \
  --warmup 2 \
  --iterations 10 \
  --output "${out_dir}/backend_suite_ort_cuda.json"

python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

onnx_path = Path("reports/ci_zisn1/model.onnx")
out_path = Path("reports/ci_zisn1/ort_cuda_provider_check.json")

sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
providers = list(sess.get_providers())
input_name = sess.get_inputs()[0].name
shape = [int(v) if isinstance(v, int) else v for v in sess.get_inputs()[0].shape]
array = np.zeros((1, 3, 64, 64), dtype=np.float32)
_ = sess.run(None, {input_name: array})

payload = {
    "providers": providers,
    "input_name": input_name,
    "input_shape": shape,
    "uses_cuda": "CUDAExecutionProvider" in providers,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
if not payload["uses_cuda"]:
    raise SystemExit("CUDAExecutionProvider is unavailable")
PY

python3 tools/yolozu.py doctor --output "${out_dir}/doctor_gpu.json"
ls -la "${out_dir}"
