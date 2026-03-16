#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash tools/ci/run_gpu_zisn2.sh

Runs the YOLOZU ZISN-2 GPU validation stage with hash-locked Python dependencies.
EOF
  exit 0
fi

cd /workspace
nvidia-smi
python3 --version
git config --global --add safe.directory /workspace || true

python3 tools/ci/install_with_hashes.py \
  --requirements requirements-runtime.lock \
  --requirements requirements-zisn2-extra.lock

out_dir="reports/ci_zisn2"
dataset_dir="${out_dir}/dataset"
mkdir -p "${out_dir}"

onnx_path="${out_dir}/model.onnx"
candidate_model=""
if [[ -f /workspace/.tmp_zisn1/reports/ci_zisn1/model.onnx ]]; then
  candidate_model="/workspace/.tmp_zisn1/reports/ci_zisn1/model.onnx"
elif [[ -f /workspace/.tmp_zisn1/ci_zisn1/model.onnx ]]; then
  candidate_model="/workspace/.tmp_zisn1/ci_zisn1/model.onnx"
elif [[ -f /workspace/.tmp_zisn1/model.onnx ]]; then
  candidate_model="/workspace/.tmp_zisn1/model.onnx"
fi

model_source="dummy_generated"
if [[ -n "${candidate_model}" ]]; then
  cp "${candidate_model}" "${onnx_path}"
  if python3 - <<'PY'
import onnx

model = onnx.load("reports/ci_zisn2/model.onnx")
outputs = {out.name for out in model.graph.output}
raise SystemExit(0 if "output0" in outputs else 1)
PY
  then
    model_source="zisn1_artifact"
  else
    rm -f "${onnx_path}"
  fi
fi

if [[ ! -f "${onnx_path}" ]]; then
  python3 tools/ci/gen_dummy_dets_onnx.py \
    --out "${onnx_path}" \
    --input-name images \
    --output-name output0 \
    --shape 1x3x64x64 \
    --opset 17 \
    --ir-version 11
fi

if [[ -d /workspace/.tmp_zisn1/reports/ci_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/reports/ci_zisn1/dataset "${dataset_dir}"
elif [[ -d /workspace/.tmp_zisn1/ci_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/ci_zisn1/dataset "${dataset_dir}"
elif [[ -d /workspace/.tmp_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/dataset "${dataset_dir}"
else
  python3 tools/ci/gen_smoke_dataset.py --out "${dataset_dir}" --split val --hw 48x64
fi

python3 - <<PY
import json
from pathlib import Path

out = Path("reports/ci_zisn2/model_source.json")
out.write_text(json.dumps({"model_source": "${model_source}"}, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

input_size="$(python3 - <<'PY'
import onnx

model = onnx.load("reports/ci_zisn2/model.onnx")
shape = model.graph.input[0].type.tensor_type.shape.dim
h = shape[2].dim_value if len(shape) > 2 else 64
w = shape[3].dim_value if len(shape) > 3 else 64
if h <= 0:
    h = 64
if w <= 0:
    w = 64
print(int(max(h, w)))
PY
)"

engine_path="${out_dir}/model_fp16.plan"
engine_meta="${out_dir}/engine.meta.json"

python3 tools/build_trt_engine.py \
  --onnx "${onnx_path}" \
  --engine "${engine_path}" \
  --precision fp16 \
  --meta-output "${engine_meta}" \
  2>&1 | tee "${out_dir}/build_trt_engine.log"

python3 tools/export_predictions_onnxrt.py \
  --dataset "${dataset_dir}" \
  --split val \
  --onnx "${onnx_path}" \
  --input-name images \
  --combined-output output0 \
  --boxes-scale norm \
  --min-score 0.0 \
  --topk 10 \
  --wrap \
  --strict \
  --output "${out_dir}/pred_onnxrt.json"

python3 tools/export_predictions_trt.py \
  --dataset "${dataset_dir}" \
  --split val \
  --engine "${engine_path}" \
  --input-name images \
  --combined-output output0 \
  --boxes-scale norm \
  --min-score 0.0 \
  --topk 10 \
  --imgsz "${input_size}" \
  --wrap \
  --strict \
  --output "${out_dir}/pred_trt.json"

python3 tools/check_predictions_parity.py \
  --reference "${out_dir}/pred_onnxrt.json" \
  --candidate "${out_dir}/pred_trt.json" \
  --image-size "${input_size}" \
  --iou-thresh 0.99 \
  --score-atol 1e-4 \
  --bbox-atol 1e-4 \
  > "${out_dir}/parity_trt_vs_onnxrt.json"

cat "${out_dir}/parity_trt_vs_onnxrt.json"

python3 tools/eval_suite.py \
  --dataset "${dataset_dir}" \
  --split val \
  --predictions-glob "${out_dir}/pred_trt.json" \
  --strict \
  --dry-run \
  --output "${out_dir}/eval_trt.json"

python3 tools/measure_trt_latency.py \
  --engine "${engine_path}" \
  --input-name images \
  --shape "1x3x${input_size}x${input_size}" \
  --iterations 50 \
  --warmup 10 \
  --output "${out_dir}/latency_trt.json"

if trtexec --help 2>&1 | grep -q -- "--warmUp"; then
  trtexec --loadEngine="${engine_path}" --iterations=1 --warmUp=0 > "${out_dir}/trtexec_load.log"
else
  trtexec --loadEngine="${engine_path}" --iterations=1 > "${out_dir}/trtexec_load.log"
fi

python3 tools/yolozu.py doctor --output "${out_dir}/doctor_gpu.json"
ls -la "${out_dir}"
