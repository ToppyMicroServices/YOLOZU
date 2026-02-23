#!/usr/bin/env bash
set -euo pipefail

cd /workspace
nvidia-smi
python3 --version
git config --global --add safe.directory /workspace || true

python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir -r requirements.txt onnx onnxruntime-gpu nvidia-cudnn-cu12 opencv-python-headless

cudnn_lib="$(python3 - <<'PY'
from pathlib import Path

try:
    import nvidia.cudnn  # type: ignore[attr-defined]
except Exception:
    print("")
else:
    print(str((Path(nvidia.cudnn.__file__).resolve().parent / "lib")))
PY
)"
if [[ -n "${cudnn_lib}" && -d "${cudnn_lib}" ]]; then
  export LD_LIBRARY_PATH="${cudnn_lib}:${LD_LIBRARY_PATH:-}"
fi

out_dir="reports/ci_zisn3"
dataset_dir="${out_dir}/dataset"
mkdir -p "${out_dir}"

onnx_path="${out_dir}/model_rtdetr_like.onnx"
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

model = onnx.load("reports/ci_zisn3/model_rtdetr_like.onnx")
outputs = {out.name for out in model.graph.output}
raise SystemExit(0 if {"boxes", "logits"}.issubset(outputs) else 1)
PY
  then
    model_source="zisn1_artifact"
  else
    rm -f "${onnx_path}"
  fi
fi

if [[ ! -f "${onnx_path}" ]]; then
  python3 - <<'PY'
from pathlib import Path

import torch


class DummyRtDetr(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.4, 0.4], [0.3, 0.4, 0.2, 0.2]]],
            dtype=torch.float32,
        )
        logits = torch.tensor(
            [[[0.1, 2.0, -1.0], [1.5, 0.2, -0.5]]],
            dtype=torch.float32,
        )
        self.register_buffer("boxes_template", boxes)
        self.register_buffer("logits_template", logits)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = images[:, :1, :1, :1] * 0.0
        boxes = self.boxes_template + anchor.view(1, 1, 1)
        logits = self.logits_template + anchor.view(1, 1, 1)
        return boxes, logits


out_path = Path("reports/ci_zisn3/model_rtdetr_like.onnx")
model = DummyRtDetr().eval()
dummy_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    str(out_path),
    input_names=["images"],
    output_names=["boxes", "logits"],
    opset_version=17,
)
print("wrote", out_path)
PY
fi

if [[ -d /workspace/.tmp_zisn1/reports/ci_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/reports/ci_zisn1/dataset "${dataset_dir}"
elif [[ -d /workspace/.tmp_zisn1/ci_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/ci_zisn1/dataset "${dataset_dir}"
elif [[ -d /workspace/.tmp_zisn1/dataset ]]; then
  cp -a /workspace/.tmp_zisn1/dataset "${dataset_dir}"
else
  python3 tools/ci/gen_smoke_dataset.py --out "${dataset_dir}" --split val --hw 64x64
fi

python3 - <<PY
import json
from pathlib import Path

out = Path("reports/ci_zisn3/model_source.json")
out.write_text(json.dumps({"model_source": "${model_source}"}, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

input_size="$(python3 - <<'PY'
import onnx

model = onnx.load("reports/ci_zisn3/model_rtdetr_like.onnx")
dims = model.graph.input[0].type.tensor_type.shape.dim
h = int(dims[2].dim_value) if len(dims) > 2 and int(dims[2].dim_value or 0) > 0 else 64
w = int(dims[3].dim_value) if len(dims) > 3 and int(dims[3].dim_value or 0) > 0 else 64
print(max(h, w))
PY
)"

python3 tools/export_predictions_opencv_dnn_rtdetr.py \
  --dataset "${dataset_dir}" \
  --split val \
  --onnx "${onnx_path}" \
  --imgsz "${input_size}" \
  --boxes-output boxes \
  --logits-output logits \
  --boxes-format cxcywh \
  --boxes-scale norm \
  --scores-activation softmax \
  --background-class last \
  --score-thr 0.0 \
  --topk 20 \
  --dnn-backend opencv \
  --dnn-target cpu \
  --strict \
  --output "${out_dir}/pred_opencv_cpu.json" \
  --meta-output "${out_dir}/pred_opencv_cpu.meta.json"

python3 - <<'PY'
import json
from pathlib import Path
import cv2

symbols = bool(hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA"))
cuda_count = 0
if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
    try:
        cuda_count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cuda_count = 0

payload = {
    "opencv_version": str(getattr(cv2, "__version__", "")),
    "has_dnn_cuda_symbols": symbols,
    "cuda_device_count": int(cuda_count),
    "cuda_path_ready": bool(symbols and cuda_count > 0),
}
out = Path("reports/ci_zisn3/opencv_runtime.json")
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

cuda_ready="$(python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("reports/ci_zisn3/opencv_runtime.json").read_text(encoding="utf-8"))
print("1" if payload.get("cuda_path_ready") else "0")
PY
)"

if [[ "${cuda_ready}" == "1" ]]; then
  python3 tools/export_predictions_opencv_dnn_rtdetr.py \
    --dataset "${dataset_dir}" \
    --split val \
    --onnx "${onnx_path}" \
    --imgsz "${input_size}" \
    --boxes-output boxes \
    --logits-output logits \
    --boxes-format cxcywh \
    --boxes-scale norm \
    --scores-activation softmax \
    --background-class last \
    --score-thr 0.0 \
    --topk 20 \
    --dnn-backend cuda \
    --dnn-target cuda \
    --strict \
    --output "${out_dir}/pred_opencv_cuda.json" \
    --meta-output "${out_dir}/pred_opencv_cuda.meta.json" \
    2>&1 | tee "${out_dir}/opencv_cuda.log"

  python3 tools/check_predictions_parity.py \
    --reference "${out_dir}/pred_opencv_cpu.json" \
    --candidate "${out_dir}/pred_opencv_cuda.json" \
    --image-size "${input_size}" \
    --iou-thresh 0.99 \
    --score-atol 1e-4 \
    --bbox-atol 1e-4 \
    > "${out_dir}/parity_opencv_cpu_vs_cuda.json"

  python3 - <<'PY'
import json
from pathlib import Path
out = Path("reports/ci_zisn3/opencv_cuda_status.json")
payload = {"status": "ok", "reason": None}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY
else
  python3 - <<'PY'
import json
from pathlib import Path

runtime = json.loads(Path("reports/ci_zisn3/opencv_runtime.json").read_text(encoding="utf-8"))
out = Path("reports/ci_zisn3/opencv_cuda_status.json")
payload = {
    "status": "skipped",
    "reason": "opencv_cuda_unavailable",
    "runtime": runtime,
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY
fi

for method in tent cotta eata sar; do
  python3 tools/export_predictions.py \
    --adapter rtdetr_pose \
    --dataset "${dataset_dir}" \
    --split val \
    --config rtdetr_pose/configs/base.json \
    --device cuda \
    --image-size 64 \
    --max-images 1 \
    --score-threshold 0.0 \
    --max-detections 20 \
    --ttt \
    --ttt-method "${method}" \
    --ttt-reset sample \
    --ttt-steps 1 \
    --ttt-batch-size 1 \
    --ttt-lr 1e-4 \
    --ttt-update-filter norm_only \
    --ttt-max-batches 1 \
    --ttt-max-grad-norm 1.0 \
    --ttt-max-update-norm 1.0 \
    --ttt-max-total-update-norm 1.0 \
    --ttt-max-loss-ratio 3.0 \
    --ttt-log-out "${out_dir}/ttt_${method}.log.json" \
    --wrap \
    --output "${out_dir}/pred_ttt_${method}.json"
done

python3 - <<'PY'
import json
from pathlib import Path

out_dir = Path("reports/ci_zisn3")
methods = ["tent", "cotta", "eata", "sar"]
summary = {"methods": {}, "totals": {"rollback_steps": 0, "instability_events": 0, "stopped_early": 0}}

for method in methods:
    log_path = out_dir / f"ttt_{method}.log.json"
    if not log_path.exists():
        summary["methods"][method] = {"status": "missing_log"}
        continue
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    report = (((obj or {}).get("ttt") or {}).get("report") or {})
    step_metrics = list(report.get("step_metrics") or [])
    rollback_steps = sum(1 for step in step_metrics if bool((step or {}).get("rolled_back")))
    non_finite_steps = sum(
        1 for step in step_metrics if bool((step or {}).get("non_finite_fields"))
    )
    stop_reason = report.get("stop_reason")
    stopped_early = bool(report.get("stopped_early"))
    instability = int(non_finite_steps + (1 if stopped_early else 0))
    summary["methods"][method] = {
        "status": "ok",
        "steps_run": int(report.get("steps_run") or 0),
        "rollback_steps": int(rollback_steps),
        "non_finite_steps": int(non_finite_steps),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "warnings": list(report.get("warnings") or []),
    }
    summary["totals"]["rollback_steps"] += int(rollback_steps)
    summary["totals"]["instability_events"] += int(instability)
    summary["totals"]["stopped_early"] += int(1 if stopped_early else 0)

out_path = out_dir / "ttt_stability_summary.json"
out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
PY

python3 tools/yolozu.py doctor --output "${out_dir}/doctor_gpu.json"
ls -la "${out_dir}"
