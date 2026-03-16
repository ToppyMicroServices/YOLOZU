#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash deploy/runpod/run_rtdetr_pose_pose_eval_coco128.sh

Runs the RT-DETR pose evaluation sweep on coco128 with hash-locked Python dependencies.
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_IN="${DATASET_IN:-data/coco128}"
SPLIT="${SPLIT:-train2017}"
DATASET_OUT="${DATASET_OUT:-/tmp/coco128_pose}"
RUN_BASE="${RUN_BASE:-/tmp/rtdetr_pose_pose_eval}"

EPOCHS_CSV="${EPOCHS_CSV:-1,5,20}"
DEVICE="${DEVICE:-cuda}"
IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
MATCH_COST_ROT="${MATCH_COST_ROT:-0.0}"
MATCH_COST_Z="${MATCH_COST_Z:-0.0}"
MAX_IMAGES="${MAX_IMAGES:-64}"
SCORE_THRESH="${SCORE_THRESH:-0.01}"
MAX_DETS="${MAX_DETS:-100}"
CONFIG="${CONFIG:-configs/yolo26_rtdetr_pose/yolo26n.json}"

echo "[pose-eval] Installing deps (idempotent)..."
python3 tools/ci/install_with_hashes.py \
  --requirements requirements-runtime.lock \
  --requirements requirements-rtdetr-pose-extra.lock

echo "[pose-eval] Ensuring coco128..."
YOLOZU_INSECURE_SSL=1 bash tools/fetch_coco128.sh >/dev/null

echo "[pose-eval] Building pose sidecar dataset: ${DATASET_OUT}"
rm -rf "${DATASET_OUT}"
python3 tools/make_coco128_pose_dataset.py \
  --in-dataset "${DATASET_IN}" \
  --out-dataset "${DATASET_OUT}" \
  --split "${SPLIT}" \
  --link-images \
  --pose-mode bbox_yaw \
  --z-mode area \
  --max-images "${MAX_IMAGES}" >/dev/null

mkdir -p "${RUN_BASE}"

IFS=',' read -r -a epochs_arr <<< "${EPOCHS_CSV}"
for ep in "${epochs_arr[@]}"; do
  ep="$(echo "${ep}" | xargs)"
  [[ -z "${ep}" ]] && continue
  stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  run_dir="${RUN_BASE}/ep${ep}_${stamp}"
  mkdir -p "${run_dir}"

  echo
  echo "[pose-eval] === epochs=${ep} run=${run_dir} ==="

  python3 rtdetr_pose/tools/train_minimal.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_OUT}" \
    --split "${SPLIT}" \
    --device "${DEVICE}" \
    --real-images \
    --image-size "${IMG_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --use-matcher \
    --cost-rot "${MATCH_COST_ROT}" \
    --cost-z "${MATCH_COST_Z}" \
    --epochs "${ep}" \
    --run-dir "${run_dir}" \
    --checkpoint-out "${run_dir}/checkpoint.pt" \
    --onnx-out "${run_dir}/model.onnx" \
    --log-every 200 >/dev/null

  python3 tools/export_predictions.py \
    --adapter rtdetr_pose \
    --dataset "${DATASET_OUT}" \
    --split "${SPLIT}" \
    --config "${CONFIG}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device "${DEVICE}" \
    --image-size "${IMG_SIZE}" \
    --score-threshold "${SCORE_THRESH}" \
    --max-detections "${MAX_DETS}" \
    --max-images "${MAX_IMAGES}" \
    --wrap \
    --output "${run_dir}/pred.json" >/dev/null

  python3 tools/eval_pose.py \
    --dataset "${DATASET_OUT}" \
    --split "${SPLIT}" \
    --predictions "${run_dir}/pred.json" \
    --output "${run_dir}/pose_eval.json" \
    --min-score "${SCORE_THRESH}" \
    --iou-threshold 0.5 \
    --max-images "${MAX_IMAGES}" >/dev/null

  python3 -c "import json; from pathlib import Path; obj=json.loads(Path(r'${run_dir}/pose_eval.json').read_text()); m=obj.get('metrics',{}); print('rot_deg_median', m.get('rot_deg_median')); print('depth_abs_median', m.get('depth_abs_median')); print('pose_success', m.get('pose_success')); print('rot_success', m.get('rot_success')); print('trans_success', m.get('trans_success'))"
done

echo
echo "[pose-eval] Done. Runs under: ${RUN_BASE}"
