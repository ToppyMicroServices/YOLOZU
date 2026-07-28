#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh [--help]

Download, safely convert, train, export, and evaluate object 6DoF pose on
BOP T-LESS. This is object pose; it does not implement human 3D skeleton pose.

Environment:
  OUT_DIR       Extracted BOP root (default: /workspace/bop)
  BOP_ROOT      T-LESS dataset root (default: $OUT_DIR/tless)
  BOP_SPLIT     BOP split (default: train_primesense)
  DATASET_OUT   YOLOZU conversion output (default: /workspace/bop-yolozu-tless-train)
  OUT_SPLIT     Converted split (default: train2017)
  MAX_TRAIN_IMAGES  Maximum converted training images (default: 200)
  MAX_VAL_IMAGES    Maximum converted/evaluated validation images (default: 50)
  PARTITION_MODULUS Fixed frame partition modulus (default: 5; remainder 0 is validation)
  VISIB_MIN     Minimum visible fraction (default: 0.2)
  CONFIG        RT-DETR pose config
  DEVICE        Training/export device (default: cuda)
  IMG_SIZE      Image size (default: 320)
  BATCH_SIZE    Batch size (default: 16)
  LR            Learning rate (default: 0.001)
  SCORE_THRESH  Prediction score threshold (default: 0.0)
  MAX_DETS      Maximum detections per image (default: 100)
  EPOCHS_CSV    Comma-separated epoch budgets (default: 1,5,20)
  SEEDS_CSV     Comma-separated seeds (default: 11,22,33)
  RUN_BASE      Fresh timestamped run parent

Each seed evaluates its deterministic zero-epoch initialization baseline before
the trained epoch budgets. Every run records config/checkpoint hashes, elapsed
seconds, metrics, and code/model/dataset license boundaries.

DATASET_OUT replacement is allowed only when the converter's ownership marker
is present. Unowned, protected, and symlink outputs are refused.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "error: unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 1) Download/extract BOP T-LESS train_primesense (real RGBD + GT pose).
OUT_DIR="${OUT_DIR:-/workspace/bop}"
bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh

# 2) Convert to YOLOZU dataset (YOLO labels + per-image sidecar with K/R/t).
BOP_ROOT="${BOP_ROOT:-${OUT_DIR}/tless}"
BOP_SPLIT="${BOP_SPLIT:-train_primesense}"
DATASET_OUT="${DATASET_OUT:-/workspace/bop-yolozu-tless-train}"
OUT_SPLIT="${OUT_SPLIT:-train2017}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-200}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-50}"
PARTITION_MODULUS="${PARTITION_MODULUS:-5}"
VISIB_MIN="${VISIB_MIN:-0.2}"

python3 tools/prepare_bop_yolozu.py \
  --bop-root "${BOP_ROOT}" \
  --split "${BOP_SPLIT}" \
  --out "${DATASET_OUT}" \
  --out-split "${OUT_SPLIT}" \
  --bbox-source bbox_visib \
  --visib-fract-min "${VISIB_MIN}" \
  --partition-modulus "${PARTITION_MODULUS}" \
  --partition-remainder 0 \
  --partition-mode exclude \
  --max-images "${MAX_TRAIN_IMAGES}" \
  --link-images \
  --overwrite

python3 tools/prepare_bop_yolozu.py \
  --bop-root "${BOP_ROOT}" \
  --split "${BOP_SPLIT}" \
  --out "${DATASET_OUT}" \
  --out-split val2017 \
  --bbox-source bbox_visib \
  --visib-fract-min "${VISIB_MIN}" \
  --partition-modulus "${PARTITION_MODULUS}" \
  --partition-remainder 0 \
  --partition-mode include \
  --max-images "${MAX_VAL_IMAGES}" \
  --link-images \
  --append-owned

# 3) Train → export predictions → COCOeval + object-pose eval on held-out frames.
CONFIG="${CONFIG:-rtdetr_pose/configs/bop_tless_smoke.json}"
DEVICE="${DEVICE:-cuda}"
IMG_SIZE="${IMG_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
SCORE_THRESH="${SCORE_THRESH:-0.0}"
MAX_DETS="${MAX_DETS:-100}"
EPOCHS_CSV="${EPOCHS_CSV:-1,5,20}"
SEEDS_CSV="${SEEDS_CSV:-11,22,33}"
RUN_BASE="${RUN_BASE:-/workspace/runs/rtdetr_pose_bop_tless}"

mkdir -p "${RUN_BASE}"
IFS=',' read -r -a epochs_arr <<< "${EPOCHS_CSV}"
IFS=',' read -r -a seeds_arr <<< "${SEEDS_CSV}"

train_checkpoint() {
  local run_dir="$1"
  local epochs="$2"
  local seed="$3"

  python3 rtdetr_pose/tools/train_minimal.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_OUT}" \
    --split "${OUT_SPLIT}" \
    --val-split val2017 \
    --device "${DEVICE}" \
    --real-images \
    --image-size "${IMG_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --use-matcher \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --run-dir "${run_dir}" \
    --checkpoint-out "${run_dir}/checkpoint.pt" \
    --onnx-out "${run_dir}/model.onnx" \
    --log-every 200 >/dev/null
}

evaluate_checkpoint() {
  local run_dir="$1"
  local run_kind="$2"
  local seed="$3"
  local epochs="$4"
  local started_epoch="$5"

  python3 tools/export_predictions.py \
    --adapter rtdetr_pose \
    --dataset "${DATASET_OUT}" \
    --split val2017 \
    --config "${CONFIG}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device "${DEVICE}" \
    --image-size "${IMG_SIZE}" \
    --score-threshold "${SCORE_THRESH}" \
    --max-detections "${MAX_DETS}" \
    --max-images "${MAX_VAL_IMAGES}" \
    --wrap \
    --output "${run_dir}/pred.json" >/dev/null

  python3 tools/eval_suite.py \
    --dataset "${DATASET_OUT}" \
    --split val2017 \
    --predictions-glob "${run_dir}/pred.json" \
    --bbox-format cxcywh_norm \
    --max-images "${MAX_VAL_IMAGES}" \
    --output "${run_dir}/eval_suite.json" >/dev/null

  python3 tools/eval_pose.py \
    --dataset "${DATASET_OUT}" \
    --split val2017 \
    --predictions "${run_dir}/pred.json" \
    --min-score "${SCORE_THRESH}" \
    --iou-threshold 0.5 \
    --max-images "${MAX_VAL_IMAGES}" \
    --output "${run_dir}/pose_eval.json" >/dev/null

  local finished_epoch
  finished_epoch="$(date +%s)"
  RUN_DIR="${run_dir}" \
  RUN_KIND="${run_kind}" \
  RUN_SEED="${seed}" \
  RUN_EPOCHS="${epochs}" \
  STARTED_EPOCH="${started_epoch}" \
  FINISHED_EPOCH="${finished_epoch}" \
  CONFIG_PATH="${CONFIG}" \
  DATASET_PATH="${DATASET_OUT}" \
  DOWNLOAD_MANIFEST_PATH="${OUT_DIR}/download_manifest.json" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run = Path(os.environ["RUN_DIR"])
config = Path(os.environ["CONFIG_PATH"])
checkpoint = run / "checkpoint.pt"
suite = json.loads((run / "eval_suite.json").read_text())
pose_report = json.loads((run / "pose_eval.json").read_text())
metrics = suite["results"][0].get("metrics", {})
pose = pose_report.get("metrics", {})
payload = {
    "schema_version": 1,
    "kind": "bop_tless_diagnostic_run",
    "run_kind": os.environ["RUN_KIND"],
    "seed": int(os.environ["RUN_SEED"]),
    "epochs": int(os.environ["RUN_EPOCHS"]),
    "runtime_seconds": int(os.environ["FINISHED_EPOCH"]) - int(os.environ["STARTED_EPOCH"]),
    "config": str(config),
    "config_sha256": sha256(config),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "dataset": os.environ["DATASET_PATH"],
    "dataset_download_manifest": os.environ["DOWNLOAD_MANIFEST_PATH"],
    "dataset_license": "CC-BY-4.0",
    "model_implementation_license": "Apache-2.0",
    "metrics": metrics,
    "pose_metrics": pose,
}
(run / "run_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(
    "run_kind", payload["run_kind"],
    "seed", payload["seed"],
    "epochs", payload["epochs"],
    "map50_95", metrics.get("map50_95"),
    "rot_deg_median", pose.get("rot_deg_median"),
    "trans_l2_median", pose.get("trans_l2_median"),
    "pose_success", pose.get("pose_success"),
    "add_mean", pose.get("add_mean"),
    "adds_mean", pose.get("adds_mean"),
)
PY
}

for seed in "${seeds_arr[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  [[ -z "${seed}" ]] && continue

  stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  baseline_dir="${RUN_BASE}/seed${seed}_baseline_${stamp}"
  mkdir -p "${baseline_dir}"
  echo
  echo "[bop-tless] === seed=${seed} baseline=zero_epoch run=${baseline_dir} ==="
  started_epoch="$(date +%s)"
  train_checkpoint "${baseline_dir}" 0 "${seed}"
  evaluate_checkpoint "${baseline_dir}" baseline "${seed}" 0 "${started_epoch}"

  for ep in "${epochs_arr[@]}"; do
    ep="$(echo "${ep}" | xargs)"
    [[ -z "${ep}" ]] && continue
    stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    run_dir="${RUN_BASE}/seed${seed}_ep${ep}_${stamp}"
    mkdir -p "${run_dir}"
    echo
    echo "[bop-tless] === seed=${seed} epochs=${ep} run=${run_dir} ==="
    started_epoch="$(date +%s)"
    train_checkpoint "${run_dir}" "${ep}" "${seed}"
    evaluate_checkpoint "${run_dir}" trained "${seed}" "${ep}" "${started_epoch}"
  done
done

echo
echo "[bop-tless] Done. Runs under: ${RUN_BASE}"
