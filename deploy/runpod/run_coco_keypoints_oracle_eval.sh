#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash deploy/runpod/run_coco_keypoints_oracle_eval.sh

Builds a COCO keypoints validation dataset and runs the oracle OKS evaluation with
hash-locked Python dependencies.
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COCO_ROOT="${COCO_ROOT:-/workspace/coco2017}"
DATASET_OUT="${DATASET_OUT:-/workspace/coco2017_kp_yolozu}"
MAX_IMAGES="${MAX_IMAGES:-500}"

python3 tools/ci/install_with_hashes.py \
  --requirements requirements-runtime.lock \
  --requirements requirements-runpod-keypoints-extra.lock

bash deploy/runpod/bootstrap_coco_keypoints_val2017.sh

rm -rf "${DATASET_OUT}"
python3 tools/prepare_coco_keypoints_yolozu.py \
  --coco-root "${COCO_ROOT}" \
  --annotations "annotations/person_keypoints_val2017.json" \
  --images-dir "val2017" \
  --out "${DATASET_OUT}" \
  --out-split "val2017" \
  --min-kps 1 \
  --max-images "${MAX_IMAGES}"

RUN_DIR="/workspace/runs/coco_kp_oracle_$(date -u +%Y-%m-%dT%H-%M-%SZ)"
mkdir -p "${RUN_DIR}"

python3 tools/make_oracle_keypoints_predictions.py \
  --dataset "${DATASET_OUT}" \
  --split "val2017" \
  --max-images "${MAX_IMAGES}" \
  --output "${RUN_DIR}/pred_oracle.json"

python3 tools/eval_keypoints.py \
  --dataset "${DATASET_OUT}" \
  --split "val2017" \
  --predictions "${RUN_DIR}/pred_oracle.json" \
  --oks \
  --output "${RUN_DIR}/keypoints_eval_oks.json"

echo "${RUN_DIR}"
