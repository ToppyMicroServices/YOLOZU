#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_external_runtime_gpu_qualification.sh \
    --output-dir PATH [--dataset-root PATH] [--runtime-root PATH]

Install pinned external YOLOX/OpenMMLab runtimes on a compatible Linux CUDA
host, execute bounded non-dry training, and retain per-lane reports. The
script continues across lane failures and exits non-zero unless all four
runtime lanes execute training.
EOF
}

OUTPUT_DIR=""
DATASET_ROOT="data/real_multitask_fewshot"
RUNTIME_ROOT="/tmp/yolozu-external-runtimes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "--output-dir is required" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "refusing to replace existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${RUNTIME_ROOT}"
python3 tools/prepare_external_runtime_smoke_datasets.py \
  --source "${DATASET_ROOT}" \
  --split train \
  --output "${OUTPUT_DIR}/datasets" \
  --max-images 6

python3 -m pip install --disable-pip-version-check \
  "numpy==1.26.4" "setuptools<81" "openmim==0.3.9" \
  "opencv-python-headless==4.10.0.84" "pycocotools==2.0.8" \
  "loguru==0.7.2" "thop==0.1.1.post2209072238" \
  "tabulate==0.9.0" "tensorboard==2.18.0"
python3 -m mim install "mmengine==0.10.7"
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install --disable-pip-version-check "xtcocotools==1.14.3"

YOLOX_COMMIT="6ddff4824372906469a7fae2dc3206c7aa4bbaee"
YOLOX_ROOT="${RUNTIME_ROOT}/YOLOX"
if [[ ! -d "${YOLOX_ROOT}/.git" ]]; then
  git clone https://github.com/Megvii-BaseDetection/YOLOX.git "${YOLOX_ROOT}"
fi
git -C "${YOLOX_ROOT}" fetch --depth 1 origin "${YOLOX_COMMIT}"
git -C "${YOLOX_ROOT}" checkout --detach "${YOLOX_COMMIT}"
python3 -m pip install --disable-pip-version-check --no-deps -e "${YOLOX_ROOT}"

MMDET_ROOT="${RUNTIME_ROOT}/mmdetection"
MMPOSE_ROOT="${RUNTIME_ROOT}/mmpose"
MMSEG_ROOT="${RUNTIME_ROOT}/mmsegmentation"
if [[ ! -d "${MMDET_ROOT}/.git" ]]; then
  git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git "${MMDET_ROOT}"
fi
if [[ ! -d "${MMPOSE_ROOT}/.git" ]]; then
  git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git "${MMPOSE_ROOT}"
fi
if [[ ! -d "${MMSEG_ROOT}/.git" ]]; then
  git clone --depth 1 --branch v1.2.2 https://github.com/open-mmlab/mmsegmentation.git "${MMSEG_ROOT}"
fi
python3 -m pip install --disable-pip-version-check -e "${MMDET_ROOT}"
python3 -m pip install --disable-pip-version-check -e "${MMPOSE_ROOT}"
python3 -m pip install --disable-pip-version-check -e "${MMSEG_ROOT}"

declare -a LANES=(yolox mmdetection mmpose mmseg)
declare -A STATUS

set +e
python3 tools/support_external_training.py train-yolox \
  --preset none \
  --dataset "${OUTPUT_DIR}/datasets/detection" \
  --split train2017 \
  --exp configs/examples/finetune_external/yolox_s_finetune_smoke.py \
  --train-script "${YOLOX_ROOT}/tools/train.py" \
  --python python3 \
  --batch 2 \
  --devices 1 \
  --work-dir "${OUTPUT_DIR}/yolox/work" \
  --output "${OUTPUT_DIR}/yolox/training_summary.json"
STATUS[yolox]=$?

python3 tools/support_external_training.py train-mmdetection \
  --preset none \
  --dataset "${OUTPUT_DIR}/datasets/detection" \
  --split train2017 \
  --config configs/examples/finetune_external/mmdetection_finetune_smoke.py \
  --train-script "${MMDET_ROOT}/tools/train.py" \
  --python python3 \
  --work-dir "${OUTPUT_DIR}/mmdetection/work" \
  --output "${OUTPUT_DIR}/mmdetection/training_summary.json"
STATUS[mmdetection]=$?

python3 tools/support_external_training.py train-mmpose \
  --preset none \
  --dataset "${OUTPUT_DIR}/datasets/keypoints" \
  --split train2017 \
  --config configs/examples/finetune_external/mmpose_finetune_smoke.py \
  --train-script "${MMPOSE_ROOT}/tools/train.py" \
  --python python3 \
  --work-dir "${OUTPUT_DIR}/mmpose/work" \
  --output "${OUTPUT_DIR}/mmpose/training_summary.json"
STATUS[mmpose]=$?

python3 tools/support_external_training.py train-mmseg \
  --preset none \
  --dataset "${OUTPUT_DIR}/datasets/segmentation" \
  --split train \
  --config configs/examples/finetune_external/mmseg_finetune_smoke.py \
  --train-script "${MMSEG_ROOT}/tools/train.py" \
  --python python3 \
  --work-dir "${OUTPUT_DIR}/mmseg/work" \
  --output "${OUTPUT_DIR}/mmseg/training_summary.json"
STATUS[mmseg]=$?
set -e

for lane in "${LANES[@]}"; do
  search_root="${OUTPUT_DIR}/${lane}"
  if [[ "${lane}" = "yolox" && -d runs/yolox_finetune ]]; then
    search_root="runs/yolox_finetune"
  fi
  checkpoint="$(find "${search_root}" -type f \( -name '*.pth' -o -name '*.pt' \) 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${checkpoint}" ]]; then
    python3 - "${checkpoint}" "${OUTPUT_DIR}/${lane}/checkpoint_evidence.json" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
digest = hashlib.sha256(path.read_bytes()).hexdigest()
pathlib.Path(sys.argv[2]).write_text(
    json.dumps({"checkpoint": str(path), "bytes": path.stat().st_size, "sha256": digest}, indent=2) + "\n"
)
PY
  fi
done

python3 - "${OUTPUT_DIR}" \
  "${STATUS[yolox]}" "${STATUS[mmdetection]}" "${STATUS[mmpose]}" "${STATUS[mmseg]}" \
  "${YOLOX_COMMIT}" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import resource
import sys

root = pathlib.Path(sys.argv[1]).resolve()
names = ["yolox", "mmdetection", "mmpose", "mmseg"]
codes = [int(value) for value in sys.argv[2:6]]
versions = {}
for package in ("torch", "yolox", "mmengine", "mmcv", "mmdet", "mmpose", "mmsegmentation"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
lanes = []
for name, code in zip(names, codes):
    report = root / name / "training_summary.json"
    payload = json.loads(report.read_text()) if report.is_file() else {}
    checkpoint = root / name / "checkpoint_evidence.json"
    lanes.append(
        {
            "id": name,
            "returncode": code,
            "training_executed": bool(payload.get("training_executed")),
            "execution_status": payload.get("execution_status"),
            "runtime_error": payload.get("runtime_error"),
            "report": str(report) if report.is_file() else None,
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest() if report.is_file() else None,
            "checkpoint_evidence": json.loads(checkpoint.read_text()) if checkpoint.is_file() else None,
        }
    )
peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
summary = {
    "schema_version": 1,
    "kind": "compatible_host_external_runtime_qualification",
    "environment": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "cuda_available": __import__("torch").cuda.is_available(),
        "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None,
        "peak_rss_bytes": peak,
    },
    "sources": {"yolox_commit": sys.argv[6]},
    "lanes": lanes,
    "all_training_executed": all(row["training_executed"] for row in lanes),
}
(root / "qualification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(root / "qualification_summary.json")
PY

for lane in "${LANES[@]}"; do
  if [[ "${STATUS[$lane]}" -ne 0 ]]; then
    exit 1
  fi
done
