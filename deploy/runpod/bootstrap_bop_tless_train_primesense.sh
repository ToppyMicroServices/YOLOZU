#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh [--help]

Download and extract the BOP T-LESS base and train_primesense archives.

Environment:
  OUT_DIR   Destination root (default: /workspace/bop)

The downloader records archive URLs, SHA-256 values, byte sizes, and the
CC-BY-4.0 dataset source in download_manifest.json.
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

# Where to put the extracted BOP dataset.
# Note: On many RunPod images, `/` (and therefore `/tmp`) is a small overlay FS.
# Prefer `/workspace` for large downloads.
OUT_DIR="${OUT_DIR:-/workspace/bop}"

mkdir -p "${OUT_DIR}/zips"

# HuggingFace BOP zips are not consistent about the root folder:
# - tless_base.zip extracts into <out>/tless/...
# - tless_train_primesense.zip extracts into <out>/train_primesense/...
# To keep a stable BOP_ROOT=<out>/tless, extract the split zip into <out>/tless.
python3 tools/download_bop_dataset.py \
  --dataset tless \
  --archives tless_base.zip \
  --out "${OUT_DIR}" \
  --cache "${OUT_DIR}/zips" \
  --allow-partial-extract

python3 tools/download_bop_dataset.py \
  --dataset tless \
  --archives tless_train_primesense.zip \
  --out "${OUT_DIR}/tless" \
  --cache "${OUT_DIR}/zips" \
  --allow-partial-extract

if [[ -d "${OUT_DIR}/train_primesense" && -d "${OUT_DIR}/tless" && ! -d "${OUT_DIR}/tless/train_primesense" ]]; then
  mkdir -p "${OUT_DIR}/tless"
  mv "${OUT_DIR}/train_primesense" "${OUT_DIR}/tless/train_primesense" || true
fi

echo "[bop] extracted under: ${OUT_DIR}"
