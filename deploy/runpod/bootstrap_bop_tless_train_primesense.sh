#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh [--help]

Download and extract the BOP T-LESS base, models, and train_primesense
archives into one BOP dataset root.

Environment:
  OUT_DIR   Destination root (default: /workspace/bop)
  PYTHON    Python executable (default: python3)
  ALLOW_PARTIAL_EXTRACT  Set to 1 only for a quota smoke; not evidence

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
PYTHON="${PYTHON:-python3}"
ALLOW_PARTIAL_EXTRACT="${ALLOW_PARTIAL_EXTRACT:-0}"

mkdir -p "${OUT_DIR}/zips"

download_args=(
  tools/download_bop_dataset.py
  --dataset tless
  --archives tless_base.zip,tless_models.zip,tless_train_primesense.zip
  --out "${OUT_DIR}"
  --cache "${OUT_DIR}/zips"
)
if [[ "${ALLOW_PARTIAL_EXTRACT}" == "1" ]]; then
  download_args+=(--allow-partial-extract)
fi
"${PYTHON}" "${download_args[@]}"

echo "[bop] dataset root: ${OUT_DIR}"
