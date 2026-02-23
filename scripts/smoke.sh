#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke.sh [options]

Options:
  -h, --help                      Show this help and exit.
  --dataset <path>                Dataset root for core smoke (default: data/smoke).
  --predictions <path>            Predictions JSON for core smoke.
                                  (default: data/smoke/predictions/predictions_dummy.json)
  --report <path>                 Output path for eval-coco dry-run report.
                                  (default: reports/smoke_coco_eval_dry_run.json)
  --synthgen-root <path>          SynthGen mini-shard root (default: data/smoke/synthgen_minishard).
  --synthgen-predictions <path>   SynthGen predictions JSON.
                                  (default: <synthgen-root>/predictions_synthgen_smoke.json)
  --output-dir <path>             Output directory for SynthGen smoke artifacts (default: reports).
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "missing value for $option" >&2
    usage >&2
    exit 2
  fi
}

if command -v yolozu >/dev/null 2>&1; then
  YOLOZU_BIN=(yolozu)
else
  YOLOZU_BIN=(python3 -m yolozu.cli)
fi

DATASET="data/smoke"
PREDICTIONS="data/smoke/predictions/predictions_dummy.json"
REPORT="reports/smoke_coco_eval_dry_run.json"
SYNTHGEN_SMOKE_ROOT="data/smoke/synthgen_minishard"
SYNTHGEN_PREDICTIONS=""
OUTPUT_DIR="reports"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="${2:-}"
      shift 2
      ;;
    --predictions)
      require_value "$1" "${2:-}"
      PREDICTIONS="${2:-}"
      shift 2
      ;;
    --report)
      require_value "$1" "${2:-}"
      REPORT="${2:-}"
      shift 2
      ;;
    --synthgen-root)
      require_value "$1" "${2:-}"
      SYNTHGEN_SMOKE_ROOT="${2:-}"
      shift 2
      ;;
    --synthgen-predictions)
      require_value "$1" "${2:-}"
      SYNTHGEN_PREDICTIONS="${2:-}"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SYNTHGEN_PREDICTIONS" ]]; then
  SYNTHGEN_PREDICTIONS="$SYNTHGEN_SMOKE_ROOT/predictions_synthgen_smoke.json"
fi

echo "[1/5] doctor"
if ! "${YOLOZU_BIN[@]}" doctor --output -; then
  echo "doctor reported environment issues; continuing smoke checks"
fi

# Prefer flag-style forms documented in smoke examples, with positional fallback
# for CLI variants that still require positional arguments.
echo "[2/5] validate dataset"
if ! "${YOLOZU_BIN[@]}" validate dataset --dataset "$DATASET" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate dataset "$DATASET" --strict
fi

echo "[3/5] validate predictions"
if ! "${YOLOZU_BIN[@]}" validate predictions --predictions "$PREDICTIONS" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict
fi

echo "[4/5] eval-coco dry-run"
"${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split val \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$REPORT"

echo "[5/5] synthgen intake smoke"
python3 tools/smoke_synthgen.py \
  --dataset-root "$SYNTHGEN_SMOKE_ROOT" \
  --predictions "$SYNTHGEN_PREDICTIONS" \
  --output-dir "$OUTPUT_DIR"

echo "smoke OK: $REPORT + $OUTPUT_DIR/smoke_synthgen_summary.json"
