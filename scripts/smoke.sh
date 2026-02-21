#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v yolozu >/dev/null 2>&1; then
  YOLOZU_BIN=(yolozu)
else
  YOLOZU_BIN=(python3 -m yolozu.cli)
fi

DATASET="data/smoke"
PREDICTIONS="data/smoke/predictions/predictions_dummy.json"
REPORT="reports/smoke_coco_eval_dry_run.json"

echo "[1/4] doctor"
if ! "${YOLOZU_BIN[@]}" doctor --output -; then
  echo "doctor reported environment issues; continuing smoke checks"
fi

# Prefer flag-style forms documented in smoke examples, with positional fallback
# for CLI variants that still require positional arguments.
echo "[2/4] validate dataset"
if ! "${YOLOZU_BIN[@]}" validate dataset --dataset "$DATASET" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate dataset "$DATASET" --strict
fi

echo "[3/4] validate predictions"
if ! "${YOLOZU_BIN[@]}" validate predictions --predictions "$PREDICTIONS" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict
fi

echo "[4/4] eval-coco dry-run"
"${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split val \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$REPORT"

echo "smoke OK: $REPORT"