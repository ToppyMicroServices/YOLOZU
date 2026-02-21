#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[1/4] smoke gate"
bash scripts/smoke.sh

echo "[2/4] artifact check"
test -f reports/smoke_coco_eval_dry_run.json

echo "[3/4] lint gate"
"$PYTHON_BIN" -m ruff check tools yolozu tests

echo "[4/4] focused pytest gate"
"$PYTHON_BIN" -m pytest -q \
  tests/test_eval_suite_export_settings.py \
  tests/test_eval_suite_determinism.py \
  tests/test_check_map_targets_contract.py \
  tests/test_yolo26_protocol.py \
  tests/test_predictions.py

echo "pre-PR quality checklist passed"