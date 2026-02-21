#!/usr/bin/env bash
set -euo pipefail

python3 -m ruff check tools yolozu tests
python3 -m pytest -q \
  tests/test_eval_suite_export_settings.py \
  tests/test_eval_suite_determinism.py \
  tests/test_check_map_targets_contract.py \
  tests/test_yolo26_protocol.py \
  tests/test_predictions.py
