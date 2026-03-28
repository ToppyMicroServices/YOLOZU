#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/ttt_compare.sh --boilerplate {tent,mim,cotta,eata,sar} --dataset <root> --checkpoint <ckpt> [options]

Short recommended entrypoint for baseline-vs-adapted TTT comparison.
It delegates to python3 tools/run_ttt_compare.py and uses per-method boilerplates.

Common examples:
  bash scripts/ttt_compare.sh --boilerplate tent --dataset data/smoke --split val --checkpoint /path/to.ckpt --run-dir reports/ttt_compare/tent
  bash scripts/ttt_compare.sh --boilerplate mim  --dataset data/smoke --split val --checkpoint /path/to.ckpt --run-dir reports/ttt_compare/mim
  bash scripts/ttt_compare.sh --boilerplate cotta --dataset data/smoke --split val --checkpoint /path/to.ckpt --device cuda --run-dir reports/ttt_compare/cotta

Pass --dry-run to write a plan without running export/eval.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

exec python3 "${REPO_ROOT}/tools/run_ttt_compare.py" "$@"
