#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/ttt_compare.sh --method {tent,mim,mim_probe,cotta,eata,sar} --data <root> --weights <ckpt> [options]

Short entrypoint for a fail-closed baseline-vs-adapted TTT local diagnostic.
It delegates to python3 tools/run_ttt_compare.py and uses per-method boilerplates.
The selected checkpoint must pass full compatibility preflight.

Common examples:
  bash scripts/ttt_compare.sh --method tent --data data/smoke --weights checkpoints/rtdetr_pose.pt --out reports/ttt_compare/tent -n 1 --no-eval
  bash scripts/ttt_compare.sh --method mim --data data/smoke --weights checkpoints/rtdetr_pose_mim.pt --out reports/ttt_compare/mim -n 1 --no-eval
  bash scripts/ttt_compare.sh --method cotta --data data/smoke --weights checkpoints/rtdetr_pose.pt --out reports/ttt_compare/cotta -n 1 --dry-run

Pass --dry-run to write a plan without running export/eval.
Reports are local diagnostics; execution does not establish efficacy.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

exec python3 "${REPO_ROOT}/tools/run_ttt_compare.py" "$@"
