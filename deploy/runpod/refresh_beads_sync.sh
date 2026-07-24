#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash deploy/runpod/refresh_beads_sync.sh [--help]

Refresh the local Beads database from origin/beads-sync in a RunPod checkout,
including clones that initially track only one branch. This wrapper delegates
the guarded snapshot import to the repository-level refresh_beads_sync.sh.

Environment:
  REMOTE       Git remote to fetch (default: origin)
  SYNC_BRANCH  Snapshot branch to fetch (default: beads-sync)
  BD_BIN       bd executable to use (default: bd)
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

if [[ "$#" -gt 1 ]]; then
  echo "error: expected no positional arguments" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BD_BIN="${BD_BIN:-bd}"

cd "${REPO_ROOT}"

echo "[runpod] Importing the latest exported Beads snapshot"
bash "${REPO_ROOT}/refresh_beads_sync.sh"

echo "[runpod] Current bd list:"
"${BD_BIN}" list
