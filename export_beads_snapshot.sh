#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash export_beads_snapshot.sh <beads-sync-issues.jsonl>

Export the local bd issue state without dropping ids from the current remote
snapshot. The destination must already contain the latest beads-sync
.beads/issues.jsonl baseline (pull the beads-sync worktree first).

Legacy tombstone placeholders used by refresh_beads_sync.sh are restored to
their exact remote tombstone JSON before the destination is atomically replaced.
The helper does not modify .beads/interactions.jsonl.

Environment:
  BD_BIN       bd executable to use (default: bd)
  PYTHON_BIN   Python executable to use (default: python3)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    echo "error: destination snapshot is required" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "$#" -ne 1 ]]; then
  echo "error: expected exactly one destination snapshot" >&2
  usage >&2
  exit 2
fi

BD_BIN="${BD_BIN:-bd}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${BD_BIN}" >/dev/null 2>&1; then
  echo "error: bd executable not found: ${BD_BIN}" >&2
  exit 127
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 127
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_COMPAT="${REPO_ROOT}/tools/beads_snapshot_compat.py"
if [[ ! -f "${SNAPSHOT_COMPAT}" ]]; then
  echo "error: snapshot compatibility helper not found: ${SNAPSHOT_COMPAT}" >&2
  exit 1
fi

destination_dir="$(cd "$(dirname "$1")" && pwd)"
destination="${destination_dir}/$(basename "$1")"
if [[ ! -f "${destination}" ]]; then
  echo "error: destination must contain the pulled remote baseline: ${destination}" >&2
  exit 1
fi

baseline="$(mktemp "${TMPDIR:-/tmp}/beads-export-baseline.XXXXXX")"
local_snapshot="$(mktemp "${TMPDIR:-/tmp}/beads-export-local.XXXXXX")"
merged_snapshot="$(mktemp "${destination_dir}/.beads-export-merged.XXXXXX")"
cleanup() {
  rm -f "${baseline}" "${local_snapshot}" "${merged_snapshot}"
}
trap cleanup EXIT

cp "${destination}" "${baseline}"
"${BD_BIN}" export -o "${local_snapshot}"
"${PYTHON_BIN}" "${SNAPSHOT_COMPAT}" restore-export \
  --local "${local_snapshot}" \
  --baseline "${baseline}" \
  --output "${merged_snapshot}"
"${PYTHON_BIN}" -c \
  'import os, stat, sys; os.chmod(sys.argv[2], stat.S_IMODE(os.stat(sys.argv[1]).st_mode))' \
  "${destination}" "${merged_snapshot}"
mv "${merged_snapshot}" "${destination}"

echo "exported compatible Beads snapshot to ${destination}"
