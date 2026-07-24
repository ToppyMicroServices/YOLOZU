#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash refresh_beads_sync.sh [--help]

Fetch the exported Beads issue snapshot from the remote beads-sync branch and
import it into the local bd database. The current working branch and its
.beads/issues.jsonl file are not changed.

Environment:
  REMOTE       Git remote to fetch (default: origin)
  SYNC_BRANCH  Snapshot branch to fetch (default: beads-sync)
  BD_BIN       bd executable to use (default: bd)

Normal imports preserve newer local issue fields. This helper does not pass
--allow-stale, so it is not an older-snapshot restore command.
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

REMOTE="${REMOTE:-origin}"
SYNC_BRANCH="${SYNC_BRANCH:-beads-sync}"
BD_BIN="${BD_BIN:-bd}"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 127
fi
if ! command -v "${BD_BIN}" >/dev/null 2>&1; then
  echo "error: bd executable not found: ${BD_BIN}" >&2
  exit 127
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

git rev-parse --is-inside-work-tree >/dev/null
git fetch "${REMOTE}" "+refs/heads/${SYNC_BRANCH}:refs/remotes/${REMOTE}/${SYNC_BRANCH}"

snapshot="$(mktemp "${TMPDIR:-/tmp}/beads-sync.XXXXXX")"
trap 'rm -f "${snapshot}"' EXIT

git show "refs/remotes/${REMOTE}/${SYNC_BRANCH}:.beads/issues.jsonl" > "${snapshot}"

"${BD_BIN}" import "${snapshot}" --json

echo "refreshed local bd database from ${REMOTE}/${SYNC_BRANCH}"
