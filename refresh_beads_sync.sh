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
  PYTHON_BIN   Python executable to use (default: python3)

Normal imports preserve newer local issue fields. This helper does not pass
--allow-stale, so it is not an older-snapshot restore command.

Legacy tombstones are imported as marked, closed placeholders because bd 1.1.0
cannot import hierarchical descendants whose deleted ancestors are skipped.
Use export_beads_snapshot.sh when publishing; it restores the exact tombstone
records and refuses to drop remote-only ids.
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
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 127
fi
if ! command -v "${BD_BIN}" >/dev/null 2>&1; then
  echo "error: bd executable not found: ${BD_BIN}" >&2
  exit 127
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 127
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

SNAPSHOT_COMPAT="${REPO_ROOT}/tools/beads_snapshot_compat.py"
if [[ ! -f "${SNAPSHOT_COMPAT}" ]]; then
  echo "error: snapshot compatibility helper not found: ${SNAPSHOT_COMPAT}" >&2
  exit 1
fi

git rev-parse --is-inside-work-tree >/dev/null
git fetch "${REMOTE}" "+refs/heads/${SYNC_BRANCH}:refs/remotes/${REMOTE}/${SYNC_BRANCH}"

snapshot="$(mktemp "${TMPDIR:-/tmp}/beads-sync.XXXXXX")"
normalized_snapshot="$(mktemp "${TMPDIR:-/tmp}/beads-sync-normalized.XXXXXX")"
backup_status="$(mktemp "${TMPDIR:-/tmp}/beads-backup-status.XXXXXX")"
backup_root="$(mktemp -d "${TMPDIR:-/tmp}/beads-refresh-backup.XXXXXX")"
backup_root="$(cd "${backup_root}" && pwd -P)"
backup_dir="${backup_root}/database"
temporary_backup_configured=0
existing_backup_url=""
preserve_backup_root=0

restore_backup_config() {
  if [[ "${temporary_backup_configured}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${existing_backup_url}" ]]; then
    if ! "${BD_BIN}" backup init "${existing_backup_url}" >/dev/null; then
      return 1
    fi
  else
    if ! "${BD_BIN}" backup remove >/dev/null; then
      return 1
    fi
  fi
  temporary_backup_configured=0
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  if ! restore_backup_config; then
    echo "error: failed to restore the prior bd backup configuration" >&2
    preserve_backup_root=1
    exit_code=70
  fi
  rm -f -- "${snapshot}" "${normalized_snapshot}" "${backup_status}"
  if [[ "${preserve_backup_root}" == "1" ]]; then
    echo "recovery backup preserved at ${backup_dir}" >&2
  else
    rm -rf -- "${backup_root}"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

git show "refs/remotes/${REMOTE}/${SYNC_BRANCH}:.beads/issues.jsonl" > "${snapshot}"

"${PYTHON_BIN}" "${SNAPSHOT_COMPAT}" normalize-import \
  --input "${snapshot}" \
  --output "${normalized_snapshot}"

"${BD_BIN}" backup status --json > "${backup_status}"
existing_backup_url="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("dolt", {}).get("backup_url", ""))' \
    "${backup_status}"
)"

"${BD_BIN}" backup init "${backup_dir}" >/dev/null
temporary_backup_configured=1
"${BD_BIN}" backup sync >/dev/null

set +e
"${BD_BIN}" import "${normalized_snapshot}" --json
import_status=$?
set -e
if [[ "${import_status}" -ne 0 ]]; then
  echo "error: bd import failed; restoring the pre-import database backup" >&2
  if ! "${BD_BIN}" backup restore "${backup_dir}" --force; then
    preserve_backup_root=1
    echo "fatal: bd import and backup restore both failed" >&2
    exit 70
  fi
  echo "restored local bd database from the pre-import backup" >&2
  exit "${import_status}"
fi

echo "refreshed local bd database from ${REMOTE}/${SYNC_BRANCH}"
