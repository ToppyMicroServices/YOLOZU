#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash release.sh [release-options]

Base-dir wrapper for tools/release.py.
This script reduces Python version/env mismatches by selecting an interpreter in this order:
  1) $YOLOZU_PYTHON (if set and executable)
  2) ./.venv/bin/python
  3) python3 from PATH
  4) python from PATH

All options are forwarded to tools/release.py.

Examples:
  bash release.sh
  bash release.sh --dry-run --allow-dirty --allow-non-main --output reports/release_report.dry_run.json
EOF
}

pick_python() {
  if [[ -n "${YOLOZU_PYTHON:-}" ]]; then
    if [[ -x "${YOLOZU_PYTHON}" ]]; then
      printf '%s\n' "${YOLOZU_PYTHON}"
      return 0
    fi
    echo "error: YOLOZU_PYTHON is set but not executable: ${YOLOZU_PYTHON}" >&2
    exit 2
  fi

  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "error: no python interpreter found (python3/python)." >&2
  exit 2
}

PY_BIN="$(pick_python)"
SCRIPT_PATH="$ROOT_DIR/tools/release.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "error: missing release script: $SCRIPT_PATH" >&2
  exit 2
fi

"$PY_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("error: YOLOZU release requires Python >= 3.10")
PY

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Delegated tool help:"
  exec "$PY_BIN" "$SCRIPT_PATH" --help
fi

exec "$PY_BIN" "$SCRIPT_PATH" "$@"
