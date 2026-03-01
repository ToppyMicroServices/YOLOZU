#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_git_hooks.sh

Enable repo-local git hooks by setting:
  git config core.hooksPath .githooks

This allows versioned hooks (e.g. pre-push) to run consistently.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d ".githooks" ]]; then
  echo "error: missing .githooks directory" >&2
  exit 2
fi
if [[ ! -f ".githooks/pre-push" ]]; then
  echo "error: missing .githooks/pre-push" >&2
  exit 2
fi

chmod +x .githooks/pre-push scripts/pre_push.sh

git config core.hooksPath .githooks
echo "installed: core.hooksPath=$(git config core.hooksPath)"

