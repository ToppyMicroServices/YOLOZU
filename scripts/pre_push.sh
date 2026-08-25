#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/pre_push.sh [options]

Run local quality gates intended to catch CI failures before pushing.

Options:
  -h, --help            Show this help and exit.
  --skip-ruff           Skip ruff lint gate.
  --skip-tests          Skip focused unit tests gate.
  --skip-smoke          Skip offline smoke script gate (scripts/smoke.sh).
  --skip-real-preflight Skip real-image gate preflight (data/real_multitask_fewshot).
  --prepare-real-data   Attempt to prepare tiny real-image fewshot dataset if missing
                        (requires network + license acceptance).

Notes:
- Default behavior is conservative: it will FAIL if real-image dataset is missing,
  because CI real-scenario gates depend on it.
- If you cannot download datasets on this machine, use --skip-real-preflight.
EOF
}

SKIP_RUFF=0
SKIP_TESTS=0
SKIP_SMOKE=0
SKIP_REAL=0
PREPARE_REAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-ruff)
      SKIP_RUFF=1
      shift 1
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift 1
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift 1
      ;;
    --skip-real-preflight)
      SKIP_REAL=1
      shift 1
      ;;
    --prepare-real-data)
      PREPARE_REAL=1
      shift 1
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[1/4] ruff"
if [[ "$SKIP_RUFF" == "1" ]]; then
  echo "skip ruff"
else
  ruff check .
fi

echo "[2/4] unit tests (focused)"
if [[ "$SKIP_TESTS" == "1" ]]; then
  echo "skip unit tests"
else
  python3 tools/generate_adaptive_vision_roadmap.py --check --json
  python3 -m unittest \
    tests.test_adaptive_vision_roadmap_generator \
    tests.test_adaptive_evidence_contracts \
    tests.test_adaptive_environment_profile \
    tests.test_adaptive_bundle_registry \
    tests.test_adaptive_selection_contracts \
    tests.test_schema_governance \
    tests.test_packaged_tools_manifest \
    tests.test_manifest_docs_references \
    tests.test_generated_cli_reference \
    tests.test_ssot_capability_coverage \
    tests.test_release_readiness_docs \
    tests.test_support_external_training_tool
fi

echo "[3/4] smoke (offline, repo assets)"
if [[ "$SKIP_SMOKE" == "1" ]]; then
  echo "skip smoke"
else
  bash scripts/smoke.sh
fi

echo "[4/4] real-image scenario preflight (data/real_multitask_fewshot)"
if [[ "$SKIP_REAL" == "1" ]]; then
  echo "skip real preflight"
  echo "pre-push OK (real preflight skipped)"
  exit 0
fi

if [[ "$PREPARE_REAL" == "1" ]]; then
  python3 tools/prepare_real_multitask_fewshot.py \
    --out data/real_multitask_fewshot \
    --train-images 1 \
    --val-images 1 \
    --download-if-missing \
    --allow-auto-download \
    --accept-dataset-license \
    --download-num-images 2 \
    --download-seed 0 \
    --strict-provenance \
    --force
fi

python3 - <<'PY'
from pathlib import Path

root = Path("data/real_multitask_fewshot")
images = root / "images" / "val"
labels = root / "labels" / "val"
if not (images.is_dir() and labels.is_dir()):
    raise SystemExit(
        "missing data/real_multitask_fewshot (required by CI real-scenario gates).\n"
        "Run:\n"
        "  python3 tools/prepare_real_multitask_fewshot.py --out data/real_multitask_fewshot "
        "--download-if-missing --allow-auto-download --accept-dataset-license --force\n"
        "Or bypass with:\n"
        "  bash scripts/pre_push.sh --skip-real-preflight\n"
    )
imgs = sorted([p for p in images.iterdir() if p.is_file()])
if not imgs:
    raise SystemExit("data/real_multitask_fewshot/images/val is empty")
print(f"real-image dataset OK: {len(imgs)} val images")
PY

echo "pre-push OK"
