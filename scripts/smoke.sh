#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke.sh [options]

Options:
  -h, --help                      Show this help and exit.
  --dataset <path>                Dataset root for core smoke (default: data/smoke).
  --predictions <path>            Predictions JSON for core smoke.
                                  (default: data/smoke/predictions/predictions_dummy.json)
  --report <path>                 Output path for eval-coco dry-run report.
                                  (default: reports/smoke_coco_eval_dry_run.json)
  --synthgen-root <path>          SynthGen mini-shard root (default: data/smoke/synthgen_minishard).
  --synthgen-predictions <path>   SynthGen predictions JSON.
                                  (default: <synthgen-root>/predictions_synthgen_smoke.json)
  --output-dir <path>             Output directory for SynthGen smoke artifacts (default: reports).
  --demo-run-dir <path>           Run directory for instance-seg smoke demo.
                                  (default: reports/smoke_demo_instance_seg)
  --skip-demo                     Skip instance-seg demo validation.
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "missing value for $option" >&2
    usage >&2
    exit 2
  fi
}

if python3 - <<'PY' >/dev/null 2>&1
import yolozu.cli  # noqa: F401
PY
then
  YOLOZU_BIN=(python3 -m yolozu.cli)
elif command -v yolozu >/dev/null 2>&1; then
  YOLOZU_BIN=(yolozu)
else
  echo "error: neither repo-local 'python3 -m yolozu.cli' nor 'yolozu' command is available." >&2
  exit 2
fi

DATASET="data/smoke"
PREDICTIONS="data/smoke/predictions/predictions_dummy.json"
REPORT="reports/smoke_coco_eval_dry_run.json"
SYNTHGEN_SMOKE_ROOT="data/smoke/synthgen_minishard"
SYNTHGEN_PREDICTIONS=""
OUTPUT_DIR="reports"
DEMO_RUN_DIR="reports/smoke_demo_instance_seg"
SKIP_DEMO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="${2:-}"
      shift 2
      ;;
    --predictions)
      require_value "$1" "${2:-}"
      PREDICTIONS="${2:-}"
      shift 2
      ;;
    --report)
      require_value "$1" "${2:-}"
      REPORT="${2:-}"
      shift 2
      ;;
    --synthgen-root)
      require_value "$1" "${2:-}"
      SYNTHGEN_SMOKE_ROOT="${2:-}"
      shift 2
      ;;
    --synthgen-predictions)
      require_value "$1" "${2:-}"
      SYNTHGEN_PREDICTIONS="${2:-}"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --demo-run-dir)
      require_value "$1" "${2:-}"
      DEMO_RUN_DIR="${2:-}"
      shift 2
      ;;
    --skip-demo)
      SKIP_DEMO=1
      shift 1
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SYNTHGEN_PREDICTIONS" ]]; then
  SYNTHGEN_PREDICTIONS="$SYNTHGEN_SMOKE_ROOT/predictions_synthgen_smoke.json"
fi

echo "[0/7] preflight runtime deps"
python3 - <<'PY'
import importlib
import sys

required = (
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
)
missing = []
for mod, pkg in required:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover
        missing.append((pkg, str(exc)))

if missing:
    print("missing runtime deps for smoke:")
    for pkg, err in missing:
        print(f"- {pkg}: {err}")
    print("install hint: python3 -m pip install -e .")
    sys.exit(2)

print("runtime deps OK: PyYAML + numpy + Pillow")
PY

echo "[1/7] doctor"
if ! "${YOLOZU_BIN[@]}" doctor --output -; then
  echo "doctor reported environment issues; continuing smoke checks"
fi

# Prefer flag-style forms documented in smoke examples, with positional fallback
# for CLI variants that still require positional arguments.
echo "[2/7] validate dataset"
if ! "${YOLOZU_BIN[@]}" validate dataset --dataset "$DATASET" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate dataset "$DATASET" --strict
fi

echo "[3/7] validate predictions"
if ! "${YOLOZU_BIN[@]}" validate predictions --predictions "$PREDICTIONS" --strict 2>/dev/null; then
  "${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict
fi

echo "[4/7] eval-coco dry-run"
"${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split val \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$REPORT"

echo "[5/7] verify eval report"
python3 - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
if not report.is_file():
    raise SystemExit(f"missing eval report: {report}")
payload = json.loads(report.read_text(encoding="utf-8"))
if payload.get("report_schema_version") != 1:
    raise SystemExit(f"unexpected report_schema_version: {payload.get('report_schema_version')}")
if payload.get("dry_run") is not True:
    raise SystemExit("eval report must be dry_run=true in smoke")
if not isinstance(payload.get("counts"), dict):
    raise SystemExit("eval report missing counts object")
print(f"eval report OK: {report}")
PY

echo "[6/7] synthgen intake smoke"
python3 tools/smoke_synthgen.py \
  --dataset-root "$SYNTHGEN_SMOKE_ROOT" \
  --predictions "$SYNTHGEN_PREDICTIONS" \
  --output-dir "$OUTPUT_DIR"

python3 - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (
    root / "smoke_synthgen_summary.json",
    root / "smoke_synthgen_eval.json",
    root / "smoke_synthgen_overlay.png",
)
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("missing synthgen smoke artifacts:\n" + "\n".join(missing))
print("synthgen artifacts OK")
PY

if [[ "$SKIP_DEMO" != "1" ]]; then
  echo "[7/7] instance-seg demo smoke (PNG evidence)"
  rm -rf "$DEMO_RUN_DIR"
  "${YOLOZU_BIN[@]}" demo instance-seg \
    --num-images 2 \
    --image-size 64 \
    --max-instances 2 \
    --background synthetic \
    --inference none \
    --run-dir "$DEMO_RUN_DIR"

  python3 - "$DEMO_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
report = run_dir / "instance_seg_demo_report.json"
if not report.is_file():
    raise SystemExit(f"missing demo report: {report}")
payload = json.loads(report.read_text(encoding="utf-8"))
if payload.get("kind") != "instance_seg_demo":
    raise SystemExit(f"unexpected demo kind: {payload.get('kind')}")
artifacts = payload.get("artifacts") or {}
overlay_dir = artifacts.get("overlays_dir")
if not isinstance(overlay_dir, str) or not overlay_dir:
    raise SystemExit("demo report missing artifacts.overlays_dir")
overlays = sorted(Path(overlay_dir).glob("*.png"))
if not overlays:
    raise SystemExit(f"no overlay PNG generated under: {overlay_dir}")
print(f"demo overlay PNG OK: {overlays[0]}")
PY
fi

echo "smoke OK: $REPORT + $OUTPUT_DIR/smoke_synthgen_summary.json"
