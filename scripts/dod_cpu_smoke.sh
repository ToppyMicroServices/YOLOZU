#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: bash scripts/dod_cpu_smoke.sh [options]

Runs the public CPU-only DoD path:
  doctor --proof -> demo -> validate dataset -> validate predictions -> eval-coco dry-run

Options:
  -h, --help              Show this help and exit.
  --run-dir <path>        Output directory for DoD artifacts (default: reports/dod_cpu_smoke).
  --split <name>          Split for proof dataset validation/eval (default: val2017).
USAGE
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

can_run_python() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import sys
print(sys.version)
PY
}

pick_python() {
  local candidates=()
  if [[ -n "${YOLOZU_PYTHON:-}" ]]; then
    candidates+=("${YOLOZU_PYTHON}")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    candidates+=("$ROOT_DIR/.venv/bin/python")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && can_run_python "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

RUN_DIR="reports/dod_cpu_smoke"
SPLIT="val2017"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-dir)
      require_value "$1" "${2:-}"
      RUN_DIR="${2:-}"
      shift 2
      ;;
    --split)
      require_value "$1" "${2:-}"
      SPLIT="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PY_BIN="$(pick_python || true)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PY_BIN" ]]; then
  echo "error: no runnable Python interpreter found." >&2
  echo "hint: python3 -m pip install -e . or python3 -m pip install -U yolozu" >&2
  exit 2
fi
YOLOZU_BIN=("$PY_BIN" -m yolozu.cli)

mkdir -p "$RUN_DIR"

DOCTOR_JSON="$RUN_DIR/doctor.json"
PROOF_DIR="$RUN_DIR/doctor_proof"
PROOF_REPORT="$PROOF_DIR/proof_report.json"
DEMO_DIR="$RUN_DIR/demo_instance_seg"
EVAL_REPORT="$RUN_DIR/eval_coco_dry_run.json"
DOD_REPORT="$RUN_DIR/dod_cpu_smoke_report.json"

echo "[1/5] doctor --proof"
"${YOLOZU_BIN[@]}" doctor --proof --output "$DOCTOR_JSON" --proof-dir "$PROOF_DIR"

DATASET="$("$PY_BIN" - "$PROOF_REPORT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["artifacts"]["dataset"])
PY
)"
PREDICTIONS="$("$PY_BIN" - "$PROOF_REPORT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["artifacts"]["predictions"])
PY
)"

echo "[2/5] demo instance-seg"
rm -rf "$DEMO_DIR"
"${YOLOZU_BIN[@]}" demo instance-seg \
  --num-images 2 \
  --max-instances 2 \
  --inference none \
  --run-dir "$DEMO_DIR"

echo "[3/5] validate proof artifacts"
"${YOLOZU_BIN[@]}" validate dataset "$DATASET" --split "$SPLIT" --strict
"${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict

echo "[4/5] eval proof predictions"
"${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$EVAL_REPORT"

echo "[5/5] verify DoD artifacts"
"$PY_BIN" - "$PROOF_REPORT" "$DEMO_DIR" "$EVAL_REPORT" "$DOD_REPORT" <<'PY'
import json
import sys
from pathlib import Path

proof_report = Path(sys.argv[1])
demo_dir = Path(sys.argv[2])
eval_report = Path(sys.argv[3])
dod_report = Path(sys.argv[4])

proof = json.loads(proof_report.read_text(encoding="utf-8"))
if proof.get("status") != "pass":
    raise SystemExit(f"doctor proof did not pass: {proof.get('status')}")

demo_report = demo_dir / "instance_seg_demo_report.json"
if not demo_report.is_file():
    raise SystemExit(f"missing demo report: {demo_report}")
demo = json.loads(demo_report.read_text(encoding="utf-8"))
overlay_dir = demo.get("artifacts", {}).get("overlays_dir")
if not isinstance(overlay_dir, str) or not sorted(Path(overlay_dir).glob("*.png")):
    raise SystemExit(f"missing demo overlay PNG under: {overlay_dir}")

eval_payload = json.loads(eval_report.read_text(encoding="utf-8"))
if eval_payload.get("dry_run") is not True:
    raise SystemExit("DoD eval report must be dry_run=true")
if not isinstance(eval_payload.get("counts"), dict):
    raise SystemExit("DoD eval report missing counts object")

summary = {
    "kind": "yolozu_dod_cpu_smoke",
    "schema_version": 1,
    "status": "pass",
    "artifacts": {
        "doctor_proof_report": str(proof_report),
        "demo_report": str(demo_report),
        "eval_report": str(eval_report),
    },
    "proof_metrics": proof.get("observed_metrics"),
}
dod_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(dod_report)
PY

echo "DoD CPU smoke OK: $DOD_REPORT"
