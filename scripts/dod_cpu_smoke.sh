#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: bash scripts/dod_cpu_smoke.sh [options]

Runs the public CPU-only DoD path:
  doctor --proof -> demo -> validate dataset -> validate predictions -> eval-coco dry-run

Options:
  -h, --help              Show this help and exit.
  --run-dir <path>        Output directory for DoD artifacts (default: reports/dod_cpu_smoke).
  --split <name>          Split for proof dataset validation/eval (default: val2017).
  --installed-package     Run the installed yolozu package without adding the repo to PYTHONPATH.
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
INSTALLED_PACKAGE=false

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
    --installed-package)
      INSTALLED_PACKAGE=true
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PY_BIN="$(pick_python || true)"

if [[ -z "$PY_BIN" ]]; then
  echo "error: no runnable Python interpreter found." >&2
  echo "hint: python3 -m pip install -e . or python3 -m pip install -U yolozu" >&2
  exit 2
fi
PY_BIN="$("$PY_BIN" -c 'import sys; print(sys.executable)')"

if [[ "$INSTALLED_PACKAGE" == true ]]; then
  mkdir -p "$RUN_DIR"
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  cd "$RUN_DIR"
  unset PYTHONPATH
  YOLOZU_BIN=("$PY_BIN" -m yolozu)
  EXECUTION_MODE="installed_package"
else
  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  YOLOZU_BIN=("$PY_BIN" -m yolozu.cli)
  EXECUTION_MODE="repo_checkout"
fi

mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
STEPS_JSONL="$RUN_DIR/steps.jsonl"
mkdir -p "$LOG_DIR"
: > "$STEPS_JSONL"

run_step() {
  local step_name="$1"
  shift
  "$PY_BIN" - "$STEPS_JSONL" "$LOG_DIR/${step_name}.log" "$step_name" "$@" <<'PY'
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

timeline = Path(sys.argv[1])
log_path = Path(sys.argv[2])
name = sys.argv[3]
command = sys.argv[4:]
started = time.perf_counter()
proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
duration = time.perf_counter() - started
output = proc.stdout or ""
log_path.write_text(output, encoding="utf-8")
record = {
    "name": name,
    "command": command,
    "command_display": shlex.join(command),
    "duration_seconds": round(duration, 6),
    "exit_code": proc.returncode,
    "output_log": str(log_path),
}
with timeline.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
if output:
    print(output, end="")
raise SystemExit(proc.returncode)
PY
}

DOCTOR_JSON="$RUN_DIR/doctor.json"
PROOF_DIR="$RUN_DIR/doctor_proof"
PROOF_REPORT="$PROOF_DIR/proof_report.json"
DEMO_DIR="$RUN_DIR/demo_instance_seg"
EVAL_REPORT="$RUN_DIR/eval_coco_dry_run.json"
DOD_REPORT="$RUN_DIR/dod_cpu_smoke_report.json"

echo "[1/5] doctor --proof"
run_step doctor_proof "${YOLOZU_BIN[@]}" doctor --proof --output "$DOCTOR_JSON" --proof-dir "$PROOF_DIR"

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
run_step demo_instance_seg "${YOLOZU_BIN[@]}" demo instance-seg \
  --num-images 2 \
  --max-instances 2 \
  --inference none \
  --run-dir "$DEMO_DIR"

echo "[3/5] validate proof artifacts"
run_step validate_dataset "${YOLOZU_BIN[@]}" validate dataset "$DATASET" --split "$SPLIT" --strict
run_step validate_predictions "${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict

echo "[4/5] eval proof predictions"
run_step eval_coco "${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$EVAL_REPORT"

echo "[5/5] verify DoD artifacts"
"$PY_BIN" - "$DOCTOR_JSON" "$PROOF_REPORT" "$DEMO_DIR" "$EVAL_REPORT" "$DOD_REPORT" "$STEPS_JSONL" "$EXECUTION_MODE" <<'PY'
import json
import sys
from pathlib import Path

doctor_report = Path(sys.argv[1])
proof_report = Path(sys.argv[2])
demo_dir = Path(sys.argv[3])
eval_report = Path(sys.argv[4])
dod_report = Path(sys.argv[5])
steps_jsonl = Path(sys.argv[6])
execution_mode = sys.argv[7]

doctor = json.loads(doctor_report.read_text(encoding="utf-8"))
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

steps = [json.loads(line) for line in steps_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

summary = {
    "kind": "yolozu_dod_cpu_smoke",
    "schema_version": 1,
    "status": "pass",
    "execution": {
        "mode": execution_mode,
        "python": sys.version,
        "yolozu_version": doctor.get("yolozu", {}).get("version"),
        "elapsed_seconds": round(sum(float(step["duration_seconds"]) for step in steps), 6),
        "steps": steps,
    },
    "artifacts": {
        "doctor_proof_report": str(proof_report),
        "demo_report": str(demo_report),
        "eval_report": str(eval_report),
        "steps": str(steps_jsonl),
    },
    "proof_metrics": proof.get("observed_metrics"),
}
dod_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(dod_report)
PY

echo "DoD CPU smoke OK: $DOD_REPORT"
