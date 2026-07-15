#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: bash scripts/fresh_install_journey.sh [options]

Creates a clean virtual environment, installs the public PyPI package, and runs:
  doctor --proof -> demo -> validate dataset -> validate predictions -> eval-coco dry-run

Options:
  -h, --help              Show this help and exit.
  --python <executable>   Python used to create the clean environment (default: python3).
  --package <spec>        Package spec installed from PyPI (default: yolozu).
  --run-dir <path>        New directory for environment, logs, and reports.
                            (default: reports/fresh_install_journey)
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

PYTHON_BIN="python3"
PACKAGE_SPEC="yolozu"
RUN_DIR="reports/fresh_install_journey"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --package)
      require_value "$1" "${2:-}"
      PACKAGE_SPEC="${2:-}"
      shift 2
      ;;
    --run-dir)
      require_value "$1" "${2:-}"
      RUN_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

case "$PACKAGE_SPEC" in
  yolozu|yolozu==*) ;;
  *)
    echo "error: --package must be yolozu or an exact yolozu==VERSION spec" >&2
    exit 2
    ;;
esac

if [[ -e "$RUN_DIR" ]]; then
  echo "error: --run-dir must not already exist for a fresh-install journey: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
LOG_DIR="$RUN_DIR/logs"
STEPS_JSONL="$RUN_DIR/steps.jsonl"
REPORT_JSON="$RUN_DIR/fresh_install_journey_report.json"
VENV_DIR="$RUN_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
INSTALL_METADATA="$RUN_DIR/install_metadata.json"
: > "$STEPS_JSONL"

run_step() {
  local step_name="$1"
  shift
  "$PYTHON_BIN" - "$STEPS_JSONL" "$LOG_DIR/${step_name}.log" "$step_name" "$@" <<'PY'
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

finish() {
  local exit_code=$?
  trap - EXIT
  "$PYTHON_BIN" - "$REPORT_JSON" "$STEPS_JSONL" "$INSTALL_METADATA" "$RUN_DIR" "$PACKAGE_SPEC" "$exit_code" <<'PY'
import json
import platform
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
steps_path = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
run_dir = Path(sys.argv[4])
package_spec = sys.argv[5]
exit_code = int(sys.argv[6])
steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
install = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
dod_report = run_dir / "dod" / "dod_cpu_smoke_report.json"
payload = {
    "kind": "yolozu_fresh_install_journey",
    "schema_version": 1,
    "status": "pass" if exit_code == 0 else "fail",
    "requested_package": package_spec,
    "runner": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "bootstrap_python": sys.version,
    },
    "installed_package": install,
    "elapsed_seconds": round(sum(float(step["duration_seconds"]) for step in steps), 6),
    "steps": steps,
    "artifacts": {
        "dod_report": str(dod_report) if dod_report.is_file() else None,
        "logs_dir": str(run_dir / "logs"),
        "steps": str(steps_path),
    },
}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(report_path)
PY
  exit "$exit_code"
}
trap finish EXIT

run_step create_venv "$PYTHON_BIN" -m venv "$VENV_DIR"
run_step install_public_package "$VENV_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --index-url https://pypi.org/simple \
  "$PACKAGE_SPEC"

cd "$RUN_DIR"
"$VENV_PYTHON" - "$INSTALL_METADATA" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

distribution = importlib.metadata.distribution("yolozu")
location = Path(distribution.locate_file("")).resolve()
venv_root = Path(sys.prefix).resolve()
if location != venv_root and venv_root not in location.parents:
    raise SystemExit(f"installed distribution resolved outside the clean environment: {location}")
payload = {
    "name": "yolozu",
    "version": distribution.version,
    "location": str(location),
    "environment_root": str(venv_root),
    "python": sys.version,
    "system": platform.system(),
    "machine": platform.machine(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_step stable_lane_dod env -u PYTHONPATH YOLOZU_PYTHON="$VENV_PYTHON" bash "$ROOT_DIR/scripts/dod_cpu_smoke.sh" \
  --installed-package \
  --run-dir "$RUN_DIR/dod"
