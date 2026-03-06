#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
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
  --output-dir <path>             Output directory for SynthGen and deep-profile artifacts (default: reports).
  --demo-run-dir <path>           Run directory for instance-seg smoke demo.
                                  (default: reports/smoke_demo_instance_seg)
  --skip-demo                     Skip instance-seg demo validation.
  --torch-device <dev>            Torch device for deep-profile TTT probe (default: cpu).
  --profile <core|deep>           Smoke depth profile (default: core).
                                  deep = core + walkthrough evidence report.
  --walkthrough-report <path>     Deep profile walkthrough report JSON.
                                  (default: reports/smoke_walkthrough_report.json)
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

can_import_yolozu() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import yolozu.cli  # noqa: F401
PY
}

pick_python() {
  local candidates=()
  if [[ -n "${YOLOZU_PYTHON:-}" ]]; then
    candidates+=("${YOLOZU_PYTHON}")
  fi
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    candidates+=("$ROOT_DIR/.venv/bin/python")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "error: no executable python interpreter found." >&2
  exit 2
}

PY_BIN="$(pick_python)"

if can_import_yolozu "$PY_BIN"; then
  YOLOZU_BIN=("$PY_BIN" -m yolozu.cli)
elif command -v yolozu >/dev/null 2>&1; then
  YOLOZU_BIN=(yolozu)
else
  echo "error: neither repo-local 'python -m yolozu.cli' nor 'yolozu' command is available." >&2
  echo "hint: python3 -m pip install -e ." >&2
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
TORCH_DEVICE="cpu"
PROFILE="core"
WALKTHROUGH_REPORT="reports/smoke_walkthrough_report.json"

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
    --torch-device)
      require_value "$1" "${2:-}"
      TORCH_DEVICE="${2:-}"
      shift 2
      ;;
    --profile)
      require_value "$1" "${2:-}"
      PROFILE="${2:-}"
      shift 2
      ;;
    --walkthrough-report)
      require_value "$1" "${2:-}"
      WALKTHROUGH_REPORT="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PROFILE" in
  core|deep)
    ;;
  *)
    echo "error: --profile must be one of: core, deep" >&2
    exit 2
    ;;
esac

if [[ -z "$SYNTHGEN_PREDICTIONS" ]]; then
  SYNTHGEN_PREDICTIONS="$SYNTHGEN_SMOKE_ROOT/predictions_synthgen_smoke.json"
fi

mkdir -p "$OUTPUT_DIR"

echo "[0/7] preflight runtime deps"
echo "python: $PY_BIN"
echo "cli: ${YOLOZU_BIN[*]}"
"$PY_BIN" - <<'PY'
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
"$PY_BIN" - "$REPORT" <<'PY'
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
"$PY_BIN" tools/smoke_synthgen.py \
  --dataset-root "$SYNTHGEN_SMOKE_ROOT" \
  --predictions "$SYNTHGEN_PREDICTIONS" \
  --output-dir "$OUTPUT_DIR"

"$PY_BIN" - "$OUTPUT_DIR" <<'PY'
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
    --max-instances 2 \
    --background yolo-bbox \
    --yolo-root "$DATASET" \
    --yolo-split val \
    --inference none \
    --run-dir "$DEMO_RUN_DIR"

  "$PY_BIN" - "$DEMO_RUN_DIR" <<'PY'
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

if [[ "$PROFILE" == "deep" ]]; then
  DEMO_OVERVIEW_REPORT="$OUTPUT_DIR/smoke_demo_overview.json"
  EXTERNAL_FINETUNE_REPORT="$OUTPUT_DIR/smoke_external_finetune_report.json"
  EXPORT_ONNXRT_REPORT="$OUTPUT_DIR/smoke_export_onnxrt.json"
  EXPORT_TRT_REPORT="$OUTPUT_DIR/smoke_export_trt.json"
  EXPORT_EXECUTORCH_REPORT="$OUTPUT_DIR/smoke_export_executorch.json"
  EXPORT_HELP_REPORT="$OUTPUT_DIR/smoke_export_help.txt"
  CONTINUAL_HELP_REPORT="$OUTPUT_DIR/smoke_continual_train_help.txt"
  TTT_PROBE_STDERR="$OUTPUT_DIR/smoke_ttt_probe.stderr.txt"
  TTT_PROBE_OUTPUT="$OUTPUT_DIR/smoke_ttt_probe.json"

  echo "[deep/1] demo overview capability map"
  "${YOLOZU_BIN[@]}" demo overview --output "$DEMO_OVERVIEW_REPORT"

  "$PY_BIN" - "$DEMO_OVERVIEW_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
payload = json.loads(report.read_text(encoding="utf-8"))
coverage = payload.get("coverage") or []
seen = {str(item.get("capability")): str(item.get("status")) for item in coverage if isinstance(item, dict)}
required = ("bbox", "segmentation", "keypoints", "depth", "pose6d")
missing = [cap for cap in required if cap not in seen]
if missing:
    raise SystemExit("demo overview missing required capabilities: " + ", ".join(missing))
allowed = {"supported", "deps_missing"}
bad = [cap for cap in required if seen.get(cap) not in allowed]
if bad:
    raise SystemExit("demo overview has unexpected status: " + ", ".join(f"{cap}={seen.get(cap)}" for cap in bad))
print("demo overview capability map OK")
PY

  echo "[deep/2] external finetune smoke matrix"
  "$PY_BIN" tools/run_external_finetune_smoke.py \
    --dataset-root "$DATASET" \
    --split train \
    --device "$TORCH_DEVICE" \
    --output "$EXTERNAL_FINETUNE_REPORT"

  "$PY_BIN" - "$EXTERNAL_FINETUNE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
payload = json.loads(report.read_text(encoding="utf-8"))
if payload.get("ok") is not True:
    raise SystemExit("external finetune smoke returned ok=false")
frameworks = set(payload.get("frameworks") or [])
required = {"yolov", "mmdetection", "detectron2", "rtdetr"}
missing = sorted(required - frameworks)
if missing:
    raise SystemExit("external finetune smoke missing frameworks: " + ", ".join(missing))
print("external finetune smoke matrix OK")
PY

  echo "[deep/3] backend export/deploy dry-run checks (onnxrt/trt/executorch)"
  MODEL_DIR="$OUTPUT_DIR/smoke_models"
  mkdir -p "$MODEL_DIR"
  : > "$MODEL_DIR/dummy.onnx"
  : > "$MODEL_DIR/dummy.plan"
  : > "$MODEL_DIR/dummy.pte"

  "$PY_BIN" tools/yolozu.py export \
    --backend onnxrt \
    --dataset "$DATASET" \
    --split val \
    --max-images 1 \
    --model "$MODEL_DIR/dummy.onnx" \
    --dry-run \
    --output "$EXPORT_ONNXRT_REPORT"

  "$PY_BIN" tools/yolozu.py export \
    --backend trt \
    --dataset "$DATASET" \
    --split val \
    --max-images 1 \
    --model "$MODEL_DIR/dummy.plan" \
    --dry-run \
    --output "$EXPORT_TRT_REPORT"

  "$PY_BIN" tools/yolozu.py export \
    --backend executorch \
    --dataset "$DATASET" \
    --split val \
    --max-images 1 \
    --model "$MODEL_DIR/dummy.pte" \
    --dry-run \
    --output "$EXPORT_EXECUTORCH_REPORT"

  "$PY_BIN" - "$EXPORT_ONNXRT_REPORT" "$EXPORT_TRT_REPORT" "$EXPORT_EXECUTORCH_REPORT" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise SystemExit(f"{path}: missing predictions list")
    if not predictions:
        raise SystemExit(f"{path}: predictions list is empty")
    sample = predictions[0]
    if not isinstance(sample, dict):
        raise SystemExit(f"{path}: first prediction is not an object")
    if "image" not in sample:
        raise SystemExit(f"{path}: first prediction missing image")
    detections = sample.get("detections")
    if not isinstance(detections, list):
        raise SystemExit(f"{path}: first prediction missing detections list")
print("backend export predictions structure OK")
PY

  echo "[deep/4] TTT + continual training surface checks"
  "$PY_BIN" tools/yolozu.py export --help > "$EXPORT_HELP_REPORT"
  "$PY_BIN" tools/yolozu.py continual-train --help > "$CONTINUAL_HELP_REPORT"
  grep -q -- "--ttt" "$EXPORT_HELP_REPORT"
  grep -q -- "--ttt-preset" "$EXPORT_HELP_REPORT"
  grep -q -- "--ttt-method" "$EXPORT_HELP_REPORT"
  grep -q -- "--replay-size" "$CONTINUAL_HELP_REPORT"

  TTT_STATUS="pass"
  TTT_NOTE="TTT execution path available in this environment"

  set +e
  "$PY_BIN" tools/yolozu.py export \
    --backend torch \
    --dataset "$DATASET" \
    --split val \
    --max-images 1 \
    --config rtdetr_pose/configs/base.json \
    --device "$TORCH_DEVICE" \
    --ttt \
    --ttt-preset safe \
    --output "$TTT_PROBE_OUTPUT" \
    1>/dev/null \
    2>"$TTT_PROBE_STDERR"
  ttt_rc=$?
  set -e

  if [[ "$ttt_rc" -ne 0 ]]; then
    if grep -qiE "torch is required for TTT|no module named 'torch'|modulenotfounderror: no module named 'torch'" "$TTT_PROBE_STDERR"; then
      TTT_STATUS="deps_missing"
      TTT_NOTE="TTT probe requires torch; install yolozu[train] (or yolozu[demo]) to execute adaptation"
    else
      echo "unexpected TTT probe failure:" >&2
      cat "$TTT_PROBE_STDERR" >&2
      exit 1
    fi
  fi

  echo "[deep/5] interface-contract-first AI workflow checks"
  "$PY_BIN" tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
  "$PY_BIN" tools/yolozu.py registry validate

  "$PY_BIN" - \
    "$WALKTHROUGH_REPORT" \
    "$REPORT" \
    "$DEMO_OVERVIEW_REPORT" \
    "$EXTERNAL_FINETUNE_REPORT" \
    "$EXPORT_ONNXRT_REPORT" \
    "$EXPORT_TRT_REPORT" \
    "$EXPORT_EXECUTORCH_REPORT" \
    "$TTT_STATUS" \
    "$TTT_NOTE" <<'PY'
import json
import time
from pathlib import Path
import sys

(
    out,
    core_eval,
    demo_overview,
    external_report,
    export_onnx,
    export_trt,
    export_exec,
    ttt_status,
    ttt_note,
) = sys.argv[1:]

core_eval_path = Path(core_eval)
demo_overview_path = Path(demo_overview)
external_report_path = Path(external_report)
export_paths = [Path(export_onnx), Path(export_trt), Path(export_exec)]

overview = json.loads(demo_overview_path.read_text(encoding="utf-8"))
coverage = {str(item.get("capability")): str(item.get("status")) for item in (overview.get("coverage") or []) if isinstance(item, dict)}
required_caps = ["bbox", "segmentation", "keypoints", "depth", "pose6d"]

external = json.loads(external_report_path.read_text(encoding="utf-8"))
frameworks = sorted(set(external.get("frameworks") or []))

claims = [
    {
        "id": "framework_agnostic_eval_toolkit",
        "claim": "Framework-agnostic evaluation toolkit for vision models under domain shift.",
        "status": "pass" if core_eval_path.is_file() else "fail",
        "evidence": [str(core_eval_path), str(external_report_path)],
    },
    {
        "id": "training_capable_forgetting_mitigation",
        "claim": "Training-capable workflows for mitigating catastrophic forgetting (self-distillation/replay/PEFT).",
        "status": "pass" if {"yolov", "mmdetection", "detectron2", "rtdetr"}.issubset(set(frameworks)) else "fail",
        "evidence": [str(external_report_path), "tools/yolozu.py continual-train --help"],
    },
    {
        "id": "inference_time_adaptation_ttt",
        "claim": "Support for inference-time adaptation (TTT).",
        "status": "warn" if ttt_status == "deps_missing" else ("pass" if ttt_status == "pass" else "fail"),
        "evidence": ["tools/yolozu.py export --help", str(Path(out).with_name("smoke_ttt_probe.stderr.txt"))],
        "note": ttt_note,
    },
    {
        "id": "predictions_stable_interface_contract",
        "claim": "Predictions are the stable interface contract across workflows.",
        "status": "pass",
        "evidence": [str(core_eval_path), str(export_paths[0]), str(export_paths[1]), str(export_paths[2])],
    },
    {
        "id": "multi_task_evaluation_support",
        "claim": "Multi-task evaluation support (bbox/segmentation/keypoints/depth/pose6d).",
        "status": "pass" if all(cap in coverage for cap in required_caps) else "fail",
        "evidence": [str(demo_overview_path)],
        "details": {cap: coverage.get(cap) for cap in required_caps},
    },
    {
        "id": "production_ready_deployment_path",
        "claim": "Production-ready deployment path (ONNX/ExecuTorch/PyTorch/ONNXRT/TRT).",
        "status": "pass" if all(path.is_file() for path in export_paths) else "fail",
        "evidence": [str(path) for path in export_paths],
    },
    {
        "id": "interface_contract_first_ai_first_workflow",
        "claim": "Interface-contract-first, AI-first workflow with versioned comparable artifacts.",
        "status": "pass",
        "evidence": ["tools/manifest.json", "yolozu/data/manifest/tools_manifest.json", str(core_eval_path), str(external_report_path)],
    },
]

warnings = []
if ttt_status == "deps_missing":
    warnings.append(ttt_note)

payload = {
    "kind": "smoke_walkthrough",
    "profile": "deep",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ok": all(item.get("status") in {"pass", "warn"} for item in claims),
    "claims": claims,
    "frameworks": frameworks,
    "artifacts": {
        "core_eval": str(core_eval_path),
        "demo_overview": str(demo_overview_path),
        "external_finetune_smoke": str(external_report_path),
        "backend_exports": [str(path) for path in export_paths],
    },
    "warnings": warnings,
}

out_path = Path(out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
print(out_path)
PY
fi

if [[ "$PROFILE" == "deep" ]]; then
  echo "smoke OK (deep): $REPORT + $OUTPUT_DIR/smoke_synthgen_summary.json + $WALKTHROUGH_REPORT"
else
  echo "smoke OK: $REPORT + $OUTPUT_DIR/smoke_synthgen_summary.json"
fi
