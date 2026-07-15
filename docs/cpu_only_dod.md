# CPU-only DoD smoke path

This page pins the public stable-lane DoD:

```bash
python3 -m pip install -U yolozu
yolozu doctor --proof --output reports/dod_cpu_smoke/doctor.json --proof-dir reports/dod_cpu_smoke/doctor_proof
yolozu demo instance-seg --inference none --run-dir reports/dod_cpu_smoke/demo_instance_seg
```

Use the proof artifacts emitted by `doctor --proof` for validation and evaluation:

```bash
PROOF=reports/dod_cpu_smoke/doctor_proof/proof_report.json
DATASET=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["dataset"])' "$PROOF")
PREDICTIONS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["predictions"])' "$PROOF")

yolozu validate dataset "$DATASET" --split val2017 --strict
yolozu validate predictions "$PREDICTIONS" --strict
yolozu eval-coco --dataset "$DATASET" --split val2017 --predictions "$PREDICTIONS" --dry-run --output reports/dod_cpu_smoke/eval_coco_dry_run.json
```

Repo checkout preflight:

```bash
bash scripts/dod_cpu_smoke.sh
```

The script writes `reports/dod_cpu_smoke/dod_cpu_smoke_report.json` and verifies the proof report, demo report/overlays, validation path, and evaluation report without requiring network, GPU, or training.
It also records the exact command, exit code, elapsed time, and combined output log for each step.

## Public PyPI fresh-install evidence

The repository includes a separate harness that creates a new virtual environment and explicitly installs from `https://pypi.org/simple` before running the same stable lane outside the source checkout:

```bash
bash scripts/fresh_install_journey.sh \
  --python python3 \
  --package yolozu \
  --run-dir reports/fresh_install_journey
```

Use `--package yolozu==VERSION` to pin a published release. The run directory must not already exist so a reused environment cannot be mistaken for fresh-install evidence.

The top-level `fresh_install_journey_report.json` records the requested and resolved package versions, interpreter and operating-system details, exact commands, exit codes, elapsed times, and log paths. Its nested `dod/dod_cpu_smoke_report.json` records the individual `doctor`, demo, validation, and evaluation steps.

The `Public PyPI fresh-install journey` GitHub Actions workflow runs this harness on Linux and macOS for Python 3.10 through 3.14. This matrix covers the current CPython releases accepted by the public `Python >=3.10` package metadata; it does not claim compatibility with unreleased Python versions. Workflow artifacts are retained for 14 days, including failures that produce a partial report.
