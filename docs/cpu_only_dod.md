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
