# A 30-minute path to a checked report

Create all tutorial inputs locally, then validate and evaluate existing predictions. This is a synthetic proof of the tooling, not model-quality evidence.

[HTML version](https://www.toppymicros.com/yolozu/docs/start.html)

## 0–5 minutes: Install the evaluation extra in an isolated environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "yolozu[coco]"
mkdir -p yolozu-quickstart && cd yolozu-quickstart
```

Expected: The active environment contains YOLOZU and pycocotools. For a candidate build, install the candidate wheel plus pycocotools instead; the remaining commands are unchanged and run outside the source checkout.

## 5–10 minutes: Create every tutorial input locally

```bash
yolozu doctor --proof --output reports/quickstart/doctor.json --proof-dir reports/quickstart/proof
```

Expected: The proof writes reports/quickstart/proof/toy_dataset, known_predictions.json, eval_report.json, and proof_report.json without downloading a model or dataset.

## 10–18 minutes: Check both interface boundaries explicitly

```bash
yolozu validate dataset reports/quickstart/proof/toy_dataset --split val2017 --strict
yolozu validate predictions reports/quickstart/proof/known_predictions.json --strict
```

Expected: Both validators exit successfully. These paths are outputs of the preceding proof command, not repository fixtures or guessed user paths.

## 18–23 minutes: Run real COCOeval with concise flags

```bash
yolozu eval-coco -d reports/quickstart/proof/toy_dataset -p reports/quickstart/proof/known_predictions.json -s val2017 -o reports/quickstart/eval_coco.json
```

Expected: eval_coco.json has status ok, dry_run false, validation mode strict, repair disabled, and real COCO metrics. Evaluation is strict by default.

## 23–27 minutes: Inspect the real report

```bash
python -m json.tool reports/quickstart/eval_coco.json
```

Expected: Confirm status ok, dry_run false, validation mode strict, repair disabled, the evaluated image/detection counts, and non-null COCO metrics.

## 27–30 minutes: Read before comparing

Expected: Confirm task, dataset/split, prediction source, export settings, protocol identity, metric family, and any skipped lane before comparing two reports.

## Core-install fallback (not real evaluation)

Use this only when pycocotools is intentionally unavailable. It performs strict validation and conversion, writes dry_run true, and leaves all metrics null; do not present it as evaluation evidence.

```bash
yolozu eval-coco -d reports/quickstart/proof/toy_dataset -p reports/quickstart/proof/known_predictions.json -s val2017 --dry-run -o reports/quickstart/eval_coco_dry_run.json
```

## Use the same strict lane from Python

The stable yolozu.api surface returns typed results and machine-readable exceptions without launching a subprocess or changing the working directory.

```python
from pathlib import Path

from yolozu.api import evaluate_coco

workspace = Path.cwd().resolve()
result = evaluate_coco(
    dataset=workspace / "reports/quickstart/proof/toy_dataset",
    predictions=workspace / "reports/quickstart/proof/known_predictions.json",
    split="val2017",
)
print(result.to_dict()["metrics"])
```

Expected: The call uses the inputs created by doctor --proof, validates strictly by default, and returns a CocoEvaluationResult. Pass dry_run=True only when metrics are intentionally unavailable.

[Stable Python API](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/python_api.md)

## Continue to the two-hour path

1. Complete the 30-minute stable path.
2. Export or import predictions from an external stack through a checked bridge.
3. Run a CPU-only evaluation with the same dataset, split, and fixed protocol.
4. Compare two wrapped prediction artifacts and their reports.
5. Fill out docs/evaluation_protocol_template.md for the run.
6. Choose stable evaluation, backend parity, or a Research lane from evidence—not from a missing feature assumption.

[Evaluation protocol template](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/evaluation_protocol_template.md)

[CPU-only procedure](https://raw.githubusercontent.com/ToppyMicroServices/YOLOZU/main/docs/cpu_only_dod.md)

[Examples](https://www.toppymicros.com/yolozu/docs/examples.html) · [Troubleshooting](https://www.toppymicros.com/yolozu/docs/troubleshooting.html) · [Agent guide](https://www.toppymicros.com/yolozu/docs/agents.md)
