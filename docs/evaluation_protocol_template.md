# Evaluation Protocol Template

Use this template when adding a benchmark, release comparison, or customer-facing
evaluation recipe. Keep it next to the protocol JSON or report bundle that uses
it.

## Purpose

- Question being answered:
- Task:
- Dataset and split:
- Primary metric:
- Decision that depends on the result:

## Fixed Inputs

| Input | Required value | Evidence |
|---|---|---|
| Dataset version/hash |  |  |
| Class map |  |  |
| Prediction artifact |  |  |
| Evaluation protocol JSON |  |  |
| Runtime environment |  |  |

## Commands

```bash
python3 -m yolozu eval-coco \
  --dataset <dataset> \
  --split <split> \
  --predictions <predictions.json> \
  --output <report.json>
```

`eval-coco` includes strict predictions validation. Run the standalone
`validate dataset` or `validate predictions` commands only when a separate
preflight artifact is useful.

## Expected Artifacts

| Artifact | Purpose | Required check |
|---|---|---|
| `predictions.json` | Predictions interface contract input | strict validation passes |
| `report.json` | Evaluation result | schema validation passes |
| `protocol.json` | Fixed settings | protocol hash recorded in report |
| `README.md` or notes | Human summary | states limits and skipped lanes |

## Report Reading Order

1. Confirm the protocol hash and dataset/split.
2. Confirm skipped lanes and skip reasons.
3. Read the primary metric and confidence/coverage notes.
4. Compare only against reports with the same fixed inputs.

## Promotion Gate

| Gate | Pass condition | Owner |
|---|---|---|
| Schema validation | All required artifacts validate |  |
| Reproducibility | Commands rerun on CPU or documented target hardware |  |
| Comparison fairness | Dataset, split, class map, and protocol hash match |  |
| Risk note | Known skips and limitations are listed |  |

## Notes

- Do not compare reports when protocol hashes differ.
- Do not hide skipped lanes; mark them as skipped with a concrete reason.
- If research processing is applied after evaluation, report it as a separate
  research-lane artifact.
