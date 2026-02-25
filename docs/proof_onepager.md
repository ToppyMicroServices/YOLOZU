# YOLOZU proof (one page)

This is a minimal, concrete “does it work?” artifact: **shortest path + expected outputs**.

## Shortest path (pip)

```bash
python3 -m pip install yolozu
yolozu doctor --output -
yolozu demo instance-seg --num-images 2 --image-size 96
```

What you should see:
- `yolozu doctor` prints environment diagnostics (Python, optional backends, resource paths).
- `yolozu demo` runs a small demo suite and writes self-contained outputs under demo_output/ (JSON + overlays).

## If you already have predictions.json

```bash
yolozu validate predictions --predictions predictions.json --strict
yolozu eval-coco --dataset /path/to/coco --split val2017 --predictions predictions.json --output reports/coco_eval.json
```

## Evidence table (inputs → validators → reports)

| Step | Input | Output | What it proves |
|---|---|---|---|
| `doctor` | none | stdout JSON/text | pip install is usable; packaged resources are visible |
| `validate predictions` | `predictions.json` | exit code + warnings | schema/contract is enforced |
| `eval-coco` | dataset + predictions | `reports/coco_eval.json` | metrics pipeline runs deterministically |

## Report example (shape)

A typical `reports/coco_eval.json` is a JSON object containing at least:

```json
{
  "summary": {
    "map50": 0.0,
    "map50_95": 0.0
  },
  "meta": {
    "protocol": null,
    "split": "val2017"
  }
}
```

Exact fields can vary by task/evaluator, but the contract is: **machine-readable metrics + metadata**.
