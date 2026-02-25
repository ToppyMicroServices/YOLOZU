# YOLOZU proof (one page)

This is a minimal, concrete “does it work?” artifact shipped inside the `yolozu` wheel.

## Shortest path (pip)

```bash
python3 -m pip install yolozu
yolozu doctor --output -
yolozu demo instance-seg
```

## If you already have predictions.json

```bash
yolozu validate predictions --predictions predictions.json --strict
yolozu eval-coco --dataset /path/to/coco --split val2017 --predictions predictions.json --output reports/coco_eval.json
```

## Evidence table

| Step | Input | Output | What it proves |
|---|---|---|---|
| `doctor` | none | stdout | packaged resources + environment are OK |
| `validate predictions` | `predictions.json` | exit code + warnings | schema/contract enforcement |
| `eval-coco` | dataset + predictions | `reports/coco_eval.json` | metrics pipeline runs |

## Report example (shape)

```json
{
  "summary": {"map50": 0.0, "map50_95": 0.0},
  "meta": {"protocol": null, "split": "val2017"}
}
```
