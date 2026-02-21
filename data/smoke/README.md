# Smoke assets (offline, repo-bundled)

This directory is a **network-free minimal dataset** used by the project smoke flow.

## What is included

- `images/val/*.jpg` — 10 sample images
- `labels/val/*.txt` — YOLO bbox labels (`class cx cy w h`, normalized)
- `labels/val/classes.json` — class/category mapping helper
- `predictions/predictions_dummy.json` — fixed predictions artifact (`schema_version: 1`)

## What is guaranteed

The following commands are expected to pass from repo root:

```bash
python3 -m yolozu.cli validate dataset data/smoke
python3 -m yolozu.cli validate predictions data/smoke/predictions/predictions_dummy.json --strict
python3 -m yolozu.cli eval-coco \
	--dataset data/smoke \
	--split val \
	--predictions data/smoke/predictions/predictions_dummy.json \
	--dry-run \
	--output reports/smoke_coco_eval_dry_run.json
```

One-command equivalent:

```bash
bash scripts/smoke.sh
```

## Provenance and copyright/license note

- Quick legal note (source/license/modification): these files are derived from
	`data/coco128` (Ultralytics package metadata with GPL-3.0 notice), then subset/copied
	and converted into deterministic smoke predictions for validation use.

- These smoke assets are generated from local `data/coco128` via
	`python3 tools/generate_smoke_assets.py`.
- Image/license provenance follows the source subset under `data/coco128`.
	See `data/coco128/README.txt` and `data/coco128/LICENSE`.
- `predictions/predictions_dummy.json` is a generated artifact derived from
	YOLO labels with fixed scores for deterministic smoke validation.

If you need strictly self-authored/CC0-only media, replace `images/val` +
`labels/val` with your own assets and regenerate predictions accordingly.

Debug tip (pretty-print one-line JSON):

```bash
python3 tools/format_json.py \
	--input data/smoke/predictions/predictions_dummy.json \
	--output reports/predictions_dummy.pretty.json
```
