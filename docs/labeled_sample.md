# Try a reusable labeled dataset

[日本語の手順](labeled_sample_ja.md)

Start here if you want a first evaluation report before preparing your own
dataset. The commands below generate the inputs, validate them, and run
COCOeval. You can then apply the same steps to predictions from your own model.

`yolozu demo dataset` creates eight visible 320×240 synthetic images with YOLO
bounding-box labels: four training images and four validation images. Classes
are `0 = circle` and `1 = rectangle`. The last validation image is deliberately
empty. No model, GPU, image download, or external dataset is needed.

The generator is available from YOLOZU 4.8.0. Older packages do not include the
command, but the generated files can be copied to another environment without
installing the generator there.

## Generate, inspect, evaluate

Requires Python 3.10 or newer. In a new working directory on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'yolozu[coco]==4.11.0'
```

On Windows PowerShell, use `py -m venv .venv` and
`.\.venv\Scripts\Activate.ps1`, then run the commands below with `python`
instead of `python3`. See [installation](install.md) if setup fails.

```bash
python3 -m yolozu demo dataset --run-dir reports/labeled_sample --seed 0
python3 -m yolozu validate dataset reports/labeled_sample --split train --strict
python3 -m yolozu validate dataset reports/labeled_sample --split val --strict
python3 -m yolozu validate predictions reports/labeled_sample/predictions.json --strict
python3 -m yolozu eval-coco --dataset reports/labeled_sample --split val \
  --predictions reports/labeled_sample/predictions.json \
  --output reports/labeled_sample_coco.json
```

Generation needs only the core dependencies. Official COCO evaluation also needs
the `coco` extra. Choose a new `--run-dir` on each generation; existing directories,
files, and symlinks are refused without overwriting their contents.
For a source checkout before publication, install `'.[coco]'` instead.

Open `labeled_preview.png` in the generated directory to inspect every image and
label. The original images have no annotation overlay burned into them.

```text
labeled_sample/
  images/{train,val}/*.png
  labels/{train,val}/*.txt
  labels/{train,val}/classes.json
  data.yaml
  predictions.json
  labeled_preview.png
  sample_manifest.json
  README.md
```

Each label row is `class_id cx cy w h`, with coordinates normalized to `[0, 1]`.
An empty label file means a negative image, not a missing annotation file.
`predictions.json` contains validation predictions derived directly from these
labels. Its expected COCO AP is approximately `1.0`; this tests data handling and
evaluation, **not model quality**.

## Check the first result

Open `reports/labeled_sample/labeled_preview.png` and
`reports/labeled_sample_coco.json`. To print the report in the terminal:

```bash
python3 -m json.tool reports/labeled_sample_coco.json
```

For the unchanged sample with seed `0`, expect `dry_run: false`,
`counts.images: 4`, `counts.detections: 6`, and `metrics.map50_95` and
`metrics.map50` approximately `1.0`. The empty validation image is included.
These are real COCOeval calculations on known synthetic predictions.

If `pycocotools` is missing, install the `coco` extra in the active environment.
If the sample directory already exists, choose a new `--run-dir` and update the
paths in the remaining commands. A `--dry-run` report has null metrics and does
not complete this evaluation path.

## Try a release gate

With YOLOZU 4.11.0, the same sample can demonstrate a threshold and a portable
JSON evidence pack:

```bash
python3 -m yolozu qualify-release create \
  --dataset reports/labeled_sample --split val \
  --predictions reports/labeled_sample/predictions.json \
  --min-map50-95 0.99 --output-dir reports/labeled_sample_qualification
python3 -m yolozu qualify-release verify reports/labeled_sample_qualification
```

The unchanged sample should produce `decision: pass`; verification should
produce `ok: true`. The `0.99` threshold is a fixture check, not a recommended
production threshold. For your model, select thresholds and any baseline before
evaluation. See [release qualification](release_qualification.md) for the YAML
spec, regression checks, and reusable GitHub Action.

## Copy it or use your own data

Copy the whole directory, including `classes.json` and `data.yaml`. Stored image
paths are relative, so YOLOZU can validate and evaluate the copy from another
working directory. Pass the new absolute directory path to `--dataset` and the
new prediction file path to `--predictions`. The compatibility check also tests
a destination containing spaces.

Keep the layout for your own images and labels, and replace the class names and
mapping consistently. Replace the known predictions with actual model outputs
before interpreting scores as accuracy. Do not keep the sample manifest as
provenance for modified data: its hashes and geometry describe only the original
fixture. Existing datasets do not need to be copied into this layout; see
[migration helpers](migrate.md) and [bring-your-own-predictions](byop_quickstarts.md).

`data.yaml` uses `path: .`, resolved relative to the YAML file by YOLOZU. Other
tools may use a different base; set `path` to the absolute dataset directory if
the receiving tool requires it. External training runtimes are not qualified by
this sample's evaluation check.

## Keep upgrades reproducible

Keep `sample_manifest.json` with the unchanged sample. It records the recipe and
seed, exact label geometry, file SHA-256 values, and generator/Pillow versions.
The same recipe and seed preserve coordinates; PNG encoding and preview fonts
may differ between Pillow releases.

The predictions interface contract has separate version numbers: wrapper `1`,
entries `2`. Missing legacy entry versions can be migrated with
`yolozu predictions migrate`; unsupported explicit versions and explicit `null`
are rejected. An ambiguous image basename is rejected during bbox COCO
evaluation; use the exact dataset image path to disambiguate it.

See [version compatibility](versions.md) for the candidate-wheel matrix and its
limits. A declared dependency floor or a passing synthetic fixture is not a
claim that every newer dependency, trained model, or GPU works.

## Questions or a first-run failure?

Use the [first-run failure form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=first_run_failure.yml)
or [evaluation question form](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=evaluation_question.yml).
The forms let you choose whether your feedback may be counted in aggregate.
Keep datasets, weights, private predictions, credentials, and personal details
out of public reports. See [support](support.md) for the intake and consent rules.
