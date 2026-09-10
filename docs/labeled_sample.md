# Try a reusable labeled dataset

`yolozu demo dataset` creates eight visible 320×240 synthetic images with YOLO
bounding-box labels: four training images and four validation images. Classes
are `0 = circle` and `1 = rectangle`. The last validation image is deliberately
empty. No model, GPU, image download, or external dataset is needed.

The generator is available from YOLOZU 4.8.0. Older packages do not include the
command, but the generated files can be copied to another environment without
installing the generator there.

## Generate, inspect, evaluate

From an activated virtual environment:

```bash
python3 -m pip install 'yolozu[coco]==4.8.0'
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
