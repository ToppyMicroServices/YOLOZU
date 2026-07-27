# Dataset preflight qualification — 2026-07-27

## Scope

This bounded qualification covers the dataset readiness contradictions recorded
in `YOLOZU-ll2.67`. It does not promote dataset I/O to a standalone Stable
capability.

- Source base: `f79183753c2eb1df412ef9776637439f4938b659`
- Candidate wheel: `yolozu-4.6.0-py3-none-any.whl`
- Candidate wheel SHA-256:
  `22eac9be348d7d00bea09faea5f255af99605676be9577bb201f763576ad7045`
- Fresh installed interpreter: CPython 3.12
- Platform: macOS arm64

## Policy

- A selected split with zero records is an error, including in warning mode.
- Direct training readiness requires strict validation of the inspected records:
  readable image files, valid label schema, and normalized bbox geometry.
- Stored `cx`, `cy`, `w`, and `h` values remain bounded by `[0,1]`.
- A tolerance of `1e-6` applies only to floating-point round-off in bbox edges
  derived from those stored values. No label is clipped or rewritten.
- `doctor train-dataset --max-images N` reports `scope=first_n` when fewer than
  all records are inspected. The default cap is 200.
- Explicit COCO preflight requires `--instances` and `--images-dir` together;
  it does not require `--dataset`.

## Reproduction

Source checkout:

```bash
.venv/bin/python tools/validate_dataset.py --dataset data/coco128 --split train2017
.venv/bin/python -m yolozu validate dataset data/coco128 --split train2017 --strict
.venv/bin/python -m yolozu doctor train-dataset --dataset data/coco128 --split train2017 --output -
```

Observed: all 128 records passed; the doctor reported
`direct_train_ready=true`, `scope=all`, `checked=128`, and zero errors. The
largest derived edge overrun in the source labels was
`5.00000000069889e-7`, below the recorded tolerance.

Missing split:

```bash
.venv/bin/python tools/validate_dataset.py --dataset data/coco128 --split val2017
.venv/bin/python -m yolozu validate dataset data/coco128 --split val2017 --strict
```

Observed: both commands exited 1 with `dataset contains no records`.

Fresh candidate-wheel install, run from `/tmp` outside the checkout:

```bash
/tmp/yolozu_dataset_preflight_venv/bin/yolozu validate dataset \
  /Users/akira/YOLOZU/data/coco128 --split train2017 --strict
/tmp/yolozu_dataset_preflight_venv/bin/yolozu validate dataset \
  /Users/akira/YOLOZU/data/coco128 --split val2017 --strict
/tmp/yolozu_dataset_preflight_venv/bin/yolozu doctor train-dataset \
  --from coco-instances \
  --instances /Users/akira/YOLOZU/data/conversion_tiny_coco/annotations/instances_val2017.json \
  --images-dir /Users/akira/YOLOZU/data/conversion_tiny_coco/images/val2017 \
  --split val2017 --output -
```

Observed:

- `train2017` validation exited 0.
- Missing `val2017` validation exited 1 with the same empty-dataset reason.
- Explicit COCO preflight checked 2/2 records, reported zero validation errors,
  `direct_train_ready=false`, and `train_ready_after_migration=true`.
- The COCO annotation fixture SHA-256 was
  `242652132b66a7945f95a5d71bc545055e2daa9e7eaf808810d89601ba6054ee`.

## Automated checks

```text
python -m unittest tests.test_dataset_validator tests.test_doctor_import_cli
Ran 28 tests ... OK

python tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
OK

python -m unittest tests.test_packaged_tools_manifest tests.test_manifest_docs_references
Ran 7 tests ... OK
```

The regression set covers empty datasets in fail and warn modes, the bbox edge
tolerance boundary, strict doctor rejection, explicit COCO arguments, source
CLI behavior, and a freshly installed candidate wheel.
