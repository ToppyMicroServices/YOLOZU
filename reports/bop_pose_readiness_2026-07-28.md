# BOP T-LESS object 6DoF readiness report

Date: 2026-07-28

## Confirmed implementation

- The dataset downloader is pinned to the BOP Hugging Face namespace, accepts
  plain ZIP names only, validates every archive member before extraction, and
  records URL, bytes, SHA-256, extraction completion, and license provenance.
  Opt-in quota-smoke partial extraction is explicitly marked incomplete.
- The converter refuses unowned replacement, symlink outputs, protected paths,
  source-overlapping outputs, and existing splits during owned append.
- Deterministic frame partitions support a train/validation diagnostic.
- BOP object-to-camera rotation/translation and intrinsics are preserved.
- Available PLY model points are deterministically sampled, converted with the
  same millimetre-to-metre scale as translation, hashed, and attached for ADD
  and ADD-S evaluation. BOP symmetry metadata is retained when present.
- Both RunPod shell entrypoints implement `-h/--help`.

## Dataset and license

- T-LESS is listed as CC BY 4.0 by the
  [BOP dataset page](https://bop.felk.cvut.cz/datasets/) and the
  [official BOP T-LESS dataset card](https://huggingface.co/datasets/bop-benchmark/tless).
- Dataset terms are separate from YOLOZU's Apache-2.0 code license.

## Observed local evidence

The repository fixture exercises safe cached extraction, provenance output,
owned replacement/append, object-pose sidecars, CAD conversion, model hashes,
and refusal cases. It is a wiring and safety qualification only; it is not a
real T-LESS efficacy run.

Completed local verification:

- BOP safety/manifest/CAD tests: 8 passed.
- Required manifest gates: 7 passed.
- Full repository unit suite: 1,294 passed, 18 skipped.
- Manifest/help audit: 111 Python tools scanned, zero execution errors, zero
  missing declared flags.
- Generated web documentation: 120 tools and 25 schemas, no missing, extra, or
  stale generated files.

The 18 skips are existing environment/hardware-dependent skips. These checks do
not replace a real T-LESS model run.

## Not yet executed

- Full real `tless_base.zip` plus `tless_train_primesense.zip` acquisition.
- Baseline and trained checkpoint comparison on the preregistered subset.
- Three completed seeds reporting detection, rotation/translation, pose
  success, ADD, and ADD-S.
- GPU runtime/cost and failure-case collection.
- Independent reproduction from a release-addressable archive.

The provided frame split is a diagnostic frame holdout, not the official BOP
test benchmark. The object 6DoF lane remains Research and efficacy is
`not_established`. Human 3D skeleton pose is unsupported.

## Reproduction commands

```bash
bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh --help
bash deploy/runpod/run_rtdetr_pose_bop_tless_pose_eval.sh --help
python3 -m unittest tests.test_bop_pose_pipeline_safety
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
```

The full protocol and promotion conditions are defined in
[`docs/bop_tless_protocol.md`](../docs/bop_tless_protocol.md).
