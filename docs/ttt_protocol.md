# Test-Time Training protocol

TTT is an **opt-in research lane** and is **OFF by default**. Default validation,
prediction export, and evaluation do not adapt model parameters. A successful
TTT run proves that the selected implementation executed under its guards; it
does not establish efficacy or promote a method, preset, or checkpoint.

The canonical implementation and evidence status are summarized in
[TTA/TTT support matrix](tta_support_matrix.md). Machine-readable method
profiles live in `yolozu/tta/method_profiles.py`.

## Evidence boundary

| Method | YOLOZU profile | Implementation status | Fidelity | Efficacy |
|---|---|---|---|---|
| Tent | `yolozu_detector_entropy_v2` | Runnable, Research | Detector-adapted; not a reference-faithful Tent reproduction | Not established |
| MIM | `yolozu_structured_mim_v1` | Conditional, Research | Requires the structured masked-reconstruction hook | Not established |
| CoTTA | `yolozu_phase1_variant` | Runnable, Research | YOLOZU phase-1 variant; not reference-faithful CoTTA | Not established |
| EATA | `yolozu_phase1_variant` | Runnable, Research | YOLOZU phase-1 variant; not reference-faithful EATA | Not established |
| SAR | `yolozu_phase1_variant` | Runnable, Research | YOLOZU phase-1 variant; not reference-faithful SAR | Not established |

For detector outputs, entropy uses the final tensor axis as the class axis and
reduces over every non-class element, including queries. It does not select
foreground queries. A no-object class is included when it is present on the
final axis; otherwise its semantics are unidentified. Cross-view and
cross-step query correspondence is not established.

The figures below are evidence-boundary illustrations generated from the
validated source `docs/assets/ttt_method_results_source.json`; the checked-in
source is a synthetic fixture and contains no measured efficacy values.

- `docs/assets/ttt_method_results_summary.png`
- `docs/assets/ttt_compare_pipeline.png`
- `docs/assets/ttt_qualitative_shifted_probe.png`

## Short compare command

Use the wrapper when a checkpoint fully compatible with the boilerplate config
is available:

```bash
bash scripts/ttt_compare.sh \
  --method tent \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose.pt \
  --out reports/ttt_compare/tent \
  -n 1 \
  --no-eval
```

Equivalent Python entrypoint:

```bash
python3 tools/run_ttt_compare.py \
  --method tent \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose.pt \
  --out reports/ttt_compare/tent \
  -n 1 \
  --no-eval
```

`--dry-run` still validates the dataset, config, and checkpoint. It requires the
checkpoint loader's canonical `status=full` and `load.loaded=true`; arbitrary
bytes, partial checkpoints, and the noncanonical value `compatible` are
rejected. This prevents a plan from being presented as runnable when its model
cannot load.

The compare runner writes stage status atomically. On failure, `plan.json`
records the failed stage and stale success artifacts are removed. The final
report contains:

- `evidence_kind=local_diagnostic`
- `efficacy_conclusion=not_established`
- `promotion_eligible=false`
- the method profile and checkpoint/config provenance
- COCO metrics only when COCO evaluation actually ran
- `proxy_ap50` and `proxy_ap50_95` for the non-COCO fallback

Dataset identity uses a path-and-file-size metadata hash by default so a
full-dataset run does not reread every image before inference. Use
`--dataset-hash-mode content` when strict byte-level image provenance is needed
for a bounded diagnostic bundle. The multi-seed evidence command selects
`content` automatically for its capped clean/shift matrix.

See [TTT compare boilerplates](ttt_compare_boilerplates.md) for artifact names
and method-specific commands.

## Multi-seed evidence command

Use one command to run clean and deterministic-shift comparisons for all five
methods. The command requires at least three unique seeds and keeps sample-reset
and continual-stream results separate:

```bash
python3 tools/run_ttt_evidence_suite.py \
  -d data/coco128 \
  -x /path/to/shifted-coco128 \
  -c /path/to/base-checkpoint.pt \
  --mim-checkpoint /path/to/mim-checkpoint.pt \
  -o reports/ttt_evidence \
  -n 8 \
  --seeds 11,22,33
```

Each comparison records a deterministic TTT seed, checkpoint/config SHA-256,
dataset order/content SHA-256, real COCO AP when `pycocotools` is available,
calibration and collapse status, update ratio, subprocess latency, peak memory,
and forward/backward/optimizer counts. The suite fails if any child comparison
fails. It does not recursively delete the output directory.

The 2026-07-27 local run completed all 30 comparisons. It found no AP50:95
improvement and remains `efficacy_conclusion=not_established`; see
[the evidence report](../reports/ttt_evidence_2026-07-27.md). The complete
checkpoint and child-artifact bundle is published as the
[2026-07-27 diagnostic prerelease](https://github.com/ToppyMicroServices/YOLOZU/releases/tag/ttt-evidence-2026-07-27)
with archive SHA-256
`bb200d0c0a36447f0b6ed262a56ee09bef44ded8f10c55673243080fe1054068`.

## Local diagnostic demo

The packaged CLI also provides a self-contained diagnostic:

```bash
python3 -m yolozu demo ttt \
  --dataset-root data/smoke \
  --max-images 2 \
  --train-epochs 2 \
  --image-size 96 \
  --run-dir reports/demo_ttt_diagnostic \
  --force
```

The demo trains a tiny local checkpoint, applies a deterministic shift, and
compares TTT off/on. Its metrics are explicitly named `proxy_ap50` and
`proxy_ap50_95` with `metric_semantics=non_coco_proxy`. The report is
`kind=ttt_diagnostic_demo`, `evidence_kind=local_diagnostic`,
`promotion_eligible=false`, and `efficacy_conclusion=not_established`.
Positive or negative delta is only a local observation.

## Shared operator workflow

1. Freeze a dataset manifest, image order, split, seed, config, and checkpoint.
2. Confirm strict checkpoint preflight reports `status=full`.
3. Run the no-TTT baseline and adapted path with the same inputs.
4. Inspect `plan.json`, both predictions files, the TTT log, and the compare
   report.
5. Treat guard stops, empty updates, query changes, runtime, and proxy metrics as
   diagnostics.
6. Use an independently specified evaluation protocol and repeated runs before
   making any efficacy claim.

The built-in source tree does not contain a promotion-quality checkpoint bundle.
A current-compatible local diagnostic bundle was generated and published on
2026-07-27, but its short COCO128 training run did not improve AP and has not
been independently reproduced. Publication makes the exact inputs and outputs
addressable; it does not change the efficacy boundary. Historical ignored
`reports/ttt_improvement_probe` paths are not part of the SSOT and are not a
clean-checkout success path.
The dated boundary and remaining evidence requirements are recorded in
[`../reports/ttt_readiness_audit_2026-07-26.md`](../reports/ttt_readiness_audit_2026-07-26.md).

### Tent

Tent means entropy minimization at test time. YOLOZU implements a
detector-adapted entropy variant, not a reference-faithful reproduction.

**When to use:** controlled research diagnostics where a short, guarded update
on normalization parameters is useful to inspect.

**Concrete repo result:** execution, loss, update-norm, rollback, and prediction
delta reporting are implemented and tested. Efficacy is not established.

Each auxiliary consistency target is a detached eval-mode snapshot from the
same current image batch. Targets are not carried across batches. Structured or
non-floating auxiliary batches are rejected so pose/keypoint/depth/seg labels
cannot be transformed as images.

### MIM

MIM means Masked Image Modeling. YOLOZU requires a model-provided structured
masked-reconstruction hook; it is not a detector-logit entropy alias.

**When to use:** a controlled model/config pair that explicitly implements the
structured MIM hook and has a fully compatible checkpoint.

**Concrete repo result:** the hook, masking controls, loss accounting, and
fail-closed preflight are tested. No current checked-in measured bundle
establishes MIM efficacy.

```bash
bash scripts/ttt_compare.sh \
  --method mim \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose_mim.pt \
  --out reports/ttt_compare/mim \
  -n 1 \
  --no-eval
```

The `mim_probe` boilerplate name is retained as a configuration option. It does
not bundle the historical ignored dataset or checkpoint and therefore is not a
clean-checkout evidence claim.

### CoTTA

CoTTA expands to Continual Test-Time Adaptation. The YOLOZU phase-1 variant uses
an EMA teacher, augmented-view aggregation, and stochastic restoration.

**When to use:** a bounded continual stream diagnostic where operator review of
EMA, restoration, update norms, and prediction drift is required.

**Concrete repo result:** the YOLOZU variant runs and emits those diagnostics.
It is not reference-faithful CoTTA and efficacy is not established.

### EATA

EATA expands to Efficient Test-Time Adaptation. The YOLOZU phase-1 variant uses
selected samples and an anchor regularization term.

**When to use:** a controlled diagnostic of selection ratios, skip streaks,
loss, runtime, and update guards.

**Concrete repo result:** selection and regularization diagnostics are
implemented. They do not establish a recommended default or efficacy.

### SAR

SAR is a Sharpness-Aware test-time adaptation family. The YOLOZU phase-1
variant performs a perturb/restore update and records its guard behavior.

**When to use:** a controlled diagnostic of sharpness-aware update cost and
stable update behavior, with rollback enabled.

**Concrete repo result:** the perturbation path and safety diagnostics run.
This does not establish robustness gain or efficacy.

## Presets and reset policy

Presets are convenience configurations, not validated recommendations:

- `safe`, `adapter_only`
- `mim_safe`, `cotta_safe`, `eata_safe`, `sar_safe`
- `pose_safe`, `keypoints_safe`, `depth_safe`, `seg_safe`, `pose_mim`

`--ttt-reset sample` restores selected parameters for each image. Use it for
order-independent diagnostics. `--ttt-reset stream` keeps adapted parameters
across the stream and is order-dependent.

Start bounded:

- `--ttt-steps 1`
- `--ttt-batch-size 1`
- `--ttt-max-batches 1`
- `--ttt-stop-on-non-finite`
- `--ttt-rollback-on-stop`
- finite gradient, per-step update, total-drift, and loss guards

Task-aware auxiliary weights are optional and do not change the evidence
boundary:

```bash
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --dataset data/smoke \
  --config configs/yolo26_rtdetr_pose/yolo26n.json \
  --checkpoint checkpoints/rtdetr_pose.pt \
  --ttt \
  --ttt-method tent \
  --ttt-reset sample \
  --ttt-sdft-task pose \
  --ttt-aux-pose-weight 0.5 \
  --ttt-log-out reports/ttt_pose_diagnostic.json \
  --output reports/predictions_pose_ttt.json \
  --wrap
```

## MCP and Actions jobs

`ttt_job` and `ctta_job` map to the installed `yolozu export` command, not
`yolozu test`. They accept typed dataset/checkpoint/config/output/report fields
and do not accept arbitrary `extra_args`.

Both require workspace-relative dataset and checkpoint paths. Before queueing,
the selected RT-DETR model is instantiated and the checkpoint must report
`status=full` plus `load.loaded=true`. Missing or incompatible inputs return
`ok=false`, `exit_code=2`, `stage=preflight`, and `queued=false`.

An accepted job returns `job_id`. Poll `jobs_status` and verify the terminal
state, nested `ok`, nested `exit_code`, predictions artifact, and TTT report.

## Current literature

The [support matrix](tta_support_matrix.md#current-literature-candidates-not-implemented)
lists primary 2024--2026 detector/TTA papers reviewed for future work. Every
listed candidate is explicitly unimplemented in YOLOZU, unverified for the
YOLOZU RT-DETR path, and without YOLOZU efficacy evidence. Paper availability
does not imply implementation or software-license suitability.

## Promotion boundary

Promotion requires a separately reviewed evidence bundle with all of:

- real git commit and tool-version provenance
- exact checkpoint, config, dataset manifest, image order, baseline predictions,
  and adapted predictions hashes
- repeated seeds and an independently specified metric protocol
- latency, failures, guard stops, rollback, and regression slices
- explicit operator decision

The current checked-in fixture and local diagnostics intentionally set
`promotion_eligible=false`.
