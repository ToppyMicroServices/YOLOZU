# TTT before-after compare boilerplates

The compare runner provides concise, fail-closed local diagnostics for Tent,
MIM, CoTTA, EATA, SAR, and detector-native response consistency. All methods remain Research. A completed run does
not establish efficacy.

## Shortest path

```bash
bash scripts/ttt_compare.sh \
  --method tent \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose.pt \
  --out reports/ttt_compare/tent \
  -n 1 \
  --no-eval
```

The wrapper delegates to `tools/run_ttt_compare.py`. Long and short flags are
equivalent:

| Concise | Long |
|---|---|
| `--method` | `--boilerplate` |
| `--data` | `--dataset` |
| `--weights` | `--checkpoint` |
| `--out` | `--run-dir` |
| `-n` | `--max-images` |
| `--no-eval` | `--skip-eval` |

## Preconditions

The runner validates these inputs before writing a successful plan:

- tracked boilerplate and its selected config
- existing dataset split with at least one image
- non-empty checkpoint
- RT-DETR model construction
- canonical checkpoint loader `status=full`
- `load.loaded=true`
- structured MIM hook when method is MIM

`--dry-run` performs the same validation. Partial/incompatible checkpoints,
arbitrary bytes, `load.loaded=false`, and the noncanonical status value
`compatible` fail before execution.

## Method map

| Method | Boilerplate | Profile | Extra precondition | Efficacy |
|---|---|---|---|---|
| Tent | `tent` | `yolozu_detector_entropy_v2` | Full checkpoint compatibility | Not established |
| MIM | `mim` | `yolozu_structured_mim_v1` | Full compatibility and structured MIM hook | Not established |
| CoTTA | `cotta` | `yolozu_phase1_variant` | Full checkpoint compatibility | Not established |
| EATA | `eata` | `yolozu_phase1_variant` | Full checkpoint compatibility | Not established |
| SAR | `sar` | `yolozu_phase1_variant` | Full checkpoint compatibility | Not established |
| Detection response | `detector_response` | `yolozu_detection_response_v1` | Full YOLO26n compatibility and foreground/no-object logits | Not established |

The `mim_probe` boilerplate remains a configuration alias for a YOLO26 MIM
model. Its historical ignored dataset/checkpoint are not bundled and are not a
clean-checkout success path.

## Commands

Tent:

```bash
bash scripts/ttt_compare.sh \
  --method tent \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose.pt \
  --out reports/ttt_compare/tent \
  -n 1 \
  --no-eval
```

MIM:

```bash
bash scripts/ttt_compare.sh \
  --method mim \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose_mim.pt \
  --out reports/ttt_compare/mim \
  -n 1 \
  --no-eval
```

CoTTA prerequisite-only plan:

```bash
bash scripts/ttt_compare.sh \
  --method cotta \
  --data data/smoke \
  --weights checkpoints/rtdetr_pose.pt \
  --out reports/ttt_compare/cotta \
  -n 1 \
  --dry-run
```

Replace `cotta` with `eata` or `sar` for those YOLOZU phase-1 variants.

Detection-native response consistency:

```bash
bash scripts/ttt_compare.sh \
  --method detector_response \
  --data data/smoke \
  --weights checkpoints/yolo26n.pt \
  --out reports/ttt_compare/detector_response \
  --image-size 320 --score-threshold 0.1
```

This boilerplate selects confident foreground queries, excludes the final
no-object class, and keeps same-query class/box responses consistent across a
weak photometric view. Its default minimum is one selected query; below the
configured `--ttt-response-min-selected` value it records abstention and skips
the pure response backward/optimizer update while restoring normalization buffers.

## Generated artifacts

For `--out reports/ttt_compare/tent`, the runner writes:

- `plan.json`
- `baseline_predictions.json`
- `tent_predictions.json`
- `tent_ttt_log.json`
- optional baseline/adapted evaluation JSON
- `tent_before_after_compare.json`
- `tent_before_after_compare.md`

The JSON/Markdown compare pair is staged and published together. A failure at
export, evaluation, fallback evaluation, artifact loading, prediction compare,
or report publication sets `execution_status.state=failed` with a specific
stage. Stale success artifacts are removed.

The final report includes:

- `evidence_kind=local_diagnostic`
- `efficacy_conclusion=not_established`
- `promotion_eligible=false`
- method profile and detector loss semantics
- config/checkpoint/dataset provenance
- before/after prediction counts and guard diagnostics
- selected-query and abstention diagnostics

If pycocotools is unavailable, fallback keys are `proxy_ap50` and
`proxy_ap50_95`. They are non-COCO diagnostics and are never emitted as COCO
`map50`/`map50_95`.

## Reading order

1. Check `plan.json` terminal state and failed stage.
2. Confirm checkpoint preflight says `status=full` and `load.loaded=true`.
3. Read the TTT log for steps, guards, rollback, update norms, and method
   profile.
4. Compare predictions and note detector-query limitations.
5. Read metrics only under their declared backend/semantics.
6. Keep the conclusion at `not_established` unless a separate reviewed evidence
   bundle satisfies the promotion criteria in
   [TTT protocol](ttt_protocol.md#promotion-boundary).
