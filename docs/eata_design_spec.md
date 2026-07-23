# EATA design spec for YOLOZU (phase 1)

This document defines a safety-bounded phase-1 Research scope for EATA on detection
and pose in YOLOZU. It does not establish production readiness.

## Goal

Reduce unstable test-time updates by:

- adapting only on informative samples,
- limiting update scope to safe parameter subsets,
- adding anti-forgetting regularization during online adaptation.

## Phase-1 update scope

### Allowed trainable targets

- normalization affine parameters (LN/GN/BN affine where used)
- LoRA parameters
- optional detection/pose head subset (explicit allowlist only)

### Explicitly disallowed

- full backbone updates
- unrestricted full-model adaptation

Default phase-1 preset should start with `norm_only` or `lora_norm_only` and keep head updates opt-in.

## Active sample selection (detection/pose aware)

EATA update steps consume only selected samples from the incoming batch.

## Selection signals

For each image, compute selection features from prediction outputs:

- detection confidence aggregate (`mean_topk_conf`)
- detection entropy aggregate (`mean_topk_entropy`)
- pose confidence aggregate (`mean_kpt_conf` when keypoints exist)
- optional agreement score across light augment branches (phase-1 optional)

## Selection rules (phase-1)

Sample is selected when all are satisfied:

- confidence lower bound is met (`conf >= conf_min`) to avoid near-random noise,
- entropy is within adaptive band (`entropy_min <= entropy <= entropy_max`),
- valid detection count satisfies floor (`num_valid >= min_valid_dets`).

Rationale:

- very low confidence samples are often harmful,
- very low entropy samples provide little adaptation signal,
- extremely high entropy samples can destabilize updates.

## Anti-forgetting regularization

Phase-1 regularization anchors online updates to the pre-adaptation snapshot.

## Regularization terms

- parameter-anchor penalty:
  - $L_{anchor} = \sum_i \lambda_i \lVert \theta_i - \theta_i^{(0)} \rVert_2^2$
  - apply only to trainable subset selected by update filter
- optional prediction-anchor penalty (future toggle):
  - KL or MSE to frozen reference logits on selected samples

Total optimization objective (phase-1):

- $L = L_{adapt} + \lambda_{anchor} L_{anchor}$

where `L_adapt` is entropy-based adaptation objective over selected samples.

## Safety boundaries and guardrails

Reuse existing TTT guardrails for EATA path:

- `max_grad_norm`
- `max_update_norm`
- `max_total_update_norm`
- `max_loss_ratio` / `max_loss_increase`
- rollback-on-stop behavior

Additional EATA-specific safety checks:

- minimum selected-sample ratio per step (`selected_ratio_min`)
- skip-step when selected set is empty
- max consecutive skipped steps (`max_skip_streak`) to trigger early stop with warning

## Required configuration knobs (phase-1)

- `ttt.method: eata`
- `ttt.update_filter` (`norm_only`, `lora_only`, `lora_norm_only`, optional head allowlist)
- `ttt.eata.conf_min`
- `ttt.eata.entropy_min`
- `ttt.eata.entropy_max`
- `ttt.eata.min_valid_dets`
- `ttt.eata.anchor_lambda`
- `ttt.eata.selected_ratio_min`
- `ttt.eata.max_skip_streak`
- standard guardrails (`max_*`, rollback)

## Logging and observability

EATA logs should include:

- selected/total sample counts per step
- selection feature summaries (confidence/entropy/valid-det stats)
- anchor loss and adapt loss components
- update scope summary (active parameter groups)
- guardrail events and rollback status

## Rollout policy

1. default-off (`--ttt` required)
2. use conservative preset first (`norm_only` + strict thresholds)
3. compare baseline vs EATA on fixed slice before broad rollout

## Out of scope (phase-1)

- heavy augmentation disagreement pipelines
- full-model EATA updates
- backend-specific kernel optimizations

## Exit criteria for phase-2

- stable selected-sample ratio across target slices
- reduced forgetting/drift proxy metrics vs baseline
- no recurring guardrail failures under default preset
