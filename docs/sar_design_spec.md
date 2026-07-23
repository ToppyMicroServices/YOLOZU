# SAR design spec for YOLOZU (phase 1)

This document defines a safety-bounded Research rollout scope for YOLOZU test-time
adaptation. It does not establish production readiness.

## Goal

Improve robustness under challenging distribution shifts by introducing sharpness-aware entropy minimization with controlled update scope.

## Phase-1 scope

### Allowed update targets

- LoRA parameters only (default)
- optional norm affine subset in controlled experiments (not default)

### Explicitly disallowed

- full-backbone sharpness-aware updates
- unrestricted full-model SAM-style optimization

Rationale: SAR can be unstable with broad update scope; phase-1 prioritizes low blast radius.

## Optimization behavior

SAR phase-1 uses two-stage sharpness-aware update over selected trainable parameters:

1. compute gradient on entropy objective,
2. perturb weights along normalized gradient direction,
3. compute second loss at perturbed point,
4. apply optimizer update on original weights.

Conceptual objective:

- $\min_{\theta} \max_{\lVert \epsilon \rVert \leq \rho} L_{entropy}(\theta + \epsilon)$

with `rho` bounded conservatively for phase-1.

## Normalization policy guidance

For SAR experiments requiring normalization updates:

- prefer GN/LN-first configurations,
- avoid BN-stat-heavy behavior in tiny/unstable batches,
- keep BN running-stat side effects monitored and rollback-capable.

Default rollout remains LoRA-only.

## Required configuration knobs (phase-1)

- `ttt.method: sar`
- `ttt.update_filter` defaulting to `lora_only`
- `ttt.sar.rho`
- `ttt.sar.adaptive` (scale by parameter norm)
- `ttt.sar.first_step_scale`
- standard guardrails: `max_grad_norm`, `max_update_norm`, `max_total_update_norm`, `max_loss_ratio`, `rollback_on_stop`

## Safety boundaries and expected costs

Known expected cost profile:

- approximately ~2x forward/backward cost per adaptation step versus single-step entropy update,
- increased memory traffic due to perturb-and-restore pass.

Required safeguards:

- strict gradient/update caps,
- immediate rollback on guardrail breach,
- explicit stop reason logging,
- conservative default `steps=1` and `max_batches=1`.

## Logging and observability

SAR logs should include:

- method + update filter summary,
- `rho`, adaptive mode, and first-step scale,
- per-step first/second loss,
- perturbation norm and final update norm,
- rollback/stop events.

## Rollout policy

1. default-off (`--ttt` required)
2. start with LoRA-only + strict guardrails
3. run limited-slice robustness checks before wider enablement

## Out of scope (phase-1)

- full-model SAM variants,
- heavy augmentation-coupled SAR,
- backend-specific fused SAR kernels.

## Exit criteria to phase-2

- robustness gain over baseline/eata on designated mixed-shift slices,
- acceptable overhead envelope,
- no recurring unsafe drift events under default profile.
