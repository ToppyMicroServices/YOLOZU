# CoTTA Design Specification for YOLOZU (Phase 1)

This document outlines the initial safety-bounded Research integration scope for
Continual Test-Time Domain Adaptation (CoTTA) within YOLOZU. It does not establish
production readiness.

## Objective

The primary goal is to mitigate long-term online drift and catastrophic forgetting during test-time adaptation while minimizing deployment risks.

Phase 1 focuses exclusively on operationally safe components:
- An Exponential Moving Average (EMA) teacher model.
- Multi-augmentation prediction averaging.
- Stochastic restoration restricted to safe parameter subsets (e.g., LoRA and normalization layers).

## Update Scope (Phase 1)

### Permitted Trainable Targets
- Low-Rank Adaptation (LoRA) parameters.
- Normalization affine parameters (e.g., LayerNorm, GroupNorm, or BatchNorm affine weights, where applicable).

### Explicitly Excluded in Phase 1
- Full backbone weight adaptation.
- Unrestricted, full-model optimizer updates.

**Rationale:** By initiating adaptation with a minimal blast radius, we can effectively prevent irreversible parameter drift and maintain model stability.

## Teacher Model Specification

- The teacher model is defined as the Exponential Moving Average (EMA) of the student model:
  - `theta_teacher <- m * theta_teacher + (1 - m) * theta_student`
- The teacher's weights are updated immediately following each adaptation step.
- The EMA momentum `m` is a configurable hyperparameter.
- Crucially, the teacher's parameters are never directly optimized via gradient descent.

## Multi-Augmentation Prediction Averaging

### Supported Initial Augmentation Set
- Identity (no augmentation)
- Horizontal flip

*Note: Additional augmentations, such as scale jitter or mild color jitter, may be introduced in future phases but are excluded from the Phase 1 default configuration.*

### Aggregation Behavior
- The student model (or the teacher-selected pathway) processes each augmented branch independently.
- Predictions are subsequently mapped back to the original image coordinate space.
- Aggregation is performed using confidence-aware averaging prior to the final post-processing and Non-Maximum Suppression (NMS) stages.
- To ensure reproducibility, the aggregation strategy must remain strictly deterministic given a specific random seed and configuration.

## Stochastic Restoration (Safe Mode)

- Restoration mechanisms are applied exclusively to the permitted trainable targets.
- For each eligible parameter element, the system restores its value from a source snapshot with a probability of `p_restore`.
- By default, the source snapshot corresponds to the model's initial, pre-adaptation state for the current session.
- This restoration process is executed at a configured cadence (e.g., every `N` adaptation steps).

## Safety Boundaries and Guardrails

To ensure a safe Phase 1 rollout, the following guardrails are mandatory:
- Gradient norm clipping (`max_grad_norm`).
- A per-step update norm cap (`max_update_norm`).
- A cumulative update norm cap (`max_total_update_norm`).
- A divergence stop condition triggered by a loss-ratio threshold (`max_loss_ratio`).
- An optional, immediate fallback to baseline weights in the event of a guardrail breach.

**Failure Behavior:**
- If a hard breach occurs, the current adaptation step is immediately aborted.
- The event is recorded in the TTT diagnostics output.
- The model state is reverted according to the defined fallback policy.

## Configuration Parameters (Phase 1)

The following minimum configuration knobs are required:
- `ttt.method: cotta`
- `ttt.cotta.ema_momentum`
- `ttt.cotta.augmentations` (Initial defaults: `identity`, `hflip`)
- `ttt.cotta.aggregation` (Initial default: confidence-weighted mean)
- `ttt.cotta.restore_prob`
- `ttt.cotta.restore_interval`
- `ttt.update_filter` (Must support `norm_only`, `lora_only`, and a combined safe subset)
- `ttt.max_grad_norm`
- `ttt.max_update_norm`
- `ttt.max_total_update_norm`
- `ttt.max_loss_ratio`

## Logging and Observability

Test-Time Training (TTT) logs must capture the following details:
- Run/session ID.
- Adaptation method (e.g., `cotta`).
- Update scope summary (detailing which parameter groups were active).
- Teacher EMA momentum value.
- The applied augmentation set and aggregation mode.

## References

- Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual Test-Time Domain Adaptation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 7201-7211). [arXiv:2203.13591](https://arxiv.org/abs/2203.13591)
- Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Huang, J. (2022). Efficient Test-Time Model Adaptation without Forgetting. In *International Conference on Machine Learning (ICML)*.
- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020). Test-Time Training with Self-Supervision for Generalization under Shifts. In *International Conference on Machine Learning (ICML)*.
- restoration probability + applied restore count
- guardrail metrics and triggered events

## Rollout policy

1. Default-off (`--ttt` required).
2. Phase-1 safe preset enables:
   - LoRA/Norm-only updates
   - conservative restore probability
   - strict guardrails
3. Run on fixed subset first; compare baseline vs CoTTA under same protocol.

## Out of scope (phase 1)

- full-parameter CoTTA updates
- heavy augmentation ensembles
- backend-specific optimized CoTTA kernels

## Exit criteria to move beyond phase 1

- no repeated guardrail breaches in smoke/protocol runs
- measurable drift suppression on fixed evaluation slices
- no unacceptable latency blow-up versus baseline TTT path
