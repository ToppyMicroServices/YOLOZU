# Test-Time Training (TTT) Integration State

This document records the implemented TTT integration. The historical filename is
retained so existing links remain valid.

TTT is an opt-in **Research** lane. The Stable predictions validation and evaluation
lane does not enable TTT, and implementing a method here does not establish production
readiness.

Method-specific Research design boundaries:

- [CoTTA phase-1 design](cotta_design_spec.md)
- [EATA phase-1 design](eata_design_spec.md)
- [SAR phase-1 design](sar_design_spec.md)

## Current guarantees

- **Zero-impact default:** `tools/export_predictions.py` does not adapt parameters
  unless `--ttt` is supplied.
- **Predictions interface contract stays unchanged:** adaptation runs before
  prediction; exported entries retain the same predictions schema.
- **Five implemented methods:** Tent, MIM, CoTTA, EATA, and SAR are dispatched by the
  shared runner.
- **Bounded updates:** parameter filters, step and batch limits, non-finite checks,
  update/loss guards, rollback, and stream/sample reset modes are available.
- **Observable runs:** wrapped exports and optional standalone logs record the resolved
  settings and the method report.
- **Scoped reproducibility controls:** `--ttt-seed` controls stochastic masks and
  restoration used by the shared runner. Full run reproducibility still depends on the
  selected model, backend, device, and runtime.

## Implemented Research-lane methods

| CLI value | State | Implemented behavior |
|---|---|---|
| `tent` | Implemented | Entropy minimization through `TentRunner`, with optional task-head consistency losses. |
| `mim` | Implemented | Masked image modeling with masked reconstruction. Models with the structured MIM branch use feature reconstruction plus an optional entropy term; compatible tensor models use masked input reconstruction. |
| `cotta` | Implemented | EMA teacher tracking, identity/horizontal-flip branches, configurable aggregation, and stochastic restoration. |
| `eata` | Implemented | Confidence/entropy/valid-detection sample selection with a pre-adaptation parameter-anchor penalty and skip controls. |
| `sar` | Implemented | Two-stage sharpness-aware entropy adaptation with configurable perturbation radius and adaptive scaling. |

`Supported` or `Implemented` on this page means that a tested implementation path
exists. It does not promote the method beyond Research maturity.

## Core modules

- `yolozu/tta/config.py`
  - Defines `SUPPORTED_TTT_METHODS` and the immutable `TTTConfig`.
- `yolozu/tta/cli_options.py`
  - Adds the shared CLI surface and converts parsed arguments to configuration,
    metadata, or forwarded command arguments.
- `yolozu/tta/integration.py`
  - Exposes `run_ttt(adapter, records, config=...)`, dispatches every supported
    method, applies shared guards, and returns `TTTReport`.
- `yolozu/tta/tent.py`
  - Implements `TentRunner`.
- `yolozu/tta/ttt_mim.py`
  - Implements block masking, parameter selection, generic and structured masked
    reconstruction steps, and MIM result helpers.
- `yolozu/tta/presets.py`
  - Provides conservative method and task presets. Presets are starting points, not
    evidence of production suitability.

## Implemented adapter interface

| Hook | `ModelAdapter` default | `RTDETRPoseAdapter` |
|---|---|---|
| `supports_ttt()` | Returns `False`. | Returns `True`. |
| `get_model()` | Returns `None`. | Lazily builds and returns the PyTorch model. |
| `build_loader(records, *, batch_size)` | Raises an unsupported-adapter error. | Preprocesses records and yields device tensors in bounded batches. |

`DummyAdapter` and `PrecomputedAdapter` retain the unsupported defaults. When TTT is
requested, the shared runner requires a non-null model and at least one batch and
fails with an explicit error when those conditions are not met.

## CLI and configuration surface

All flags below are implemented by `tools/export_predictions.py` through the shared
TTT argument builder.

| Area | Implemented flags |
|---|---|
| Enable and preset | `--ttt`, `--ttt-preset`, `--ttt-method` |
| Lifecycle and budget | `--ttt-reset`, `--ttt-steps`, `--ttt-batch-size`, `--ttt-lr`, `--ttt-max-batches`, `--ttt-seed` |
| Guards and rollback | `--ttt-stop-on-non-finite`, `--ttt-rollback-on-stop`, `--ttt-max-grad-norm`, `--ttt-max-update-norm`, `--ttt-max-total-update-norm`, `--ttt-max-loss-ratio`, `--ttt-max-loss-increase` |
| Parameter scope | `--ttt-update-filter`, `--ttt-include`, `--ttt-exclude` |
| MIM | `--ttt-mask-prob`, `--ttt-patch-size`, `--ttt-mask-value` |
| CoTTA | `--ttt-cotta-ema-momentum`, `--ttt-cotta-augmentations`, `--ttt-cotta-aggregation`, `--ttt-cotta-restore-prob`, `--ttt-cotta-restore-interval` |
| EATA | `--ttt-eata-conf-min`, `--ttt-eata-entropy-min`, `--ttt-eata-entropy-max`, `--ttt-eata-min-valid-dets`, `--ttt-eata-anchor-lambda`, `--ttt-eata-selected-ratio-min`, `--ttt-eata-max-skip-streak` |
| SAR | `--ttt-sar-rho`, `--ttt-sar-adaptive`, `--ttt-sar-first-step-scale` |
| Task-aware auxiliary losses | `--ttt-sdft-task`, `--ttt-aux-pose-weight`, `--ttt-aux-keypoints-weight`, `--ttt-aux-depth-weight`, `--ttt-aux-seg-weight`, `--ttt-aux-temperature` |
| Standalone report | `--ttt-log-out` |

The Boolean guard flags also expose their `--no-...` forms. `--ttt-method` accepts
`tent`, `mim`, `cotta`, `eata`, and `sar`; the default is `tent`. If TTT is enabled
with otherwise default-like core settings, the CLI resolves a conservative
method-specific preset before building `TTTConfig`.

## Export execution flow

1. Parse arguments, build the dataset manifest, and initialize the adapter.
2. Resolve the requested or automatic TTT preset.
3. If `--ttt-reset stream` is selected, call `run_ttt` before the normal
   `adapter.predict(records)` call and retain the adapted weights for that prediction
   stream.
4. If `--ttt-reset sample` is selected, restore the selected parameters and
   normalization buffers around each per-record adaptation and prediction.
5. Apply opt-in TTA, if requested, after the base prediction call.
6. Serialize predictions without changing the predictions interface contract.

## Reports and logs

With `--wrap`, `meta.ttt` contains the resolved configuration and the `TTTReport`.
The report includes the method, reset mode, requested and completed steps, batches
used, elapsed time, losses, selected parameter count, warnings, stop state, and
bounded step metrics. Method-specific fields include MIM mask ratio, CoTTA
augmentation/EMA/restoration details, EATA selection and anchor metrics, and SAR
first/second losses and perturbation data.

When `--ttt-log-out` is used with `--ttt`, the same settings and report are written to
a standalone JSON artifact.

## Verified test coverage

- `tests/test_ttt_integration.py` exercises Tent, generic and structured MIM, CoTTA,
  EATA, SAR, guard rollback, and unsupported adapters.
- `tests/test_ttt_mim.py` covers masking, masked reconstruction, parameter filters,
  and MIM steps.
- `tests/test_ttt_safety.py` covers non-finite and update/loss guard behavior.
- `tests/test_ttt_cli_options.py` covers parsing and configuration/forwarding helpers.
- `tests/test_export_predictions_ttt_cli.py` covers real `--help`, disabled defaults,
  and unsupported-adapter failure behavior.
- `tests/test_ttt_docs_implementation_alignment.py` keeps the method tables, adapter
  hooks, documented flags, implementation method list, and rendered help aligned.

## Genuine remaining constraints

- TTT requires PyTorch and a model adapter that exposes a differentiable model and
  tensor loader. Only `RTDETRPoseAdapter` provides that built-in path today.
- A compatible output is method-dependent: entropy methods need extractable logits,
  generic MIM needs a reconstruction-compatible tensor, and structured MIM needs its
  model branch enabled.
- CoTTA augmentation branches are currently limited to identity and horizontal flip.
- Presets and guard rails limit operational risk but do not prove accuracy gains,
  acceptable latency, or production readiness for a new dataset, model, backend, or
  device.
- Promotion beyond Research requires fixed-shift before/after evidence, rollback and
  drift review, and an acceptable resource envelope. See
  [TTT protocol](ttt_protocol.md) and
  [TTT compare boilerplates](ttt_compare_boilerplates.md).

## References

- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent:
  Fully Test-Time Adaptation by Entropy Minimization. In *International Conference
  on Learning Representations (ICLR)*.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked
  Autoencoders Are Scalable Vision Learners. In *Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (CVPR)*.
