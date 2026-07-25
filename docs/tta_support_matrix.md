# TTT / TTA support matrix

Updated: 2026-07-25

This page separates runnable code, implementation fidelity, measured efficacy,
and production maturity. These are independent properties.

## Maturity boundary

- Non-parameter-updating TTA is **Experimental** and opt-in.
- Parameter-updating TTT is **Research** and opt-in.
- A runnable path does not establish reference fidelity, efficacy, or production
  readiness.
- Stable parent commands do not promote their Experimental or Research options.

### TTA execution modes

- `--tta` uses `postprocess` by default. It transforms exported predictions and
  does not rerun the model.
- `--tta --tta-mode model` reruns one horizontally flipped branch for
  `rtdetr_pose`, maps it back, and merges it with the baseline predictions.
- Other adapters warn and fall back to `postprocess` when `model` is requested.

## Algorithms

| Method | Execution | Machine-readable profile | Fidelity | Efficacy |
|---|---|---|---|---|
| Tent | Runnable, Research | `yolozu_detector_entropy_v2` | Detector-adapted YOLOZU entropy variant; not a reference-faithful Tent reproduction | Not established |
| MIM | Conditional, Research | `yolozu_structured_mim_v1` | Masked image modeling with masked reconstruction; real compare preflight requires a compatible model exposing the structured MIM hook | Not established |
| CoTTA | Runnable, Research | `yolozu_phase1_variant` | YOLOZU phase-1 variant; not reference-faithful CoTTA | Not established |
| EATA | Runnable, Research | `yolozu_phase1_variant` | YOLOZU phase-1 variant; not reference-faithful EATA | Not established |
| SAR | Runnable, Research | `yolozu_phase1_variant` | YOLOZU phase-1 variant; not reference-faithful SAR | Not established |

`yolozu.tta.method_profiles.TTT_METHOD_PROFILES` is the machine-readable source
for the profile, runnable, maturity, fidelity, efficacy, and loss-semantics
fields.

## Detector loss semantics

The generic entropy paths reduce the final class axis over every remaining
element, including detector queries.

- Foreground selection: none.
- No-object/background: included when the model exposes it in the final class
  axis; otherwise its identity is unknown to the generic runner.
- Query correspondence: not established across views or steps.
- CoTTA's augmented-view aggregation does not establish object-query
  correspondence.

These semantics differ from classification-only formulations and are part of
the YOLOZU profile, not evidence of paper-level equivalence.

## Evidence boundary

- `synthetic_fixture` sources may reproduce documentation layout only. They
  cannot contain measured metric, score, loss, latency, quality, or improvement
  fields and are never promotion eligible.
- `measured` figure sources require hash-bound checkpoint, config, dataset
  manifest, image order, baseline predictions, adapted predictions, seed,
  resolvable commit, and tool versions.
- Local resources must be Git-tracked and hash verified. A release HTTPS URL
  plus hash remains `declared_not_fetched` unless another workflow downloads
  and verifies it.
- `simple_map_proxy` values use `proxy_ap50` / `proxy_ap50_95`. They are not
  COCO `mAP50` / `mAP50-95`.
- A local diagnostic compare records `efficacy_conclusion=not_established` and
  cannot promote a checkpoint.

## Current detector-adaptation candidates

The entries below are literature-review candidates, not YOLOZU features. As of
2026-07-25 they are unimplemented in YOLOZU, unverified on YOLOZU's RT-DETR
adapter, and have no YOLOZU efficacy result. Software-license suitability must
be checked before any implementation decision.

| Year / venue | Candidate | Relevance | YOLOZU status |
|---|---|---|---|
| CVPR 2024 | [What, How and When Should Object Detectors Update in Continually Changing Test Domains?](https://openaccess.thecvf.com/content/CVPR2024/html/Yoo_What_How_and_When_Should_Object_Detectors_Update_in_Continually_CVPR_2024_paper.html) | Detector-specific update gating | Unimplemented; RT-DETR unverified |
| CVPRW 2024 | [Fully Test-time Adaptation for Object Detection](https://openaccess.thecvf.com/content/CVPR2024W/MAT/html/Ruan_Fully_Test-time_Adaptation_for_Object_Detection_CVPRW_2024_paper.html) | IoU-filtered detector adaptation | Unimplemented; RT-DETR unverified |
| CVPR 2025 | [Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Efficient_Test-time_Adaptive_Object_Detection_via_Sensitivity-Guided_Pruning_CVPR_2025_paper.pdf) | Reduces adaptive-detector update cost | Unimplemented; RT-DETR unverified |
| ICCV 2025 | [Continual Adaptation: Environment-Conditional Parameter Generation for Object Detection in Dynamic Environments](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Continual_Adaptation_Environment-Conditional_Parameter_Generation_for_Object_Detection_in_Dynamic_ICCV_2025_paper.pdf) | Dynamic detector parameter generation | Unimplemented; RT-DETR unverified |
| NeurIPS 2025 | [Foundation-model test-time adaptation for object detection](https://nips.cc/virtual/2025/poster/118473) | Foundation-model-assisted detector adaptation | Unimplemented; RT-DETR unverified |
| WACV 2026 | [Logit-Adjusted Test-Time Adaptation under Partial Class Imbalance](https://openaccess.thecvf.com/content/WACV2026/html/Weerasinghe_Logit-Adjusted_Test-Time_Adaptation_under_Partial_Class_Imbalance_WACV_2026_paper.html) | Class-imbalance handling; paper evaluation is classification-oriented | Unimplemented; detector and RT-DETR applicability unverified |
| CVPRW 2026 | [Topology-Guided Test-Time Adaptation via Persistent Homology](https://openaccess.thecvf.com/content/CVPR2026W/ABAW/papers/Mutlu_Topology-Guided_Test-Time_Adaptation_via_Persistent_Homology_From_Affective_Behavior_Analysis_CVPRW_2026_paper.pdf) | Includes YOLOS/DETR experiments in a different task context | Unimplemented; YOLOZU RT-DETR applicability unverified |

## Safety and reproducibility controls

| Control | Status | Boundary |
|---|---|---|
| Reset policy | Implemented | `stream` retains adapted state; `sample` resets at explicit boundaries |
| Update budget | Implemented | Steps, batches, gradient/update norms, and loss guards are bounded |
| Failure state | Implemented | Compare plans record `running`, `failed`, `not_executed`, or `completed` with the exact stage |
| Checkpoint preflight | Implemented | Full config/checkpoint compatibility is required, including structured MIM support for a MIM compare |
| Metrics comparability | Conditional | COCO metrics require the COCO evaluator; fallback diagnostics remain explicitly named proxy AP |

## Where to go next

- [TTT protocol](ttt_protocol.md)
- [TTT compare boilerplates](ttt_compare_boilerplates.md)
- [Predictions interface contract](predictions_schema.md)
- [External inference](external_inference.md)
