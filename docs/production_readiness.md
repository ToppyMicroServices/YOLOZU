# Production Readiness

This page is the source of truth for how YOLOZU classifies its main lanes:

- `Stable`: the default production lane
- `Experimental`: useful, but qualification is environment- or workflow-dependent
- `Research`: reproducible and supported, but not the first production lane

## Primary production lane

YOLOZU's primary role is an evaluation layer built around one stable predictions interface contract:

- validate wrapped `predictions.json`
- evaluate predictions reproducibly
- compare outputs across frameworks and runtimes

If your team already has inference outputs and wants fair evaluation without rewriting the whole pipeline, this is the main production path.

## Capability map

| Area | Maturity | Production posture | Primary references |
|---|---|---|---|
| Predictions validation/evaluation | Stable | Default production lane | [`predictions_schema.md`](predictions_schema.md), [`external_inference.md`](external_inference.md), [`../README.md`](../README.md) |
| Environment-qualified adaptive local vision | Experimental MCP recommendation and pinned processing; qualification, reviewed activation, and pure-selection foundations implemented | `doctor` emits a privacy-safe EnvironmentProfile. `qualify-image-pipeline` pins bounded inputs/assets, runs the frozen repeat/soak protocol behind a child-process watchdog, and publishes an unactivated managed report. `activate-qualification-evidence` dry-runs every trust, freshness, lifecycle, and stale-head gate and mutates only with explicit review plus `--approve`. `recommend_image_pipeline` returns a selected or abstained SelectionDecision without execution. `process_images` requires that complete decision, repeats pinned current-state checks, defaults to no-write dry-run, and permits explicit bounded managed publication only through registered code-owned network-free routes. Local reports are limited to site-qualified scope; arbitrary JSON remains nonselectable. The packaged registry, runner maps, and public evidence stream are empty, so the default recommendation abstains, no real bundle can run, and no support or performance evidence is claimed. No model adapter is available. | [`doctor_diagnostics.md`](doctor_diagnostics.md), [`adaptive_image_routing.md`](adaptive_image_routing.md), [`../reports/adaptive_qualification_foundation_2026-08-25.md`](../reports/adaptive_qualification_foundation_2026-08-25.md), [`../reports/adaptive_vision_roadmap.md`](../reports/adaptive_vision_roadmap.md), [`roadmap.md`](roadmap.md) |
| Dataset I/O and mask-only label derivation | Deferred as standalone capabilities | Implemented and tested inside dataset workflows, but implementation presence and a Stable parent CLI are not standalone production-readiness evidence | [`yolozu_spec.md`](yolozu_spec.md), [`dataset_contract.md`](dataset_contract.md), [`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md) |
| Inference constraints and template gating | Deferred as standalone capabilities | Adapter-internal utilities with no independent public production lane; qualify them with the consuming model and protocol | [`yolozu_spec.md`](yolozu_spec.md), [`gate_weight_tuning.md`](gate_weight_tuning.md), [`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md) |
| Backend parity / benchmark orchestration | Experimental | Useful after environment-specific qualification; classification, OBB, segmentation, keypoints, depth, and pose6d have artifact-backed real eval/parity lanes, without claiming backend inference | [`backend_parity_matrix.md`](backend_parity_matrix.md), [`benchmark_mode.md`](benchmark_mode.md), `manual/chapters/09_parity_bench_protocols.tex` |
| BOP T-LESS object 6DoF | Research | Real CC-BY-4.0 T-LESS acquisition, strict GT, three-seed task-native evaluation, official BOP19 target export/evaluation, and independent semantic reproduction are tracked. Official protocol completion does not establish pose efficacy. This is rigid-object pose, not human 3D skeleton pose. | [`bop_tless_protocol.md`](bop_tless_protocol.md), [`../reports/bop_tless_evidence_2026-07-30.md`](../reports/bop_tless_evidence_2026-07-30.md), [`../reports/bop19_tless_official_evidence_2026-07-30.md`](../reports/bop19_tless_official_evidence_2026-07-30.md), `manual/chapters/16_depth_6dof_symmetry.tex` |
| YOLOZU-synthgen handoff | Experimental | Intake/eval path is reproducible, but external generator handoff still needs qualification | [`synthgen_repo_integration.md`](synthgen_repo_integration.md), [`synthgen_contract.md`](synthgen_contract.md), `manual/chapters/21_synthgen_repo_integration.tex` |
| macOS / MPS paths | Experimental | Supported only when `torch.backends.mps.is_available()` is true; treat as qualification, not blanket readiness | [`install.md`](install.md), [`doctor_diagnostics.md`](doctor_diagnostics.md), [`continual_learning.md`](continual_learning.md) |
| TTA | Experimental | Opt-in and non-parameter-updating: the default postprocess mode transforms exported predictions, while the `rtdetr_pose` model mode reruns one augmented branch and merges predictions; qualify each mode against the Stable baseline | [`tta_support_matrix.md`](tta_support_matrix.md), [`learning_features.md`](learning_features.md) |
| Continual learning / self-distillation | Research | Use for governed experiments and promotion-gated workflows over evaluated artifacts, not as the first production lane | [`research_lanes.md`](research_lanes.md), [`continual_learning.md`](continual_learning.md), `manual/chapters/14_continual_learning.tex` |
| TTT | Research | Short-horizon inference adaptation over evaluated artifacts; do not treat as an automatic checkpoint-promotion path | [`research_lanes.md`](research_lanes.md), [`ttt_protocol.md`](ttt_protocol.md), `manual/chapters/15_ttt_tent_mim.tex` |
| Hessian refinement | Research | Offline/local post-inference correction path over evaluated artifacts | [`research_lanes.md`](research_lanes.md), [`hessian_solver.md`](hessian_solver.md), `manual/chapters/10_ttt_hessian.tex` |
| Training platform | Stable reference interface + environment-qualified external execution | RT-DETR pose is the richest in-repo path. Strict T-LESS GT covers task-native evidence. Ultralytics, HF DETR, and Detectron2 completed real training in two environments. YOLOX, MMDetection, MMPose, MMSeg, and TAO completed two independent compatible Linux/CUDA runs with checkpoints, resource evidence, and structural handoff validation. Runtime availability, structural handoffs, and bounded zero/null metrics do not establish training quality. | [`training_backend_interface.md`](training_backend_interface.md), [`training_capability_matrix.md`](training_capability_matrix.md), [`training_orchestration.md`](training_orchestration.md), [`external_finetune_smoke.md`](external_finetune_smoke.md), [`../reports/external_runtime_evidence_2026-07-30.md`](../reports/external_runtime_evidence_2026-07-30.md), [`../reports/external_runtime_compatible_host_evidence_2026-07-30.md`](../reports/external_runtime_compatible_host_evidence_2026-07-30.md) |
| Installed CLI and mixed-lane entrypoints | Mixed by capability | A parent entrypoint may be Stable while opt-in subcommands or flags remain Experimental or Research | [`generated/cli_reference.md`](generated/cli_reference.md), [`tools_index.md`](tools_index.md), [`research_lanes.md`](research_lanes.md) |

## Maturity applies at the narrowest declared surface

Manifest `maturity` is entrypoint-level metadata, not a transitive guarantee for every
subcommand or flag exposed by that entrypoint. When an entrypoint and a narrower
capability have different labels, use the narrower capability label:

- `yolozu` is a Stable parent entrypoint for the core validation/evaluation surface.
  Benchmark and external-training sub-lanes retain their separately declared
  capability or backend maturity, and research workflows remain Research.
- `export_predictions` has a Stable baseline export path. Optional acceleration flags
  require backend/device qualification. Opt-in TTA is Experimental, and opt-in TTT
  flags are Research; both are disabled by default.
- A backend-specific Stable label does not promote an Experimental orchestration
  entrypoint, and an Experimental parent does not demote separately documented Stable
  artifacts or interface contracts.

Implementation presence, passing unit tests, and discoverability from a Stable parent
are evidence of availability, not sufficient evidence for maturity promotion.

## Version Compatibility Matrix

| Surface | Pinned / expected version | Production rule | Reference |
|---|---|---|---|
| Predictions schema | `schema_version=1`, entry `schema_version=2` | Stable validation/evaluation path; breaking changes require schema governance | [`schema_governance.md`](schema_governance.md), [`predictions_schema.md`](predictions_schema.md) |
| Python package | `pyproject.toml` package metadata | Release gates must keep package version, changelog, and release trigger aligned | [`release_reliability_checklist.md`](release_reliability_checklist.md) |
| PyTorch / ONNX Runtime | Package floors and the current CPU/test pins are separate: Torch-backed extras require `torch>=2.10.0`; CI pins `torch==2.10.0+cpu`, `onnx==1.21.0`, and `onnxruntime==1.24.2` | Treat the pins as evidence for their named repository environments, not every device or wheel variant | [`versions.md`](versions.md), `requirements-locks/requirements-ci.lock` |
| ONNX export | `tools/export_trt.py` defaults to opset 18; current TensorRT/YOLO examples and GPU smoke explicitly pin opset 17 | Record the chosen opset and regenerate artifacts when the exporter/runtime protocol changes | [`versions.md`](versions.md), [`onnx_export_parity.md`](onnx_export_parity.md), [`tensorrt_pipeline.md`](tensorrt_pipeline.md) |
| TensorRT / CUDA | No universal static version pair; the self-hosted workflow is configured with `nvcr.io/nvidia/tensorrt:24.08-py3` | Require run-specific container, driver/GPU, CUDA context, TensorRT version, engine metadata, and parity evidence | [`versions.md`](versions.md), [`tensorrt_pipeline.md`](tensorrt_pipeline.md) |
| Training handoff | `training_run_summary` / `training_handoff` schemas | Machine-readable handoff must validate before external promotion | [`training_orchestration.md`](training_orchestration.md), [`schema_governance.md`](schema_governance.md) |

## Production Readiness Matrix

| Lane | Default? | Required proof before relying on it | Promotion boundary |
|---|---:|---|---|
| Evaluate existing predictions | Yes | `doctor --proof`, strict validation, evaluation report | Production-ready when report schema and protocol match |
| External inference bridge | Yes, when predictions are supplied | Predictions interface contract validation plus license/runtime note | Production remains with the caller's inference stack |
| Benchmark / parity | No | Artifact-backed or real backend report with explicit skipped lanes | Promote per backend/runtime after environment qualification |
| Training / external bridge | No | Dry-run or run bundle with next command, expected outputs, and license boundary | Promote only after exported predictions evaluate cleanly |
| Research lanes | No | Separate research report over already evaluated artifacts | Never overwrite the stable evaluation result |

## Stable today

- predictions interface contract: wrapped `predictions.json` plus protocol-pinned `meta.export_settings`
- prediction validation and evaluation flows
- install / `doctor` / repo smoke path
- CPU-friendly demo and smoke paths

These are the areas to rely on first for production adoption.

## Experimental adaptive local vision program

The adaptive local-vision roadmap targets an Experimental lane. The current
recommendation and pinned-processing surfaces do not add a current model,
streaming, tracking, or OCR support claim. Their machine-readable scope is
[`adaptive_vision_roadmap.json`](../yolozu/data/manifest/adaptive_vision_roadmap.json),
and the generated public projection is
[`adaptive_vision_roadmap.md`](../reports/adaptive_vision_roadmap.md). Live progress
remains in Beads under `YOLOZU-ll2.81`.

[`adaptive_image_routing.md`](adaptive_image_routing.md) is the normative v1 policy.
Typed request, bundle/lifecycle, artifact-inventory, qualification-report,
evidence-activation, eligibility-observation, and SelectionDecision validators
implement record interface contracts. The bounded bundle-registry loader validates
only the packaged SSOT or operator-asserted workspace catalog. A separate pure
selector evaluates validated in-memory inputs without I/O. The Experimental MCP-only
`recommend_image_pipeline` service adds bounded read-only orchestration over those
components. The default packaged registry is empty and therefore abstains. The
paired `process_images` service adds dry-run-by-default pinned revalidation and an
explicit bounded execution/publication gate. The packaged runner maps are empty,
so it cannot currently run a real model. Neither service provides a model adapter,
and their presence is not model availability or qualification evidence.

## Experimental

- backend parity and benchmark orchestration
- YOLOZU-synthgen intake and handoff
- macOS / MPS evaluation paths
- TTA

These can be useful in production-oriented work, but they still need environment-specific qualification and should be treated as capability-dependent.

## Research

- continual learning
- self-distillation
- TTT
- Hessian refinement

These areas are supported for reproducible experimentation, but they are not the first production lane for most adopters.
Start from [`research_lanes.md`](research_lanes.md) so the stable evaluation result and the opt-in research result stay separate.

## Recommended adoption order

1. Start with prediction validation/evaluation on CPU.
2. Adopt the predictions interface contract in your export path.
3. Add repo smoke / `doctor` checks to CI.
4. Qualify experimental paths only where they are needed.
5. Keep continual learning, TTT, and Hessian refinement behind explicit evaluation or operator review gates.

## How this maps to the manifest

Every tool entry in `tools/manifest.json` and the packaged `yolozu/data/manifest/tools_manifest.json` carries an entrypoint-level `maturity` field. A Stable entrypoint can expose opt-in Experimental or Research sub-lanes, so do not infer the maturity of every subcommand or flag from the parent entry alone. Capability-specific matrices and research-lane docs provide the narrower boundary.

The dated mapping from capability claims to implementation, CLI, manifest, packaged copy, docs, tests, and evidence is in [`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md).

## Related docs

- [`../README.md`](../README.md)
- [`README.md`](README.md)
- [`evaluation_protocol_template.md`](evaluation_protocol_template.md)
- [`ssot_capability_coverage_audit.md`](ssot_capability_coverage_audit.md)
- [`predictions_schema.md`](predictions_schema.md)
- [`external_inference.md`](external_inference.md)
- [`install.md`](install.md)
- [`research_lanes.md`](research_lanes.md)
- [`continual_learning.md`](continual_learning.md)
