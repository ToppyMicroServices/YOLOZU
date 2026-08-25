# SSOT Capability Coverage Audit

Audit date: 2026-07-26
Benchmark reconciliation update: 2026-07-23
TTT evidence update: 2026-07-27
SynthGen evidence update: 2026-07-28
Non-TTT artifact research evidence update: 2026-07-28
BOP object 6DoF evidence update: 2026-07-30
SDFT-style continual reproduction update: 2026-07-30
External runtime qualification update: 2026-07-30
TTT detector-response and SDFT preregistration update: 2026-08-01
Adaptive local-vision roadmap projection update: 2026-08-22
Adaptive live environment profile update: 2026-08-25
Adaptive bundle registry loader update: 2026-08-25
Adaptive managed-output transaction update: 2026-08-25
Adaptive qualification foundation update: 2026-08-25
Adaptive reviewed evidence activation update: 2026-08-25
Adaptive deterministic selector update: 2026-08-25
Adaptive pinned processing update: 2026-08-26
Adaptive installed-artifact verification update: 2026-08-26
Adaptive baseline bundle registration update: 2026-08-26
Adaptive monitored-source scout update: 2026-08-26
Adaptive dormant support-profile governance update: 2026-08-26

The corresponding 30-run diagnostic artifacts and checkpoints are fixed in the
[2026-07-27 prerelease](https://github.com/ToppyMicroServices/YOLOZU/releases/tag/ttt-evidence-2026-07-27)
(archive SHA-256
`bb200d0c0a36447f0b6ed262a56ee09bef44ded8f10c55673243080fe1054068`).
The complete 30-cell TTT diagnostic was independently reproduced in a clean
Python 3.12 environment with zero semantic differences. Efficacy remains open
because every improvement delta was zero.

This audit maps the capability claims in [Production Readiness](production_readiness.md) and
[YOLOZU Spec](yolozu_spec.md) to implementation, CLI, manifest, packaged metadata, documentation,
tests, and reproducible evidence. Implementation presence is not evidence of production maturity.
When the source documents do not declare a maturity boundary, this audit records it as undeclared
instead of promoting it by inference.

## Coverage matrix

| Capability | Maturity | Implementation | CLI | Manifest / packaged copy | Docs | Tests / evidence | Result / follow-up |
|---|---|---|---|---|---|---|---|
| Predictions validation/evaluation | Stable | `yolozu/api.py`, `yolozu/predictions/`, `yolozu/eval/` | `yolozu validate`, `eval-coco`, `eval-instance-seg`, `parity` | `validate_predictions`, `eval_coco`, `eval_instance_segmentation`, and `yolozu`; source and packaged manifests/schemas match | `python_api.md`, `predictions_schema.md`, `external_inference.md` | `tests/test_public_api.py`, `tests/test_eval_cli_guardrails.py`, `tests/test_predictions.py`, `data/smoke/predictions/predictions_dummy.json` | Aligned. `eval-coco` is strict by default, explicit repair records warnings, bounded subsets count/exclude known unselected predictions, and the typed in-process API ships `py.typed`. |
| Environment-qualified adaptive local vision | Experimental MCP recommendation and pinned processing; candidate-screening, qualification, reviewed activation, dormant support-profile review, pure-selection, and monitored-source foundations implemented | Strict typed validators, live environment profiling, trusted registry loading, non-executing candidate screening, pinned input/artifact readers, bounded qualification/publication, reviewed evidence activation, dormant support-profile review, pure selection, read-only recommendation, dry-run-by-default pinned processing, and a bounded official-source scout exist. Three model-zoo entries remain non-promoted Candidate metadata with unbound execution. Scout and screening output cannot load as bundle registry state. A reviewed support-profile set remains dormant until a separate lifecycle assignment. No registered adaptive model runner, model adapter, or real execution claim is made. | `yolozu scout-algorithms` collects only with `--collect`; screening keeps mandatory unknowns on hold; qualification emits an unactivated report; evidence activation and dormant support-profile review require exact review gates plus `--approve`; MCP recommendation abstains by default; MCP processing defaults to no-write dry-run and reprojects the lifecycle-pinned historical support set before runner resolution | Candidate-only registry/lifecycle SSOT, empty screening/support-profile/evidence streams, support-profile proposal/record schemas, screening/scout schemas, roadmap, synchronized manifests, and generated MCP reference are packaged | `algorithm_intake/README.md`, `adaptive_image_routing.md`, `schema_governance.md`, `production_readiness.md`, `reports/adaptive_support_profile_governance_2026-08-26.md`, `reports/adaptive_algorithm_scout_foundation_2026-08-26.md`, `reports/adaptive_candidate_screening_foundation_2026-08-26.md`, `reports/adaptive_baseline_bundle_registry_2026-08-26.md`, `reports/adaptive_routing_installed_verification_2026-08-26.md`, `reports/adaptive_vision_roadmap.md` | `tests/test_adaptive_support_profile_governance.py` covers complete-set review, stale/private/invalid input, atomic failure/readback, historical snapshots, provider trust/statuses, recommendation wiring, and execution-time tamper; routing/installed tests retain the abstention and package boundaries | Beads `YOLOZU-ll2.81` is the live planning source. Fixtures are interface tests, not performance evidence. The three records remain Candidate; screening, support-profile, and evidence streams and runner maps remain empty; default recommendation abstains with `maturity_disallowed`. No real bundle qualification, support, execution, or human adoption is claimed. |
| Dataset I/O and mask-only label derivation | Explicitly deferred as standalone capabilities | `yolozu/datasets/dataset.py`, `dataset_contract.py`, `dataset_validator.py`, `tools/make_subset_dataset.py`, `rtdetr_pose/rtdetr_pose/train_dataset.py` | `yolozu validate dataset`, `doctor train-dataset`, `export-dataset`, `import`, sidecar-safe subset helper | Covered by stable `yolozu` and dataset preparation entries; no separate mask-derivation command | `yolozu_spec.md`, `dataset_contract.md`, `dataset_processing_matrix.md`, `training_inference_export.md`, `reports/dataset_preflight_2026-07-27.md`, `reports/dataset_roundtrip_2026-07-27.md` | Dataset/doctor/export/subset tests; `data/smoke/`, `data/coco128/`, `data/conversion_tiny_coco/`, `data/real_multitask_fewshot/` | Empty splits fail closed; doctor and validator share strict checks; installed-wheel COCO round trips preserve counts/classes and bbox geometry within the recorded no-clipping tolerance. Subsets retain mask/depth/keypoint/object-pose sidecars and variable-resolution aux arrays collate after task-appropriate resizing. Real keypoint/semantic upstream qualification and fixture license review remain explicitly unqualified; standalone maturity remains deferred. |
| RT-DETR pose reference trainer | Stable reference lane | `rtdetr_pose/rtdetr_pose/`, `rtdetr_pose/tools/train_minimal.py` | `yolozu train` | Stable `yolozu` entry; packaged copy matches | `training_backend_interface.md`, `training_capability_matrix.md`, `run_contract.md` | `tests/test_rtdetr_pose_adapter.py`, `rtdetr_pose/tests/test_train_minimal_integration.py` | Aligned. Runtime/GPU qualification remains environment-specific. |
| Reference backbone and neck boundary | Stable within the reference trainer | `rtdetr_pose/rtdetr_pose/backbone_interface.py`, `models/backbones/` | `yolozu train --config ...` | Covered by stable `yolozu`; no independent public command | `yolozu_spec.md`, `training_backend_interface.md` | `tests/test_backbone_shapes.py`, `tests/test_rtdetr_backbone_neck_parity.py` | Aligned at the adapter boundary; no repository-wide model-family claim. |
| Inference constraints and template gating | Explicitly deferred as standalone capabilities | `yolozu/geometry/constraints.py`, `template_verification.py`, `yolozu/inference/inference.py` | Internal inference path; research `tune_gate_weights` helper for gate tuning | `tune_gate_weights` is research; no standalone maturity for inference utilities | `yolozu_spec.md`, `gate_weight_tuning.md`, `repo_map.md` | `tests/test_gates_constraints.py`, `tests/test_inference_constraints.py`, `tests/test_template_verification.py` | Implemented and tested; qualification belongs to the consuming adapter, model, and protocol. |
| Backend parity / benchmark orchestration | Experimental | `yolozu/eval/benchmark_mode.py`, `tools/benchmark_model.py`, parity tools | `yolozu benchmark`, `yolozu parity` | Experimental benchmark/parity entries; source and packaged manifests match | `benchmark_mode.md`, generated `benchmark_support_matrix.md` | `tests/test_benchmark_model_tool.py`, `tests/test_benchmark_support_matrix_generator.py`; 7 formats x 7 tasks | Canonical matrix and conditional OpenVINO wording are current; classification/OBB artifact parity now records task-specific metrics, thresholds, source checksums, and run provenance under `YOLOZU-ll2.11`. |
| YOLOZU-synthgen handoff | Experimental | `yolozu/contracts/synthgen.py`, SynthGen datasets, eval tools | `smoke_synthgen --synthgen-repo` generates and qualifies a fresh handoff; existing validation/render/eval tools remain available | SynthGen entries are experimental; source and packaged manifests match | `synthgen_contract.md`, `synthgen_repo_integration.md`, `reports/synthgen_handoff_2026-07-28.md` | `tests/test_contract_synthgen.py`; five-sample Open3D run; strict QA, loader, overlay, and oracle interface self-check | `render_only` and deterministic-stub `bg_only_inpaint` are locally qualified. External providers, `appearance_only_conditioned`, `full_regen`, and downstream model quality remain unqualified. |
| BOP T-LESS object 6DoF | Research | `tools/download_bop_dataset.py`, `tools/prepare_bop_yolozu.py`, `tools/export_bop19_rtdetr_pose.py`, `tools/summarize_bop19_pose_evidence.py`, BOP toolkit | Safe downloader/converter plus diagnostic and official BOP19 target-conditioned evaluation | Research entries with source/packaged interface contracts and diagnostic/official qualification schemas | `bop_tless_protocol.md`, both 2026-07-30 BOP evidence reports, `manual/chapters/16_depth_6dof_symmetry.tex` | Official archive/toolkit hashes, strict GT counts, three seeds, full target coverage, official and task-native metrics, and independent semantic comparison | The official protocol gap is closed without rewriting the earlier diagnostic. The result remains `hold`/`not_established`; this is rigid-object pose and human 3D skeleton pose is unsupported. |
| macOS / MPS paths | Experimental and conditional | `yolozu/core/doctor.py`, reference training device checks | `yolozu doctor`, qualified `yolozu train` path | Platform fields are declared per entry; no blanket MPS claim | `install.md`, `doctor_diagnostics.md`, `training_capability_matrix.md` | `tests/test_mps_smoke.py`; hardware-dependent probe skips when MPS is unavailable | Aligned as conditional. Public fresh-install macOS/Linux evidence is tracked by `YOLOZU-ll2.3`. |
| Training platform: external and real-image fine-tuning | Stable YOLOX interface sub-lane; other lanes Experimental; execution environment-qualified | `yolozu/training/platform.py`, external training helpers, runtime fixture preparation, compatible-host workflow | `qualify_finetune_lanes`, `run_external_runtime_gpu_qualification.sh`, and backend helpers remain separately callable | Experimental qualification entries and installed/compatible-host schemas; source and packaged manifests match | `training_capability_matrix.md`, `external_finetune_smoke.md`, both external-runtime evidence reports | Three independently repeated macOS-available lanes plus two same-commit Linux/CUDA runs completing YOLOX, MMDetection, MMPose, MMSeg, and TAO training | Config projection is not training, swallowed traceback/zero-exit is rejected, and resource/checkpoint/report hashes are recorded. All compatible-host fail-closed gates and structural handoffs reproduced, but checkpoint bytes were not deterministic and runtime availability does not establish quality; maturity remains Experimental. |
| Continual learning / self-distillation | Research | `rtdetr_pose/tools/train_continual.py`, `tools/eval_continual.py`, `tools/qualify_sdft_continual.py`, response selection with abstention, replay, continual regularizers | `qualify_sdft_continual` runs fixed primary or independent sequences; train/eval/decision helpers remain separately callable | Research entry and extended `sdft_continual_qualification_json` schema; source and packaged manifests match | `research_lanes.md`, `continual_learning.md`, `distillation.md`, both SDFT evidence reports | Original release-addressable all-zero bundle, a three-seed non-zero confirmatory run, independent semantic comparison, and a prospective four-group response/replay preregistration | All confirmatory seeds were measurable, but only 2/3 passed the prior gate. The unused-seed response/replay ablation freezes minimum selection and maximum abstention, but is not run; decision remains `hold` and efficacy `not_established`. |
| TTA and TTT | TTA is Experimental; TTT is Research | `yolozu/tta/`, `yolozu/response_selection.py`, `tools/export_predictions.py`, `tools/run_ttt_evidence_suite.py` | Opt-in `--tta` has default postprocess and `rtdetr_pose` model-branch modes; `--method detector_response` is the concise selected-foreground compare; opt-in `--ttt` updates parameters or explicitly abstains below the minimum selection count | `export_predictions` is stable at entrypoint level; optional TTA/TTT features retain narrower maturity | `tta_support_matrix.md`, `training_inference_export.md`, `ttt_protocol.md`, `research_lanes.md` | Independently reproduced zero-delta matrix plus a one-checkpoint 10-image detection-native diagnostic with non-zero clean/shifted deltas; abstention tests assert zero backward, optimizer steps, and parameter drift | Detection-native class/box consistency produced a bounded positive observation with no guard stops. Abstention is structurally verified, but neither efficacy nor an optimal threshold is established; maturity remains Research. |
| Hessian refinement | Research | `yolozu/calibration/hessian_solver.py`, `tools/refine_predictions_hessian.py` | `refine_predictions_hessian`, `qualify_artifact_research` | Research manifest entries and schemas; packaged manifest matches | `hessian_solver.md`, `research_lanes.md`, `reports/artifact_research_evidence_2026-07-28.md` | `tests/test_hessian_solver.py`, `tests/test_refine_predictions_hessian_cli.py`, `tests/test_qualify_artifact_research_cli.py`; three deterministic COCO128 repetitions | Wrapped output now satisfies the predictions interface contract and reports measured latency/hashes. All 1,280 detections per repetition were `no_signal`, metrics were unchanged, and promotion remains `hold`. |
| Searchable web onboarding | Stable generated documentation over per-capability maturity labels | `tools/generate_web_docs.py`, `docs/web_docs_content.json` | Self-contained strict CLI journey plus stable typed Python example | Stable `generate_web_docs` entry; source and packaged manifests match | `web_docs_plan.md`, generated `web_docs/start.html`, `python_api.md` | `tests/test_web_docs_generation.py`, `tests/test_web_docs_candidate_wheel.py`; adversarial path/URL/output checks plus candidate wheel outside checkout through the installed console script | Aligned. Sources are repository-confined, replacement requires owned provenance, all referenced SSOT files are hashed, and CI fails unless the canonical path completes real COCOeval. The dependency-free dry run remains an explicitly non-metric fallback outside that gate. |
| Installed CLI and mixed-lane entrypoints | Mixed; maturity is per entrypoint, with narrower sub-lane rules | `yolozu/cli.py`, `cli_entry.py`, `cli_commands.py` | 32 canonical commands/aliases in current top-level help | 137 entries: 60 stable, 56 experimental, 21 research; source and packaged copies match | `generated/cli_reference.md`, `tools_index.md`, `manifest_declarative_spec.md` | Per-entrypoint help/manifest audit and manual audit are required quality gates | Stable parent maturity is explicitly non-transitive; generated reference and manifest descriptions repeat that boundary. |

## Confirmed checks

- `tools/manifest.json` and `yolozu/data/manifest/tools_manifest.json` are byte-identical.
- Strict manifest validation passes for all 137 entries.
- Per-entrypoint help audit scans the current declared Python tool set with zero execution errors and zero missing flags.
- Manual CLI drift audit passes for the current 32-command/alias top-level surface.
- Public docs example audit passes 114 shell examples.
- The generated benchmark support matrix is current for 7 formats, 7 tasks, and 49 rows.
- The generated web-docs bundle is current for 137 tools and 50 JSON Schemas.
- Public PyPI `yolozu==4.5.1` completed the fresh-install stable lane in all 10
  Linux/macOS jobs for Python 3.10 through 3.14 in
  [workflow run 29421807474](https://github.com/ToppyMicroServices/YOLOZU/actions/runs/29421807474).
  This records only the tested matrix and does not establish evidence for other
  Python versions or platforms; package metadata separately declares
  `Python >=3.10`.
- Targeted manifest, training, benchmark, predictions, evaluation, research, SynthGen, and MPS tests pass; the only skip is the hardware-dependent MPS probe when MPS is unavailable.

## Confirmed gaps and disposition

| Gap | Evidence | Disposition |
|---|---|---|
| Generated CLI reference was stale after a manifest summary change | `tests.test_generated_cli_reference` failed on main before regeneration | Corrected in this audit; added to PR-focused CI and local pre-push tests. |
| Benchmark support wording drifted across surfaces | OpenVINO detect is conditional real while artifact-backed OpenVINO tasks bypass runtime checks; four artifact tasks omitted TorchScript/OpenVINO from report `support_level`; classification/OBB advertised parity as real when comparable despite always writing placeholders; ExecuTorch/OpenCV DNN reports were skipped while text surfaces or report notes called them planning/synthetic-only; strict mode omitted partial eval/parity failures | Corrected under `YOLOZU-ll2.12`; report runtime evidence now distinguishes required/checked runtime probes from artifact-backed lanes, and canonical/standalone OpenVINO CLI synchronization landed under `YOLOZU-ll2.23` with artifact-eval flag applicability under `YOLOZU-ll2.24`. |
| Explicit detect `artifact_eval` previously disagreed with executed work | A patched-runtime reproduction recorded `execution_mode=synthetic_planning_only` while invoking the normal backend prediction command | Corrected under `YOLOZU-ll2.28`: both CLI surfaces reject the combination before report, artifact, or backend writes. |
| Classification and OBB artifact lanes lacked parity artifacts | Earlier canonical benchmark support matrix reported parity as skipped | Resolved under `YOLOZU-ll2.11` with strict-JSON artifact comparisons, task-specific diagnostics, thresholds, and source/run provenance. |
| Spec-only and mixed-entrypoint maturity boundaries were ambiguous | Dataset/mask/constraint/template/TTA claims lacked standalone rows; stable export exposes research TTT flags | Resolved under `YOLOZU-ll2.15`: standalone maturity is deferred for dataset/mask/constraint/template, TTA is Experimental, TTT is Research, and parent maturity is non-transitive. |
| Public fresh-install proof was incomplete across the supported OS/Python matrix | Closed `YOLOZU-ll2.3` records 10/10 public-PyPI jobs on Linux/macOS for Python 3.10 through 3.14 in workflow run 29421807474 | Resolved under `YOLOZU-ll2.3`; keep future release verification scoped to the package's declared Python boundary. |
| Public value proposition still needs cross-surface verification | Repository, PyPI metadata, and company page require one evidence-bounded message | Align only after this audit under `YOLOZU-ll2.5`. |

## Non-gaps and boundaries

- Optional runtimes being absent is not a defect when the corresponding lane reports `skipped` and does not claim bundling.
- Tool implementation and passing unit tests do not automatically promote a capability to Stable.
- Planned runtime breadth does not justify adding an adapter without adopter evidence; that decision remains under `YOLOZU-ll2.13`.
- Raw aggregate flag comparisons across all subcommands are not used as evidence of drift; the per-entrypoint help audit is the applicable check.

## Re-run commands

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 tools/audit_manifest_inputs_vs_help.py --timeout 5
python3 tools/audit_manual_cli_drift.py --json
python3 tools/audit_docs_examples_drift.py --json
python3 tools/generate_benchmark_support_matrix.py --check --json
python3 tools/generate_adaptive_vision_roadmap.py --check --json
python3 tools/generate_web_docs.py --check --json
python3 -m unittest \
  tests.test_ssot_capability_coverage \
  tests.test_generated_cli_reference \
  tests.test_manifest_tool_coverage \
  tests.test_packaged_tools_manifest \
  tests.test_benchmark_support_matrix_generator
```
