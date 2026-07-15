# SSOT Capability Coverage Audit

Audit date: 2026-07-15

This audit maps the capability claims in [Production Readiness](production_readiness.md) and
[YOLOZU Spec](yolozu_spec.md) to implementation, CLI, manifest, packaged metadata, documentation,
tests, and reproducible evidence. Implementation presence is not evidence of production maturity.
When the source documents do not declare a maturity boundary, this audit records it as undeclared
instead of promoting it by inference.

## Coverage matrix

| Capability | Maturity | Implementation | CLI | Manifest / packaged copy | Docs | Tests / evidence | Result / follow-up |
|---|---|---|---|---|---|---|---|
| Predictions validation/evaluation | Stable | `yolozu/predictions/`, `yolozu/eval/` | `yolozu validate`, `eval-coco`, `eval-instance-seg`, `parity` | `validate_predictions`, `eval_coco`, `eval_instance_segmentation`, and `yolozu`; source and packaged manifests match | `predictions_schema.md`, `external_inference.md` | `tests/test_predictions.py`, `tests/test_eval_suite_determinism.py`, `data/smoke/predictions/predictions_dummy.json` | Aligned. Wrapper schema is v1 and entry schema is v2. |
| Dataset I/O and mask-only label derivation | Undeclared as a standalone capability | `yolozu/datasets/dataset.py`, `dataset_contract.py`, `dataset_validator.py` | `yolozu validate dataset`, `export-dataset`, `import` | Covered by stable `yolozu` and dataset preparation entries; no separate mask-derivation command | `yolozu_spec.md`, `dataset_contract.md` | `tests/test_dataset_contract.py`, `tests/test_dataset_mask_labels.py`, `data/smoke/` | Implemented and tested; capability maturity decision is tracked by `YOLOZU-ll2.15`. |
| RT-DETR pose reference trainer | Stable reference lane | `rtdetr_pose/rtdetr_pose/`, `rtdetr_pose/tools/train_minimal.py` | `yolozu train` | Stable `yolozu` entry; packaged copy matches | `training_backend_interface.md`, `training_capability_matrix.md`, `run_contract.md` | `tests/test_rtdetr_pose_adapter.py`, `rtdetr_pose/tests/test_train_minimal_integration.py` | Aligned. Runtime/GPU qualification remains environment-specific. |
| Reference backbone and neck boundary | Stable within the reference trainer | `rtdetr_pose/rtdetr_pose/backbone_interface.py`, `models/backbones/` | `yolozu train --config ...` | Covered by stable `yolozu`; no independent public command | `yolozu_spec.md`, `training_backend_interface.md` | `tests/test_backbone_shapes.py`, `tests/test_rtdetr_backbone_neck_parity.py` | Aligned at the adapter boundary; no repository-wide model-family claim. |
| Inference constraints and template gating | Undeclared as a standalone capability | `yolozu/geometry/constraints.py`, `template_verification.py`, `yolozu/inference/inference.py` | Internal inference path; research `tune_gate_weights` helper for gate tuning | `tune_gate_weights` is research; no standalone maturity for inference utilities | `yolozu_spec.md`, `gate_weight_tuning.md`, `repo_map.md` | `tests/test_gates_constraints.py`, `tests/test_inference_constraints.py`, `tests/test_template_verification.py` | Implemented and tested; public maturity boundary is tracked by `YOLOZU-ll2.15`. |
| Backend parity / benchmark orchestration | Experimental | `yolozu/eval/benchmark_mode.py`, `tools/benchmark_model.py`, parity tools | `yolozu benchmark`, `yolozu parity` | Experimental benchmark/parity entries; source and packaged manifests match | `benchmark_mode.md`, generated `benchmark_support_matrix.md` | `tests/test_benchmark_model_tool.py`, `tests/test_benchmark_support_matrix_generator.py`; 7 formats x 7 tasks | Canonical matrix is current. OpenVINO wording drift is tracked by `YOLOZU-ll2.12`; classification/OBB parity gaps by `YOLOZU-ll2.11`. |
| YOLOZU-synthgen handoff | Experimental | `yolozu/contracts/synthgen.py`, SynthGen datasets, eval tools | Manifested validation, smoke, render, and eval tools | SynthGen entries are experimental; source and packaged manifests match | `synthgen_contract.md`, `synthgen_repo_integration.md` | `tests/test_contract_synthgen.py`, `data/smoke/synthgen_minishard/` | Aligned for local intake/eval; external generator qualification remains explicit. |
| macOS / MPS paths | Experimental and conditional | `yolozu/core/doctor.py`, reference training device checks | `yolozu doctor`, qualified `yolozu train` path | Platform fields are declared per entry; no blanket MPS claim | `install.md`, `doctor_diagnostics.md`, `training_capability_matrix.md` | `tests/test_mps_smoke.py`; hardware-dependent probe skips when MPS is unavailable | Aligned as conditional. Public fresh-install macOS/Linux evidence is tracked by `YOLOZU-ll2.3`. |
| Training platform: external training bridges | Stable YOLOX sub-lane; other lanes experimental | `yolozu/training/platform.py`, `tools/support_external_training.py` | `yolozu train --external-backend ...`, `train-orchestrate` | `support_external_training` is experimental while the backend matrix labels YOLOX Stable; other external rows are Experimental | `training_capability_matrix.md`, `training_orchestration.md` | `tests/test_support_external_training_tool.py`, `tests/test_training_platform.py`, `tests/test_training_family_recipes.py` | Implementation and backend matrix agree. Tool-level versus sub-lane maturity rules are tracked by `YOLOZU-ll2.15`. |
| Continual learning / self-distillation | Research | `yolozu/training/continual_regularizers.py`, `distillation.py`, research tools | Manifested research helpers; not the default stable lane | `continual_decide`, `distill_predictions`, and related tools are research | `research_lanes.md`, `continual_learning.md`, `distillation.md` | `tests/test_continual_decide_tool.py`, `tests/test_distill_predictions_cli.py` | Aligned and kept behind evaluation or operator review boundaries. |
| TTA and TTT | TTT is Research; standalone TTA boundary is undeclared | `yolozu/tta/`, `tools/export_predictions.py` | Opt-in `--tta` and `--ttt` on stable export entrypoints; research comparison tools | `export_predictions` is stable at entrypoint level; dedicated TTT tools are research | `tta_support_matrix.md`, `ttt_protocol.md`, `research_lanes.md` | `tests/test_tta.py`, `tests/test_export_predictions_ttt_cli.py`, `tests/test_run_ttt_compare_tool.py`, `docs/assets/ttt_*` | No promotion inferred from the stable parent command. Mixed-lane classification is tracked by `YOLOZU-ll2.15`. |
| Hessian refinement | Research | `yolozu/calibration/hessian_solver.py`, `tools/refine_predictions_hessian.py` | `refine_predictions_hessian` tool | Research manifest entry; packaged copy matches | `hessian_solver.md`, `research_lanes.md` | `tests/test_hessian_solver.py`, `tests/test_refine_predictions_hessian_cli.py` | Aligned as an opt-in offline/local correction path. |
| Installed CLI and tool registry | Mixed; maturity is per entrypoint, with narrower sub-lane rules | `yolozu/cli.py`, `cli_entry.py`, `cli_commands.py` | 28 canonical commands/aliases in current top-level help | 113 entries: 57 stable, 43 experimental, 13 research; source and packaged copies match | `generated/cli_reference.md`, `tools_index.md`, `manifest_declarative_spec.md` | 105 Python entrypoints scanned with zero help/manifest flag gaps; manual audit passes | Generated reference drift was found and is corrected in this audit; PR and pre-push gates now cover it. |

## Confirmed checks

- `tools/manifest.json` and `yolozu/data/manifest/tools_manifest.json` are byte-identical.
- Strict manifest validation passes for all 113 entries.
- Per-entrypoint help audit scans 105 Python tools with zero execution errors and zero missing flags.
- Manual CLI drift audit passes for the current 28-command/alias top-level surface.
- Public docs example audit passes 65 shell examples.
- The generated benchmark support matrix is current for 7 formats, 7 tasks, and 49 rows.
- Targeted manifest, training, benchmark, predictions, evaluation, research, SynthGen, and MPS tests pass; the only skip is the hardware-dependent MPS probe when MPS is unavailable.

## Confirmed gaps and disposition

| Gap | Evidence | Disposition |
|---|---|---|
| Generated CLI reference was stale after a manifest summary change | `tests.test_generated_cli_reference` failed on main before regeneration | Corrected in this audit; added to PR-focused CI and local pre-push tests. |
| Benchmark OpenVINO state conflicts across docs | `benchmark_support_matrix.md` says conditional real; `benchmark_backend_runtime_matrix.md` says planned / conditional; implementation tests exercise real orchestration | Reconcile from current behavior under `YOLOZU-ll2.12`. |
| Classification and OBB artifact lanes lack parity artifacts | Canonical benchmark support matrix reports parity as skipped | Implement evidence-backed parity under `YOLOZU-ll2.11`. |
| Several spec-only or mixed-entrypoint capabilities lack an unambiguous maturity boundary | Dataset/mask/constraint/template/TTA claims are not standalone rows in production readiness; stable export exposes research TTT flags | Decide and synchronize boundaries under `YOLOZU-ll2.15`. |
| Public fresh-install proof is incomplete across the supported OS/Python matrix | Current checks cover source checkout and one clean local install, not the full macOS/Linux acceptance matrix | Execute and record under `YOLOZU-ll2.3`. |
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
python3 -m unittest \
  tests.test_ssot_capability_coverage \
  tests.test_generated_cli_reference \
  tests.test_manifest_tool_coverage \
  tests.test_packaged_tools_manifest \
  tests.test_benchmark_support_matrix_generator
```
