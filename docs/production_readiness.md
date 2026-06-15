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
| Backend parity / benchmark orchestration | Experimental | Useful after environment-specific qualification; `keypoints`, `depth`, and `pose6d` currently qualify via artifact-backed real eval/parity lanes rather than end-to-end backend inference benchmarking | [`backend_parity_matrix.md`](backend_parity_matrix.md), [`benchmark_mode.md`](benchmark_mode.md), `manual/chapters/09_parity_bench_protocols.tex` |
| YOLOZU-synthgen handoff | Experimental | Intake/eval path is reproducible, but external generator handoff still needs qualification | [`synthgen_repo_integration.md`](synthgen_repo_integration.md), [`synthgen_contract.md`](synthgen_contract.md), `manual/chapters/21_synthgen_repo_integration.tex` |
| macOS / MPS paths | Experimental | Supported only when `torch.backends.mps.is_available()` is true; treat as qualification, not blanket readiness | [`install.md`](install.md), [`doctor_diagnostics.md`](doctor_diagnostics.md), [`continual_learning.md`](continual_learning.md) |
| Continual learning / self-distillation | Research | Use for governed experiments and promotion-gated workflows over evaluated artifacts, not as the first production lane | [`research_lanes.md`](research_lanes.md), [`continual_learning.md`](continual_learning.md), `manual/chapters/14_continual_learning.tex` |
| TTT | Research | Short-horizon inference adaptation over evaluated artifacts; do not treat as an automatic checkpoint-promotion path | [`research_lanes.md`](research_lanes.md), [`ttt_protocol.md`](ttt_protocol.md), `manual/chapters/15_ttt_tent_mim.tex` |
| Hessian refinement | Research | Offline/local post-inference correction path over evaluated artifacts | [`research_lanes.md`](research_lanes.md), [`hessian_solver.md`](hessian_solver.md), `manual/chapters/10_ttt_hessian.tex` |
| Training platform | Stable reference lane + qualified external lanes | RT-DETR pose reference trainer is the richest in-repo path and supports depth / pose6d training; external lanes now share a standardized external run bundle even when the backend-native trainer remains outside YOLOZU | [`training_backend_interface.md`](training_backend_interface.md), [`training_capability_matrix.md`](training_capability_matrix.md), [`training_orchestration.md`](training_orchestration.md) |

## Stable today

- predictions interface contract: wrapped `predictions.json` plus protocol-pinned `meta.export_settings`
- prediction validation and evaluation flows
- install / `doctor` / repo smoke path
- CPU-friendly demo and smoke paths

These are the areas to rely on first for production adoption.

## Experimental

- backend parity and benchmark orchestration
- YOLOZU-synthgen intake and handoff
- macOS / MPS evaluation paths

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

Every tool entry in `tools/manifest.json` and the packaged `yolozu/data/manifest/tools_manifest.json` carries a `maturity` field so agents and operators can tell whether a command belongs to the stable, experimental, or research lanes.

## Related docs

- [`../README.md`](../README.md)
- [`README.md`](README.md)
- [`predictions_schema.md`](predictions_schema.md)
- [`external_inference.md`](external_inference.md)
- [`install.md`](install.md)
- [`research_lanes.md`](research_lanes.md)
- [`continual_learning.md`](continual_learning.md)
