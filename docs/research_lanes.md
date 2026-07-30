# Research Lanes

YOLOZU's stable product lane is the evaluation layer: validate existing predictions, run evaluation, and compare reports.

Use the research lanes only after that stable path has produced an already evaluated artifact. Research commands may adapt, refine, distill, or analyze the artifact, but they should not replace the stable evaluation result or be presented as the default production path.

## Scope

Research lanes are opt-in workflows for controlled studies:

- continual learning and promotion-gated anti-forgetting experiments
- TTT / CTTA adaptation under fixed domain-shift protocols
- offline prediction distillation over existing `predictions.json` artifacts
- Hessian-based post-inference refinement
- research notes and paper-oriented evidence packages

These workflows are supported for reproducible experimentation. They still require explicit evaluation gates, cost/latency accounting, and operator review before any production promotion.

## Default Order

1. Produce or import `predictions.json`.
2. Validate the predictions interface contract.
3. Run the stable evaluator and save the report.
4. Start a research lane from that evaluated artifact.
5. Write a separate research report that points back to the stable evaluation result.

This keeps the stable lane and research lane readable side by side.

## Lane Map

| Lane | Use when | Primary docs | Required boundary |
|---|---|---|---|
| Continual learning | You are studying forgetting across task or domain sequences | [`continual_learning.md`](continual_learning.md) | Promotion requires `continual_eval.json` plus an explicit promotion decision report |
| TTT / CTTA | You are testing short-horizon inference adaptation under fixed shifts | [`ttt_protocol.md`](ttt_protocol.md), [`ttt_compare_boilerplates.md`](ttt_compare_boilerplates.md) | Report adaptation cost, reset policy, rollback behavior, and before/after metrics |
| Offline distillation | You want to blend teacher/student prediction artifacts before investing in training | [`distillation.md`](distillation.md) | Keep the distilled output separate from the original stable evaluation report |
| Hessian refinement | You are running offline/local post-inference correction studies | [`hessian_solver.md`](hessian_solver.md) | Treat the refined predictions as a new evaluated artifact with its own log |
| SynthGen research handoff | You are qualifying generated shard data or synthetic evaluation inputs | [`synthgen_repo_integration.md`](synthgen_repo_integration.md), [`synthgen_contract.md`](synthgen_contract.md) | Validate the SynthGen sample interface contract before using the shard |

## One-command artifact qualification

For the tracked three-seed naive-versus-checkpoint-distillation continual
diagnostic:

```bash
./.venv/bin/python tools/qualify_sdft_continual.py \
  --output-dir /tmp/yolozu-sdft-qualification
```

The command uses real COCOeval, an explicit initial-checkpoint FWT baseline,
identical data/order/budget checks, and a schema-defined JSON summary. It
refuses existing output and archive paths. The repository implementation is an
SDFT-style detector regularizer, not a faithful reproduction of the
demonstration-conditioned language-model SDFT algorithm. Its bounded local
domain shift remains Research.

The clean 2026-07-28 execution completed all three seeds, but all real-COCOeval
task scores and method deltas were zero. It is published as a measured negative
result with decision `hold` and efficacy `not_established`; see the
[evidence report](../reports/sdft_continual_evidence_2026-07-28.md) and
[release-addressable bundle](https://github.com/ToppyMicroServices/YOLOZU/releases/tag/sdft-evidence-2026-07-28).
The bundle was independently reproduced in a second Python/Torch environment
on 2026-07-30 with matching semantic results. The remaining blocker is the
all-zero retention/adaptation result, not reproducibility.

A stronger confirmatory spec then produced non-zero source and target scores
for seeds 44, 55, and 66. Seeds 44 and 55 passed the preregistered
retention/adaptation checks, while seed 66 failed the strict old-task
improvement threshold. An independent run reproduced the protocol, metric
directions, and gate outcomes. This removes the all-zero measurement blocker
but does not establish efficacy; the result remains `hold`. See
[`reports/sdft_confirmatory_evidence_2026-07-30.md`](../reports/sdft_confirmatory_evidence_2026-07-30.md).

For offline distillation and Hessian refinement, the shortest measured path is:

```bash
./.venv/bin/python tools/qualify_artifact_research.py \
  --output-dir /tmp/yolozu-artifact-research
```

The command refuses an existing output directory, confines student, teacher,
dataset, and config inputs to the repository, runs three deterministic
repetitions, and evaluates the stable baseline and every transformed artifact
with real COCOeval. The result is a hash-bound
`qualification_summary.json` shaped by
[`schemas/artifact_research_qualification.schema.json`](schemas/artifact_research_qualification.schema.json).

The 2026-07-28 checked-in evidence remains `hold` for both lanes. The
distillation teacher is a ten-image smoke fixture rather than an independently
inferred full-set teacher. COCO128 provides no depth/mask auxiliary signal for
the current Hessian offset objective, so that run is a measured negative
control. See
[`../reports/artifact_research_evidence_2026-07-28.md`](../reports/artifact_research_evidence_2026-07-28.md).

## Report Expectations

Every research result should say:

- which stable evaluation artifact it starts from
- which method changed or analyzed the artifact
- whether the output is promoted, held for review, or only a research note
- latency or compute overhead when the method runs at inference time
- rollback or reset behavior for adaptive methods
- schema and validation status for any new artifact

Research-lane machine-readable reports should carry a `research_report` object shaped by
[`schemas/research_lane_report.schema.json`](schemas/research_lane_report.schema.json):

- `stable_baseline_artifact`
- `research_output_artifact`
- `latency_overhead`
- `rollback`
- `promotion_gate`

If the report cannot provide those fields, keep the result in a local experiment log rather than presenting it as a stable YOLOZU result.

## DoD Gate

The research lane DoD is intentionally separate from the stable evaluation DoD:

- stable evaluation reports remain unchanged and do not embed `research_report`
- only research-lane artifacts carry `research_report`
- research workflows start from an evaluated input artifact
- research workflows write a separate research output artifact or research report
- TTT / continual / distillation / Hessian workflows are opt-in and not production defaults
- Hessian examples stay framed as offline analysis or controlled studies
- promotion gates decide whether a research result is promoted, reviewed, or held

## Related Entrypoints

- Research feature overview: [`learning_features.md`](learning_features.md)
- Research note template: [`research_note_template.md`](research_note_template.md)
- Production posture source of truth: [`production_readiness.md`](production_readiness.md)
- Tool registry and maturity labels: [`tools_index.md`](tools_index.md)
