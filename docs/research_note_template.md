# Research Note Template

Use this template when a research lane changes, adapts, refines, or analyzes an already evaluated YOLOZU artifact.

The goal is to keep the stable evaluation result and the research result separate. Do not rewrite the stable metric as if the research lane were the default product result.

## YAML Front Matter

```yaml
schema_version: 1
kind: yolozu_research_note
title: "<short title>"
lane: "ttt | continual | hessian | distillation | synthgen"
status: "draft | review | accepted | rejected"
stable_baseline_artifact: "reports/baseline_eval.json"
research_output_artifact: "reports/research_output.json"
research_report_artifact: "reports/research_report.json"
dataset:
  name: "<dataset or fixture>"
  split: "<split>"
  max_images: null
method:
  name: "<method>"
  parameters_artifact: "reports/plan.json"
metrics:
  stable:
    map50: null
    map50_95: null
  research:
    map50: null
    map50_95: null
  delta:
    map50: null
    map50_95: null
latency_overhead:
  baseline_mean_seconds: null
  research_mean_seconds: null
  delta_mean_seconds: null
rollback:
  status: "not_applicable | none | review_required | rolled_back"
  counters: {}
promotion_gate:
  decision: "promote | review | hold | review_required"
  reason: "<why>"
environment:
  yolozu_version: "<version>"
  python: "<version>"
  device: "cpu | cuda | mps"
limitations:
  - "<known limitation>"
```

## Markdown Body

### Stable Baseline

- Artifact:
- Dataset / split:
- Evaluator:
- Stable metrics:

### Research Transformation

- Lane:
- Method:
- Parameters / plan artifact:
- Research output artifact:
- Research report artifact:

### Delta

Keep stable and research metrics in separate columns.

| Metric | Stable | Research | Delta |
|---|---:|---:|---:|
| map50 |  |  |  |
| map50_95 |  |  |  |

### Cost And Rollback

- Latency overhead:
- Rollback / reset behavior:
- Guardrail or promotion-gate status:

### Decision

- Promotion gate decision:
- Required review:
- Next command:

### Limitations

- 

## Checklist

- [ ] Stable baseline artifact is named and remains unchanged.
- [ ] Research output artifact is separate from the stable baseline.
- [ ] Delta metrics are shown separately from stable metrics.
- [ ] Latency overhead is reported or marked `not_measured`.
- [ ] Rollback/reset behavior is reported or marked `not_applicable`.
- [ ] Promotion gate decision is explicit.
- [ ] Environment metadata is captured.
- [ ] Limitations are written in the note, not only in chat.
