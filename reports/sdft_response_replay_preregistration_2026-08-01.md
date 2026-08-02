# SDFT response-selection + replay preregistration — amended 2026-08-02

Status: **prospectively registered; not executed; no result**.

The 2026-08-01 draft was amended on 2026-08-02, before any preregistered run,
to freeze the abstention threshold and gate requested after the draft. The
machine-readable amendment log is part of the protocol.

The frozen machine-readable protocol is
[`configs/continual/sdft_response_replay_preregistration.json`](../configs/continual/sdft_response_replay_preregistration.json).
It does not modify the prior 2026-07-30 confirmatory result.

## Fixed comparison

- Seeds: 88, 99, 111; all prior result/calibration seeds are excluded.
- Groups: naive, response-selected SDFT, replay-only, and combined.
- SDFT: reverse KL on selected foreground logits plus selected boxes; the final
  no-object class is excluded; confidence threshold 0.2; top-k 20.
- Abstention: response distillation contributes zero below 2 selected queries;
  an abstention ratio above 0.5 fails the execution gate.
- Replay: reservoir size 32, replay fraction 0.25, per-task cap 16.
- Evaluation: real COCOeval `map50_95`, 128 images per task.

## Fixed gates

The primary run fails its execution gate if response methods record no selected
foreground query, replay methods consume no old-task record on task 2, task-0
weights differ, the response abstention ratio exceeds 0.5, or data order/budget
differs. Efficacy gates retain the prior
non-zero source/target floors, strictly positive old-task improvement, and a
new-task non-degradation tolerance of `1e-6`.

Independent reproduction and an external benchmark remain required for any
promotion. Until execution, this document makes no before/after or efficacy
claim; the current lane decision remains `hold` / `not_established`.

## Future execution

```bash
./.venv/bin/python tools/qualify_sdft_continual.py \
  --spec configs/continual/sdft_response_replay_preregistration.json \
  --output-dir /tmp/yolozu-sdft-response-replay
```
