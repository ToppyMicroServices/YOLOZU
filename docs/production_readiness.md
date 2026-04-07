# Production Readiness

This page separates the main production lane from the experimental and research lanes.

## What YOLOZU is primarily for

YOLOZU's primary role is an evaluation layer built around one stable predictions interface contract:

- validate wrapped `predictions.json`
- evaluate predictions reproducibly
- compare outputs across frameworks and runtimes

If your team already has inference outputs and wants fair evaluation without rewriting the whole pipeline, this is the main production path.

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

## Research-oriented

- continual learning
- self-distillation
- TTT
- Hessian refinement

These areas are supported for reproducible experimentation, but they are not the first production lane for most adopters.

## Recommended adoption order

1. Start with prediction validation/evaluation on CPU.
2. Adopt the predictions interface contract in your export path.
3. Add repo smoke / `doctor` checks to CI.
4. Qualify experimental paths only where they are needed.
5. Treat continual learning and TTT as separate research tracks until they have their own promotion criteria.

## Related docs

- [`README.md`](../README.md)
- [`predictions_schema.md`](predictions_schema.md)
- [`external_inference.md`](external_inference.md)
- [`install.md`](install.md)
- [`continual_learning.md`](continual_learning.md)
