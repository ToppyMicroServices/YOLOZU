# SDFT-Style Confirmatory Qualification — 2026-07-30

Status: **non-zero task metrics measured and independently reproduced; one of
three preregistered seed gates failed; Research; efficacy not established**.

## Protocol

The fixed confirmatory spec is
`configs/continual/sdft_coco128_blur_confirmatory_qualification.json`.
It compares naive sequential fine-tuning with YOLOZU's detector-specific
checkpoint-distillation lane on COCO128 and a deterministic Gaussian-blur
target:

- seeds: 44, 55, 66;
- initial training: 10 epochs, at most 64 optimizer steps;
- continual training: 20 steps per task, batch size 2, image size 64, CPU;
- evaluation: real `pycocotools` COCOeval mAP50:95 over 128 images;
- identical data, order, initial checkpoint, and budget within each seed.

The thresholds were fixed before the confirmatory run:

- source and target score: at least `1e-6`;
- old-task SDFT-minus-naive delta: strictly greater than zero;
- new-task SDFT-minus-naive delta: at least `-1e-6`;
- every seed must pass.

## Observed result

| Seed | Source score | Target score | Old-task delta | New-task delta | Gate |
|---:|---:|---:|---:|---:|---|
| 44 | 0.0000017431321 | 0.0000011156045 | +0.0000004880770 | +0.0000004081646 | pass |
| 55 | 0.0000058983244 | 0.0000057238210 | +0.0000017560120 | +0.0000015719495 | pass |
| 66 | 0.0000047108945 | 0.0000047111726 | -0.0000007100044 | -0.0000007190602 | fail |

All source and target scores were non-zero. Seed 66 failed the strict
old-task improvement gate, so the aggregate confirmatory gate failed.

## Reproduction boundary

The independent role reruns all training/evaluation paths into a fresh output
root and compares protocol identity, metric directions, and gate outcomes with
the primary `qualification_summary.json`. Reproduction and efficacy are
separate fields: matching a negative gate outcome does not turn it into
efficacy evidence.

Both final runs used source commit
`aefd28202d09b56f78cf825ae8c3b4354c1cb127` with no tracked changes.
The independent summary records `reproduced=true`,
`direction_and_gate_outcome_match=true`, and
`efficacy_supported=false`.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| Primary archive | `59234100efbfe6f4dd96b37ad6774b03da42f9045a0ab963afeff63465132e85` |
| Primary checksum manifest | `81027a9453356555affc834998092f7927e2b3a967e7c428916a99d46d6d440a` |
| Primary summary | `c3f67c56dde4c41f34180599cb97eac769b6b18f43d34719afdac1d81d8a1757` |
| Independent archive | `0dfe2f62f4f943595af30a5bba60aa79d76d90af7f6a473be04295c51c352bcf` |
| Independent checksum manifest | `56c2ecbd74ed17917aef2739c7c54dfe876bdb5aa83912134dcf4462f485fab7` |
| Independent summary | `09eaff8138a6e485e8d62a9a3ac73a6cd0a8325da03bbb900d7fac142c33bad5` |

The tracked machine-readable summaries are
[`primary`](sdft_confirmatory_primary_2026-07-30.json) and
[`independent`](sdft_confirmatory_independent_2026-07-30.json).

## Decision

This run removes the earlier all-zero measurement blocker, but it does not
support a seed-robust retention/adaptation advantage. The decision remains
`hold`, efficacy remains `not_established`, and the lane remains Research.
