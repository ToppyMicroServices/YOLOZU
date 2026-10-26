# Part-level multi-view inspection evaluation

Reviewed: 2026-10-26. Status: **proposal; not implemented in YOLOZU**.
Research intake: `YOLOZU-9x3e`. Implementation follow-up: `YOLOZU-yz67`.
Beads remains the source of truth for task status.

## Evidence motivating the proposal

Srivishnu Ethirajulu Krishnaraj and Rishikesh R. Chopade,
*Region-Aware Multi-View Inspection Using Deep Learning and Domain-Informed
Decision Fusion*, [arXiv:2610.29398v1](https://arxiv.org/abs/2610.29398v1),
posted October 21, 2026. This is an arXiv preprint, not a YOLOZU result.
The record lists DOI `10.48550/arXiv.2610.29398` as pending registration on the
review date; use the versioned arXiv URL as the reference.

The authors evaluated 380 fuel-injector needles: 276 conforming (OK) and
104 non-conforming (NOK). Each part had 18 seat and 18 shaft views. Comparisons
held both models, inference settings, thresholds and predictions fixed.

| Part-level result | Any-defect baseline | Complete decision layer |
|---|---:|---:|
| Correctly rejected NOK (TP) | 104 | 104 |
| Incorrectly accepted NOK (FN) | 0 | 0 |
| Incorrectly rejected OK (FP) | 159 | 30 |
| Correctly accepted OK (TN) | 117 | 246 |
| False-reject rate, FP / 276 | 57.6% | 10.9% |

Source: [Sections IV-A–IV-E and Table X](https://arxiv.org/html/2610.29398v1).
Seat rules operate within each frame before multi-view aggregation; they do
not count physical defects across views. Shaft classification uses a separate
multi-frame rule. Rule-group ablations are independent, not cumulative.

## Interpretation and limits

The reported change is in product acceptance decisions, not improved detection
weights or exhaustive instance recall. Zero misses among 104 NOK parts does not
guarantee zero future misses. A 10.9% false-reject rate may still exceed a site's
acceptance limit. No YOLOZU reproduction or production qualification is claimed;
access and permissions for the original data and weights are not established
by this review.

## Proposed YOLOZU scope

Add an opt-in, offline comparison over **existing predictions**, with physical
parts as the evaluation unit. Keep the Stable predictions interface contract
and image-level evaluators unchanged. The proposed names below are design
fields, not shipped schemas, CLI options or public APIs.

Inputs should bind each image to a `part_id`, `view_id` and `region_id`, with
the expected view set and independently obtained part-level OK/NOK labels.
Use a separate mapping artifact so grouping does not rewrite prediction files.
All images of a physical part must remain in the same development/evaluation
split; label and mapping validation must catch cross-split leakage.

Freeze the prediction digest, model/inference identity, class map, part/view
mapping, labels and evaluation cohort for both comparison arms. Freeze rules
before evaluating the held-out parts. Record their version/hash and the source
of each acceptance limit, including calibration identity, units and image
geometry where dimensions are used. The evaluation protocol should bind both
the baseline and rule configurations while identifying their intended
difference. A lower false-reject rate must not trigger post-hoc threshold tuning.

Start with a declared any-defect baseline: any above-threshold defect prediction
in an expected view rejects that part. Separately evaluate explicit
region/category, dimension and per-frame count rules, then aggregate retained
evidence across the part's views. Classification and localization outputs need
declared adapters; do not invent boxes for image-level classifications.

Rule ordering and overlap resolution must be explicit, deterministic and traced.
Do not copy the paper's pixel limits into a universal preset. Do not sum repeated
views as independent physical-defect counts. A cross-view physical-defect count
would need a separately evaluated correspondence method and is outside this
proposal. Independent rule-group ablations must each start from the same raw
predictions; label cumulative experiments separately.

Missing/duplicate views, inconsistent mappings, unsupported rules and missing
calibration must never silently produce OK. Reject malformed inputs before
scoring; represent incomplete parts as abstained with a reason. A complete view
with an explicitly empty prediction is different from a missing view.

No model training, weight download, live camera/PLC control, CNN activation,
automatic production acceptance or telemetry is included.

## Comparison report

Report image-level metrics separately from part-level decisions. Each arm must
include TP, FP, FN and TN with NOK as the positive class, plus abstained OK/NOK
counts and total support. For decided parts, report NOK recall `TP/(TP+FN)` and
false-reject rate `FP/(FP+TN)`. Mark a zero denominator as undefined, not zero
error or perfect recall.

Also report decision coverage and cohort-wide yields: `TP / all labeled NOK`
and `TN / all labeled OK`. Abstentions stay in these denominators, so withholding
a difficult part cannot silently improve the measured result. Show the paired
comparison on parts decided by both arms and retain the full-cohort accounting.

The paired transition table should separate:

- repaired false rejects: a truly OK part changes from NOK to OK;
- introduced false accepts: a truly NOK part changes from NOK to OK;
- repaired false accepts and introduced false rejects;
- unchanged decisions and transitions to/from abstention.

Include confidence intervals with the method and sampling assumptions stated.
Use parts, not correlated images, as the unit; account for lot/part dependence
when relevant. Record per-part reason codes and contributing views/rules so a
reviewer can trace why a decision changed. Reports should reference local
artifact digests without uploading images or customer identifiers.

The site owner must set acceptable miss, false-reject and abstention limits
before testing. Lower false rejects alone cannot establish a passing result.
This report is evaluation evidence, not permission to activate a model or
release products.

## Implementation acceptance checks

`YOLOZU-yz67` should cover these checks before being closed:

- Small deterministic CPU fixtures exercise every decision transition, including
  a filter that reduces false rejects but introduces a missed NOK part.
- Boundary, unit/calibration, overlapping-rule and per-frame count tests prevent
  silent reinterpretation. Input ordering must not change decisions or metrics.
- Missing/duplicate view, explicit empty prediction, inconsistent label/mapping,
  cross-split part, no-OK/no-NOK and all-abstained cases have explicit outcomes.
- Original prediction bytes remain unchanged; reports retain input, rule and
  comparison-protocol hashes and reconcile counts with the original cohort.
- Synthetic tests establish wiring and accounting only. Any industrial benefit
  claim needs a separate frozen evaluation with authorized data.

If implementation adds a CLI, it must support `-h/--help` and update both tool
manifests and related docs in the same PR. Until then, do not document an
executable command for this proposal. Use the existing
[evaluation protocol template](../evaluation_protocol_template.md) to record
the fixed inputs and evidence boundaries.
