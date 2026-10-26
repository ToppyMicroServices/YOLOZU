# Part-level inspection comparison (Experimental)

Compare the same saved predictions with an any-defect baseline and explicit
acceptance rules. This offline tool does not run a model, change predictions,
read images, or authorize product release. The Stable predictions interface
contract and existing image-level evaluators are unchanged.

## Run the synthetic example

From a source checkout:

```bash
python3 tools/eval_part_inspection.py --help
python3 tools/eval_part_inspection.py \
  --predictions docs/examples/part_inspection/predictions.json \
  --mapping docs/examples/part_inspection/mapping.json \
  --rules docs/examples/part_inspection/rules.json \
  --protocol docs/examples/part_inspection/protocol.json \
  --output reports/part_inspection_example.json
```

Use a new output path on each run: existing reports and input files are never
overwritten. Each input must be a regular JSON file of at most 64 MiB. Invalid
input exits with code 2 without publishing a report. Duplicate JSON keys,
nonfinite numbers and unknown configuration fields are errors. No network,
GPU, subprocess worker, telemetry or additional dependency is used.

The example deliberately repairs one false reject **and introduces one missed
NOK part**. Baseline counts are TP=2, FP=1, FN=0, TN=0; rule counts are TP=1,
FP=0, FN=1, TN=1. This is synthetic accounting evidence, not industrial efficacy.
Its all-zero model digest and rule limits are synthetic, not reusable defaults.

The package API is `yolozu.eval.part_inspection.compare_inspection_files`;
`compare_inspection` accepts the four decoded JSON values without mutating them.
No top-level `yolozu` subcommand or MCP tool is added.

## Input interface contract

The four example files are complete inputs. Mapping, rules and protocol use
`schema_version: 1` and the following fields.

**Predictions.** Use a list of entries or a wrapper with `predictions: [...]`.
Detection entries use existing `image`, `detections`, `class_id`, `score` and
normalized `bbox: {cx, cy, w, h}` fields. Scores and box coordinates must be
finite and in range; dimensions must be positive. There is no clamping or
basename matching. Image keys must match the mapping exactly and occur once.
An explicit empty `detections: []` is a valid complete view; a missing entry is
not. Schema versions 1 and 2, or versionless existing detection entries, are
accepted. This lane does not accept image-to-detection dictionary shorthand.

For a region declared `adapter: "classification"`, use a separate top-1 entry:
`{"image": "shaft-a.png", "classification": {"class_id": 0, "score": 0.9}}`.
This Experimental adapter is not an extension to the Stable detection schema.
Do not add fake boxes. The class map must include clean as well as defect
classes; `defect_class_ids` specifies which classes are defect candidates.

**Mapping.** `label_source` records where independently obtained labels came
from. `parts` contains `part_id`, `label` (`OK` or `NOK`), `split` and a nonempty
`expected_views` list of `{view_id, region_id}`. Each part must cover every
configured region. Part IDs must be unique across the supplied splits: putting
one physical part in train and test is an error. This cannot detect an omitted
split, aliases for the same physical part, or untruthful source labels.

`views` contains `{image, part_id, view_id, region_id}` for available frames.
Detection views also require positive integer `width_px` and `height_px`
(at most 2^31-1), referring to the image geometry in which boxes are normalized.
An optional `calibration_id` references the rules file. Classification views
do not carry image geometry. Duplicate part/view pairs, images, inconsistent
regions, undeclared views and predictions without mappings are errors.

**Protocol.** Pin `protocol_id`, evaluation `split`, `score_threshold` (inclusive),
`class_map` (canonical integer-string IDs to names), and region-keyed `models`.
Each model records `model_id`, lowercase `model_sha256` and `inference_settings`.
These are operator assertions: the evaluator neither loads nor verifies weights.
Set `evidence_kind` to `synthetic` or `operator_supplied`. Set `sampling` to
`independent_parts` only when that assumption is justified; otherwise use
`dependent_parts`. Include all relevant split records when checking leakage.

**Rules.** Pin `ruleset_id`, string `version`, `precedence: "first_match"`,
`regions`, `rules` and `calibrations`. Each region defines `adapter`, nonempty
`defect_class_ids` and positive `bad_view_threshold`. A threshold larger than
that region's expected view count in any supplied part is an error.

Each rule has a unique `rule_id`, `region_id`, defect `class_id`, nonnegative
integer `priority`, positive integer `min_count` and a nonempty `source` for the
acceptance limit. Optional `min_width` and `min_height` require `unit: "px"` or
`"mm"` and the detection adapter. Unspecified dimensions have threshold zero.
For mm, every mapped view in the affected region needs a calibration entry:
`{"unit": "mm_per_pixel", "x": 0.01, "y": 0.02, "source": "calibration record"}`.
Both scales must be positive and finite. Width and height are bounding-box
dimensions, **not** mask area, physical length along a rotated defect, or a
paper-specific metrology measurement. Calibration and geometry must be suitable
for the site; the evaluator cannot establish this from JSON.

## Decision semantics

The baseline rejects a complete part if any declared defect class has a score
at or above the pinned threshold in any expected view. The rule arm uses those
same candidates. For each region/class, the lowest numeric priority wins;
equal-priority overlaps are errors. Dimensions are not match conditions:
a failed size limit does **not** fall through to another rule. Shadowed rule
IDs appear in the trace. Unmatched defect classes remain rejection evidence.

A candidate qualifies when both size limits are met, inclusively. Limits are
converted to normalized coordinates without an epsilon. `min_count` counts
qualifying detections **within that frame and class**. A classification view
has one result, so its `min_count` must be 1. A view is bad if any class retains
evidence. A region is bad if at least `bad_view_threshold` of its views are bad;
any bad region rejects the part. No cross-view physical-defect counting occurs.

Missing expected views or missing prediction entries abstain the entire part
in both arms, even if observed views already contain defects. Missing calibration
or malformed configuration instead fails the run. Empty complete views remain
distinct from missing views. A supplied frame is not proof of camera coverage.

This filtering-only rule arm cannot turn a baseline OK into NOK. Those transition
counts are still reported as zero. Independent ablations require separate runs
from the same raw predictions, with different frozen rule files.

## Read the report

`arms` reports TP/FP/FN/TN with NOK positive, abstained OK/NOK counts and support.
Every rate includes its numerator and denominator. Decided-only NOK recall is
TP/(TP+FN); false-reject rate is FP/(FP+TN). Coverage and full-cohort
`nok_detection_yield` and `correct_accept_yield` retain abstained parts in their
denominators. Zero denominators produce JSON `null`, never perfect performance.
`paired` repeats metrics on the common decided cohort and reports all 18 labeled
transitions, repaired errors and newly introduced errors. `parts` records local
part/view IDs, missing views, eligible counts, rule IDs and per-region evidence.

Independent-part rates include two-sided 95% Wilson intervals. For dependent
parts, intervals are `null`: use an appropriate lot/cluster analysis separately.
No paired-effect interval or production pass/fail test is performed. Image-level
metrics are explicitly marked `not_computed`; run existing evaluators separately.

`provenance` records raw-byte SHA-256 hashes from the four files read once,
canonical JSON hashes, model assertions, class-map and cohort-label hashes, and
a comparison hash binding the input hashes and engine version. Ordering changes
may change input hashes, but not decisions or metrics. File digests identify the
snapshot read, not later edits to those files. Keep part IDs and traces private;
the tool does not upload them or include input filesystem paths in the report.

Freeze inputs, rule limits and site acceptance criteria before held-out testing.
The report always says `comparison_only` and `production_qualified: false`.
Lower false rejection alone is not success. See the
[research motivation and scope](roadmaps/part_level_inspection.md) and the
[evaluation protocol template](evaluation_protocol_template.md).
