# Adaptive Image Routing v1 Policy

## Status and authority

This document is the normative v1 policy for future adaptive local image routing.
Later interface contracts, qualification evidence, selectors, MCP surfaces, local
execution, and tests must follow it.

This policy does not add a router, model adapter, recommendation service, or
execution path to the current release. YOLOZU's current Stable predictions
validation and evaluation lane remains unchanged. Adaptive routing is future
delivery work whose target product classification is Experimental.

The source tree now includes strict Python validators and packaged schemas for
the typed request, workload, environment, bundle, lifecycle, local artifact
inventory, qualification report, evidence activation, screening/support eligibility
observations, and selection-decision interface contracts.
`yolozu doctor` additively emits the validated EnvironmentProfile from bounded,
privacy-safe live probes. An unsupported or failed probe remains unknown, and
the profile is configuration input rather than qualification evidence.
The public bundle registry, lifecycle/support/activation streams, and qualification
report directory contain no selectable evidence. The package validates and loads the
exact empty bundle-registry/lifecycle SSOT without importing a model runtime.
An explicitly supplied workspace catalog is always operator-asserted and fails the
public selection trust gate even when its checksums are internally consistent.

The Experimental `yolozu qualify-image-pipeline` surface is now implemented for
exact managed bundles. It pins bounded input and artifact descriptors, runs only a
repository-owned network-free runner in a terminable child process group, applies
the frozen v1 schedule and handoff, and publishes one unactivated report through
`ManagedOutputTransaction`. The packaged registry and code-owned runner factory map
are currently empty, so the default command fails with an actionable error and does
not create synthetic or no-op qualification evidence. It is not a selector,
recommendation service, model adapter, or general image-processing capability.

`selected` means that one registered pipeline survived every hard filter, matched
one active trusted qualification record for the exact measured configuration, and
ranked first under the requested policy. `abstained` is a valid routing outcome when
that proof is absent. It is not an internal error and it must never be replaced by
an unqualified default.

The word "recommended" has only that bounded meaning in this document. YOLOZU does
not claim one universal model or the latest model is suitable for every task,
machine, dataset, or service objective.

## Experimental qualification command

The command accepts a canonical `ImageJobSpec` JSON file, a workspace-confined
single image or bounded directory, an exact packaged bundle ID/version and lifecycle
channel, an optional workspace-confined artifact root, and a fresh managed output
directory. A ground-truth path can be paired only with an exact registered code-owned
evaluator and a preregistered quality requirement; no evaluator is registered in the
current empty baseline. `--qualification-timeout-seconds` is restricted to `60..14400`.
Soft-real-time requests require at least 1,260 seconds so the ten-minute section and
bounded setup can fit. `--smoke` is a short wiring check and can emit only `smoke`,
never `qualified`.

```bash
yolozu qualify-image-pipeline --help
```

A non-smoke batch qualification uses a fresh runner, one cold-start input, 20
warm-up iterations, and exactly three reset-at-zero repeats of 200 successful
validated handoffs. Soft-real-time adds an exact all-sample section lasting at
least 600 seconds, with a hard one-million-sample cap. The report retains aggregate
latency, throughput, coverage, and complete-memory status, not raw per-image timing
events. Missing full runner-tree or all-device memory coverage stays `unknown` and
cannot pass a matching hard memory gate.

The managed public tree contains only `qualification_report.json` and
`checksums.json` in the current implementation. It omits filenames, prompt text,
input hashes, private labels, and raw predictions. Report presence never activates
the evidence. Activation and support claims require separate reviewed records.

## Rationale

The repository already validates and evaluates predictions and can inspect parts of
the local environment. Those capabilities do not by themselves establish which
model pipeline should run on a particular machine. A fixed policy is needed before
schemas and adapters are added, otherwise each implementation could invent a
different meaning of qualification, privacy, ranking, or fallback.

## Alternatives considered

- A universal model choice was rejected because task, runtime, license, hardware,
  workload, and measured quality differ.
- A weighted `balanced` score was rejected for v1 because hidden or changing weights
  can obscure hard trade-offs.
- Direct backend choice by an AI client remains an explicit manual operation. It is
  not an evidence-based recommendation.
- Deferring policy until model adapters exist was rejected because adapters must not
  define conflicting eligibility rules.

## V1 request surface

The calling AI client may translate natural-language intent into the typed request.
YOLOZU core does not embed an LLM and does not accept a free-form operation paragraph
as a prompt payload.

| Field | V1 decision |
|---|---|
| Task | Exactly `object_detection` or `instance_segmentation` |
| Input mode | Exactly `single_image` or `bounded_directory` |
| Execution mode | Exactly `batch` or `soft_realtime`; never inferred from an FPS field |
| Batch and concurrency | `batch_size=1` and `concurrency=1` |
| Image count | Required `max_images` in `1..100`; a single-image request contains exactly one image |
| Result count | Required `max_results_per_image` in `1..1000` |
| Job deadline | Required `job_timeout_seconds` in `1..3600` |
| Prompt mode | Exactly `fixed_classes` or `text` |
| Ranking policy | Exactly `accuracy_first`, `latency_first`, `throughput_first`, or `memory_first` |
| Compute policy | Exactly `auto`, `cpu_only`, or `accelerator_required` |
| Network policy | Exactly `deny` |
| `allowed_maturities` | A lifecycle-channel allowlist: Stable, or Experimental only when explicitly allowed; never Candidate or Research |

Optional provider restrictions contain `0..8` exact provider IDs. An absent or empty
list means no additional provider restriction. Optional precision restrictions are
an exact subset of `fp32`, `tf32`, `fp16`, `bf16`, and `int8`; absent or empty means
no additional precision restriction. An optional SPDX allowlist works the same way,
but it never converts an unknown or unreviewed license into an approved one.

`cpu_only` rejects accelerator providers even when the host has an accelerator.
`accelerator_required` rejects CPU-only bundles. Explicit provider and precision
restrictions are hard gates and never silently fall back to `auto`.
`memory_first` with `compute_policy=auto` is invalid because applicable accelerator
memory would be ambiguous. It requires `cpu_only` or `accelerator_required`.

The request may set finite positive hard limits for cold start, p95 latency,
runner-tree RSS, and accelerator process-tree memory. A batch request may set
`min_repeat_throughput_fps` and must not set `min_sustained_fps`. A soft-real-time
request may set `min_sustained_fps` and must not set
`min_repeat_throughput_fps`. A wrong-mode metric is invalid.

A quality requirement is complete only when it contains all of these fields:

- metric ID;
- direction, exactly `higher_is_better` or `lower_is_better`;
- finite threshold;
- approved evaluation-dataset ID and digest;
- evaluation-protocol SHA-256; and
- approved evaluation-vocabulary ID.

For `higher_is_better`, equality with the threshold passes and a lower value fails.
For `lower_is_better`, equality passes and a higher value fails. `accuracy_first`
requires this complete object. Other policies may omit it. When it is omitted,
quality is omitted from every candidate's rank key; a merely known quality value is
not preferred over an unknown or differently measured value.

Input and output paths are tool or service arguments, not `ImageJobSpec` fields. A
typed request cannot carry an arbitrary filesystem path, command, module name,
environment assignment, or free-form backend argument.

## Typed prompt payload

The payload is required and mutually exclusive by prompt mode:

- `fixed_classes` carries `fixed_classes`;
- `text` carries `text_prompts`.

Each payload is a list of `1..128` short object phrases. Each phrase is normalized
with Unicode NFKC, surrounding whitespace is removed, and then the validator rejects
an empty phrase, a control character, or a duplicate. Each normalized phrase has
`1..256` Unicode code points. The complete normalized payload is at most 4096 UTF-8
bytes.

For `fixed_classes`, every normalized phrase must be an exact subset member of the
bundle's immutable normalized class vocabulary. Core does not guess case variants,
synonyms, translations, or fuzzy matches. Native detections for unrequested classes
are filtered and retained classes use the pinned request-to-bundle index mapping.

For `text`, missing qualified text support causes abstention. A text-capable runner
returns an integral request-prompt index. Core maps it only to the exact normalized
requested phrase and rejects generated label strings or an out-of-range index.

NFKC is field-specific prompt processing. The canonical JSON serializer defined
below does not apply global Unicode normalization.

## Input and output safety budgets

All bounds are inclusive unless stated otherwise. A deployment may declare smaller
limits; it may not exceed the v1 maxima.

| Boundary | V1 maximum or rule |
|---|---|
| Decoder formats | JPEG, PNG, and single-frame WebP, confirmed by magic and decoder result |
| Source file bytes | 64 MiB (67,108,864 bytes) per image |
| Source bytes per job | 512 MiB (536,870,912 bytes) |
| Decoded dimensions | Width and height each in `1..16384` |
| Decoded pixels | 64,000,000 per image and 512,000,000 per job |
| Directory scan | Direct children only, no recursion, at most 1,024 entries before decoding |
| Masks | At most 1,000 mask artifacts |
| Output files | At most 1,003 files: `predictions.json`, `provenance.json`, `checksums.json`, and masks |
| Output bytes | 4 GiB (4,294,967,296 bytes) for the complete managed tree |

The input validator rejects archives, animation, malformed data, decompression-bomb
conditions, symlinks, nested directories, non-regular entries, workspace escapes,
and normalized-name collisions before model load. A bounded directory enumerates
direct children exactly once without recursion. Every child counts toward the
1,024-entry scan cap. Symlinks, directories, sockets, devices, FIFOs, and escapes
fail; other direct regular files are ignored.
Candidate images are direct regular files whose NFKC-normalized basename ends in
`.jpg`, `.jpeg`, `.png`, or `.webp`, case-insensitively. They are sorted by normalized
basename UTF-8 bytes. Magic and the bounded decoder, not the suffix, make the final
format decision. The job fails rather than truncates when the entry or image count
exceeds its cap. A dimension may pass while the pixel product fails; for example,
8000 by 8000 reaches the per-image pixel maximum, while 16384 by 16384 exceeds it.

`checksums.json` lists every other declared regular output with its exact relative
path, byte size, and SHA-256, plus the expected path set, file count, and total bytes.
Entries are in ascending order of the already validated exact relative-path UTF-8
bytes, with no additional Unicode normalization. The expected set, count, and byte
total cover exactly those listed files and exclude the manifest.
The 1,003-file and 4-GiB managed-tree limits include the manifest itself. The manifest
never lists itself; when an enclosing output digest exists, that digest covers the
manifest bytes. A missing, extra, duplicate, reordered, self-referential, or
mismatched entry is invalid.

`ManagedOutputTransaction` is the shared implementation for these future
repository-owned trees. It pins the approved root and destination parent, uses
directory-relative no-follow operations, and publishes a fresh same-filesystem
stage only after every declared file and the control manifest validate. `force`
first validates the exact old manifest and tree. Cleanup unlinks only those
validated singly linked regular files, then `checksums.json`, then known empty
directories; it never recursively removes a caller path.

The implementation currently fails closed outside POSIX because the required
directory-descriptor operations are unavailable there. A same-directory rename
probe is performed for each transaction. Successful POSIX same-filesystem rename
supports atomic directory visibility, but power-loss durability remains
best-effort and is reported separately according to directory-fsync support.
A recovery marker can finish or roll back only a fully validated known state.
Changed, missing, or ambiguous state returns `manual_recovery_required` and keeps
the remaining data for operator review.

## Control-record and decision budgets

One registry contains at most 128 bundle specs and 4,096 lifecycle events. One
recommendation consumes at most 512 evidence reports, at most 4 MiB per control or
evidence JSON record, and at most 128 MiB in total.

Every file-backed registry, lifecycle, screening, build, evidence, or decision
control record is parsed before object construction with these limits:

- UTF-8 only and no duplicate object keys;
- nesting depth at most 64, with the root object or array at depth 1 and each nested
  object or array increasing depth by one;
- at most 100,000 aggregate object members plus array items;
- object keys at most 256 UTF-8 bytes; and
- individual strings at most 1,048,576 UTF-8 bytes; and
- integer tokens at most 128 ASCII bytes, with canonical zero written only as `0`.

Binary-float, exponent, non-finite, leading-zero, and negative-zero number spellings
are invalid. Fractional governed values use CanonicalDecimalV1 strings.

Large user-owned prediction, stream, tracking, or OCR outputs use separate bounded
incremental writers and validators. They are not loaded as an unrestricted JSON DOM.

A `CandidateEvaluation` contains bounded IDs and digests, rank state, at most 32
reason codes for that candidate, at most one 256-UTF-8-byte detail for each reason, a
human summary of at most 1,024 UTF-8 bytes, and at most 32 bounded ranking-trace
steps. A complete canonical `SelectionDecision` is at most 1 MiB and includes every
validated registry entry. It is never truncated. If the full input or full decision
cannot fit, processing fails before producing a decision with
`registry/evidence_limit_exceeded`; this is a validation/service failure, not an
abstention reason. Requiring a 33rd reason or trace step for any candidate fails with
the same code before decision creation; the implementation never keeps only the first
32 or discards an exclusion.

Candidate summaries and details use code-owned fixed templates. They do not copy
registry descriptions, source text, URLs, local paths, untrusted exception bodies,
or runner output into the decision.

An eligible candidate that loses only by rank uses an `eligible_not_selected` rank
state. It is not assigned a false hard-failure reason.

Decision ordering is independent of registry and evidence input order. After the
per-channel collapse, `CandidateEvaluation` records are sorted by the canonical
identity tuple defined below. Within one evaluation, reason-code/detail pairs are
sorted by reason-code ASCII bytes. Trace steps follow the fixed filter-step number;
ties use Stable before Experimental and then reason-code ASCII bytes. Top-level
outcome codes are also ASCII-byte sorted. A detail remains attached to its reason and
is never independently reordered.

## Eligibility and filter order

The selector applies these steps in order. Unknown values never satisfy a hard gate.
P0 registry and lifecycle trust is `yolozu_managed` only when derived from the
canonical managed source. A custom file or a JSON field that claims its own trust is
`operator_asserted` and nonselectable.

The bounded screening observation status is exactly `not_applicable`,
`current_pass`, `current_hold`, `current_reject`, `absent`, `untrusted`, `conflict`,
or `revision_mismatch`. `existing_code_owned` requires `not_applicable`.
`screened_candidate` requires one `yolozu_managed` `current_pass` bound to the exact
screening stream and source revision. `untrusted` uses `screening_untrusted`; every
other nonpassing status uses `screening_not_current_pass` and fails closed.

The per-channel support observation status is exactly `matching_one`, `no_match`,
`absent`, `untrusted`, `conflict`, or `not_required_site`. Public selection requires
one canonical managed `matching_one`. `no_match` or `absent` uses
`support_profile_mismatch`; `untrusted` uses `support_profile_untrusted`; `conflict`
uses `support_profile_conflict`. `not_required_site` is valid only for reviewed
`site_managed` evidence with `support_scope=site_qualified`.

1. Validate managed registry and lifecycle trust, current channel pointers, current
   screening state for screened candidates, and the support-profile observation for
   each allowed channel. The pointed spec must be non-test, enabled, non-revoked, and
   in an allowed lifecycle channel. Every runner-consumed artifact must have a
   complete current license-review projection; malformed or missing lifecycle state
   fails closed.
   Public selection requires exactly one matching current `SupportProfileSpec` for
   the environment, workload, protocol, and advertised gates. No exact match uses
   `support_profile_mismatch`; multiple current matches or a malformed projection use
   `support_profile_conflict`; untrusted provenance uses
   `support_profile_untrusted`. Site-managed evidence may bypass the public support
   set only with `support_scope=site_qualified`.
2. Check task and prompt compatibility. Fixed classes require the exact normalized
   subset and deterministic index mapping.
3. Require approved license review for every runner-consumed artifact and enforce
   any request SPDX allowlist.
4. Reject a bundle that requires execution network access.
5. Check hardware, architecture, compute policy, provider, precision, runtime,
   memory, loader, execution trust class, and isolation compatibility. Unsafe or
   bundle-supplied code never falls back to host execution.
6. Require one passing local regular-file observation for every ordered artifact,
   including exact byte size and SHA-256. Registry metadata and evidence do not prove
   local presence.
7. Require exactly one active trusted evidence activation for the immutable
   selection key.
8. Require the referenced `QualificationReport` status to be `qualified` and its
   time interval to be valid.
9. Require exact bundle spec, ordered artifact-set, measured environment,
   qualification-workload, and protocol fingerprints. When quality is requested,
   also require the exact dataset and vocabulary identities.
10. Apply every hard quality, cold-start, repeat-throughput or sustained-FPS, p95,
    RSS, and accelerator-memory gate using only the metric for the requested
    execution mode.
11. Apply the policy-specific lexicographic rank and final canonical identity tie
    break.

The lifecycle channel, product capability classification, trust domain, and support
scope are separate axes:

- the current bundle channel is `Candidate`, `Experimental`, or `Stable`;
- product documentation classifies a capability as Stable, Experimental, Research,
  or future delivery;
- trust is `yolozu_managed`, `site_managed`, `operator_asserted`, or `unknown`;
- support scope is `public_qualified`, `site_qualified`, or `none`.

`allowed_maturities` is the ImageJobSpec field name, but v1 interprets it only as an
allowlist of current lifecycle channels. Maturity is not an immutable bundle-spec
field. A Stable-channel bundle may be selected, or an Experimental-channel bundle
may be selected only when the request allows Experimental. Candidate-channel and
Research-only work are never selectable. `CandidateEvaluation` is only the name of
an evaluation record; it does not make a Candidate-channel bundle selectable.

The adaptive-routing capability itself remains future delivery whose target product
classification is Experimental. Selecting a Stable-channel bundle does not promote
adaptive routing itself to Stable.

The stable code `maturity_disallowed` means the pointed lifecycle channel is absent
from `allowed_maturities`; it does not imply an immutable bundle maturity field.

After per-channel screening and support-profile checks, identical `spec_digest`
pointers across surviving allowed channels collapse to one evaluation. Stable is the
effective channel when both Stable and Experimental match; otherwise the matching
Experimental channel survives. The trace records every pointed and matching channel.
A nonmatching Stable pointer does not hide a matching Experimental pointer. Different
spec digests remain separate candidates.

P0's `network_policy=deny` is an application-level policy for audited code-owned
runners: they declare no network requirement and must not initiate network access.
It is not a claim of OS-enforced isolation. A third-party runner that needs enforced
isolation remains ineligible until the separate isolation work provides and proves a
matching backend. Missing isolation causes abstention, not ordinary subprocess or
in-process fallback.

## Deterministic ranking

Quality values are comparable only after exact metric, direction, dataset, protocol,
and evaluation-vocabulary matching. Direction-aware order is descending for
`higher_is_better` and ascending for `lower_is_better`.

The final identity key for every policy is the ascending tuple
`(family_id, bundle_id, exact bundle_version UTF-8 bytes, spec_digest)`.
`bundle_version` is not parsed or normalized as SemVer.

| Policy | Lexicographic keys after hard filters |
|---|---|
| `accuracy_first` | Direction-aware quality, p95 latency ascending, runner-tree peak RSS ascending, identity |
| `latency_first` | p95 latency ascending, request-defined quality when present, runner-tree peak RSS ascending, identity |
| `throughput_first` in `batch` | Repeat throughput descending, p95 ascending, request-defined quality when present, identity |
| `throughput_first` in `soft_realtime` | Sustained FPS descending, sustained p95 ascending, request-defined quality when present, identity |
| `memory_first` with `accelerator_required` | Accelerator process-tree peak bytes ascending, runner-tree peak RSS ascending, p95 ascending, request-defined quality when present, identity |
| `memory_first` with `cpu_only` | Runner-tree peak RSS ascending, p95 ascending, request-defined quality when present, identity |

With `compute_policy=cpu_only`, accelerator memory is `not_applicable`, not unknown
and not zero. It is omitted from the hard gate and rank key.

Every metric present in the requested policy's rank key must be known. Unknown p95
excludes a candidate for every policy. Unknown runner-tree RSS excludes it from
`accuracy_first`, `latency_first`, and `memory_first`; unknown accelerator memory
excludes it from accelerator-required `memory_first`; and unknown requested
throughput excludes it from `throughput_first`. Use `ranking_metric_unknown` when no
hard constraint requested that metric and `requested_metric_unknown` when one did.
A complete quality object remains a hard gate under every policy. When no quality
object is supplied for a non-accuracy policy, quality is removed from every rank key
rather than comparing unrelated model-card metrics. An unknown value is never
coerced to zero, infinity, or another numeric sentinel.

There is no `balanced` policy in v1. The selector does not add weights, normalize
against the current candidate set, or approximate hardware families.

## Qualification evidence and lifecycle

Measured evidence is bound to the immutable bundle spec, ordered artifact-set digest,
privacy-safe measured environment fingerprint, qualification-workload fingerprint,
and protocol fingerprint. A request mismatch makes a report inapplicable to that
request; it does not rewrite the historical report.

Current lifecycle state is checked separately. A current global revoke, disable, or
license block excludes execution. A channel assignment, promotion, or documentation
classification change does not by itself stale a measurement collected while the
bundle was Candidate. The decision
pins the current lifecycle projection digest only to detect change between
recommendation and execution.

The environment fingerprint identifies a measured configuration, not one unique
physical host. It excludes identifying host data. Evidence from representative
images describes only that measured workload. It does not prove the same latency or
quality for every possible image-content distribution.

### Activation and freshness

The activation key is the canonical hash of the immutable bundle spec, artifact set,
environment, qualification workload, and protocol. There may be exactly one active
report for that key:

- zero active reports is a valid projection only after a complete terminal revoke or
  withdrawal event and causes abstention with `evidence_revoked` or
  `evidence_inactive` as defined by that event;
- one active report is evaluated; and
- more than one active report causes `evidence_conflict`. The selector never chooses
  the newest or most favorable record.

A dangling supersession, incomplete transition, gap or fork, or history that still
claims an active report while projecting none is a validation failure, not a valid
zero-active abstention. A superseded or revoked report ID/digest cannot be reactivated;
a corrected measurement requires a new report ID/digest.

Freshness is anchored only to the report's `completed_at` instant. `valid_until` must
be later than `completed_at` and no later than
`completed_at + 7,776,000 seconds`. At decision time, a report is time-valid only
when `completed_at <= decision_time < valid_until`. Equality with `valid_until` is
expired. A future `completed_at`, supersession, terminal revocation, or immutable
bundle, artifact, runtime, environment, workload, or protocol mismatch makes it
unusable earlier.

The active `EvidenceActivationRecord` has its own `activated_at` and `valid_until`.
It is time-valid only when
`report.completed_at <= activated_at < activation.valid_until <= report.valid_until`
and `activated_at <= decision_time < activation.valid_until`. A future activation,
expired activation, reversed interval, or activation that outlives the report is
invalid even when the report itself remains fresh.

### Trust domains

- `yolozu_managed` requires a canonical bundle, public non-sensitive inputs and
  protocol, a repository-owned reproducible report, and explicit repository review
  and activation. It may support `public_qualified` scope.
- `site_managed` requires a report produced by the code-owned qualifier and explicit
  local review and activation. It supports only `site_qualified` scope and creates no
  public YOLOZU support claim.
- `operator_asserted` applies to custom workspace records until a governed review
  workflow derives stronger trust.
- `unknown` means the issuer or workflow could not be established.

`operator_asserted` and `unknown` evidence are not selectable. A checksum proves
post-creation integrity, not who made or reviewed the claim.

## Adaptive-routing v1 qualification protocol

This protocol fixes the following schedule; it is not a statement about every other
YOLOZU benchmark.

1. Measure `cold_start_ms` once with input index 0. Start the monotonic elapsed clock
   before creating a fresh runner process or sandbox, before probe, model load,
   artifact deserialization, and preprocessing. Stop after the first strict validated
   postprocessed result. Reused runner or model state is forbidden. OS
   filesystem-cache state is recorded as uncontrolled; the run does not claim a
   cleared filesystem cache. Keep this runner loaded through the remaining steps.
2. Run exactly 20 warm-up iterations in that runner. Warm-up iteration `i` uses
   `i mod input_count`.
3. For `soft_realtime`, run the sustained section defined below immediately after
   warm-up. For `batch`, omit it with the explicit status `not_required`.
4. Run exactly three timed repeats sequentially in the same loaded runner. Each is a
   separate aggregation window, not a fresh process start. Each repeat resets at input
   index 0 and makes exactly 200 timed attempts using `i mod input_count`. Every
   attempt must complete successfully; a failed or partial attempt invalidates the
   repeat and is never replaced to reach 200. Every accepted input is therefore
   covered at least twice per completed repeat. The repeats are diagnostic only for
   `soft_realtime`.
5. When the job contains a complete quality object, evaluate task-native quality
   exactly once per unique input after the repeats using predictions from this loaded
   bundle and run, outside every timed sample set. Use only the preregistered metric,
   direction, threshold, dataset, vocabulary, and protocol. Without that object,
   record quality as `not_required`; do not run or infer a model-native score. Then
   close the runner and memory collectors.

A cold-start, warm-up, sustained-section, repeat, or required-quality timeout or
failure fails qualification. It never becomes a numeric value, passing unknown, or
replacement iteration.

The schedule version, input-order rule, cold-start and latency interval definitions,
bounded handoff identity and limits, percentile rule, and memory collector ID,
version, source, and scope are bound into the protocol fingerprint. Evidence from a
different definition is not comparable.

The timed latency interval ID is `image_e2e_validated_handoff_v1`. It starts
immediately before decoding pinned source bytes and ends only after preprocessing,
prediction, postprocessing, requested-class or prompt mapping, strict result and mask
validation, mask encoding, and completion of the same bounded code-owned result and
mask handoff used by execution. Final managed-directory checksum publication and
rename are outside the interval. A partial or inference-only interval is not
comparable.

For sorted sample count `N`, p50, p95, and p99 use nearest-rank index
`ceil(p * N) - 1`. A completed repeat records only aggregate
`repeat_processed_count`, `repeat_duration_ns`, `p50_latency_ms`, `p95_latency_ms`,
`p99_latency_ms`, exact repeat-throughput count/duration inputs, and complete
process-tree memory status in `repeat_runner_tree_peak_rss_bytes` and
`repeat_accelerator_process_tree_peak_bytes`. It does not persist raw timing events.
Batch qualification and ranking use the minimum repeat throughput and the maximum
`p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, and two repeat memory fields
across the three completed repeats. A failed or shorter run is only smoke or failed
evidence; it is not qualified.

Memory collection remains active from fresh runner creation through close so no
descendant is omitted. The top-level `runner_tree_peak_rss_bytes` is the maximum
aggregate resident memory of the complete runner process group or isolation cgroup
over that full lifetime. Top-level `accelerator_process_tree_peak_bytes` covers memory
attributed to every runner-tree process on every declared device over the same full
lifetime. These lifetime fields remain mandatory diagnostic evidence, but they do
not replace the repeat-window fields for batch selection or sustained-section fields
for soft-real-time selection. Collector ID, version, source, and scope are part of
the protocol. Wrapper-only, leader-only, one-device, or otherwise incomplete
collection is `unknown` for the affected lifetime and window fields.

For `soft_realtime`, the three 200-iteration repeats remain diagnostic. A separate
sustained section starts after cold start and warm-up, resets the input schedule to
index 0, and continues until both 600,000,000,000 monotonic nanoseconds and the current
strict handoff complete. Only successful complete handoffs enter `processed_count`.
Authoritative FPS is the exact ratio
`processed_count * 1,000,000,000 / duration_ns`, without binary floating point.

The sustained section retains every latency interval in a preallocated array capped
at 1,000,000 unsigned 64-bit samples and 8,000,000 bytes. Reaching the cap before the
duration and current-handoff conditions fails with `sustained_sample_limit`; no
sampling, approximation, overwrite, or early success is allowed. Sustained p95/p99
use all section samples and are stored as `sustained_p95_latency_ms` and
`sustained_p99_latency_ms`. `sustained_runner_tree_peak_rss_bytes` and
`sustained_accelerator_process_tree_peak_bytes` cover the complete runner tree during
the whole section, including the preallocated sample array and bounded handoff.
Cold start, warm-up, failed or partial results, and the three timed repeats are
excluded. Soft-real-time hard gates and ranking use only sustained throughput,
latency, and memory. A timeout or failure invalidates the section.

This static-image v1 runs one request at a time with no streaming queue. Queue depth,
drop count, and drop fraction are `not_applicable`, and an image job cannot request a
drop gate. They must not be recorded as zero or reused as stream evidence. Known
power or thermal mode and observed throttling status are recorded as bounded
environment facts when available; unknown remains unknown.

Soft real-time is an SLO observed for one exact environment, workload, and protocol
fingerprint. Hard real-time is unsupported.

## Identity, privacy, and retention

Three request/workload identities remain separate:

1. A sensitive local-only `local_job_digest` binds the canonical typed request and
   actual normalized prompt between recommendation and execution.
2. A sensitive local-only `local_input_digest` binds the ordered list of input index,
   source byte length, and source-byte SHA-256 without returning a filename or
   per-file hash.
3. A shareable `QualificationWorkloadProfile` fingerprint contains its schema and
   collector versions; task, input and execution modes; actual input count and the
   deterministic order policy; per-index decoded dimensions, color and applied
   orientation policy; decoder identity and version; batch size, concurrency, and
   `max_results_per_image`; compute policy and provider and precision restrictions;
   fixed latency-interval and code-owned result/mask-handoff IDs, versions, and
   scratch/output caps; `max_sustained_samples=1,000,000` for `soft_realtime`; prompt
   mode, count, maximum-codepoint bucket
   `1-16|17-32|33-64|65-128|129-256`, and total-UTF-8-byte bucket
   `1-256|257-1024|1025-2048|2049-4096`; and, when quality is required, the approved
   dataset, protocol, and vocabulary identities. It excludes paths, filenames,
   source bytes and hashes, pixels, labels, prompt text, every sensitive local digest,
   ranking policy, SLO thresholds, chosen model-input shape, and preprocessing or
   postprocessing identity. Those chosen bundle-specific execution facts remain bound
   by the immutable bundle spec and QualificationReport.

The sensitive local-only `artifact_resolver_state_digest` binds resolver version,
store kind, local root identity, and the ordered logical cache keys used between
recommendation and execution. Separately, `artifact_state_fingerprint` binds the
bundle-spec digest, artifact-set digest, and ordered artifact observations: artifact
ID, role, order, status, observed byte size, and observed SHA-256. It excludes
verification time, diagnostic text, and paths. Every model/pipeline artifact remains
bound; model and runtime asset hashes are not user-content hashes.

The three sensitive local digests may appear only in the MCP-returned decision and
user-owned local provenance. They never appear in qualification evidence, telemetry,
logs, repository pilot records, or public summaries. Per-file input hashes are
ephemeral during digest calculation and are never persisted or returned.

Exported evidence, telemetry, logs, and public summaries must not retain:

- raw images, pixels, prompts, labels, predictions, private dataset artifacts,
  private messages, or raw event telemetry;
- filenames, absolute paths, usernames, names, email addresses, hostnames, IP
  addresses, device serials, UUIDs, or persistent device identifiers;
- image, prediction, prompt, label, or private-dataset content hashes; or
- raw per-image timing events or untrusted runner output.

Workspace-relative input and mask identifiers may exist only inside user-declared
local output required by the existing predictions interface contract. They do not
enter selection evidence or public records.

Exported `actor`, `owner`, and reviewer values are exactly one non-personal role ID:
`site_operator`, `repo_maintainer`, `release_reviewer`, or `automation`. They are
never derived from git identity, support-account identity, or ticket prose.
Identifying consent and review references remain site-local. Exported consent
contains only status `granted|declined|withdrawn|not_required`, one to eight unique
ASCII scope tokens matching `[a-z][a-z0-9_]{0,63}` in ascending byte order, and a
valid `YYYY-MM-DD` date. A public review reference is 1..128 ASCII bytes matching
`[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` and identifies a repository-owned review record;
it is never a username, email address, URL, private ticket, or message.

## Canonical representation

All adaptive-routing digests reuse one code-owned `canonical_json_v1`
implementation. It:

- emits UTF-8 with no BOM or insignificant whitespace;
- preserves array order and sorts object keys by UTF-8 byte sequence;
- deterministically escapes quote, backslash, and U+0000 through U+001F, using the
  short JSON escapes for backspace, tab, LF, form feed, and CR and lowercase
  `\u00xx` otherwise;
- leaves other valid Unicode as UTF-8 without global normalization;
- rejects duplicate keys, invalid Unicode or surrogates, NaN, infinity, binary
  floats, and booleans used as integers;
- writes integers in minimal base 10 with zero exactly `0`;
- writes JSON `null`, `true`, and `false` in lowercase;
- omits a record's own digest field while retaining referenced digests; and
- hashes the exact bytes with SHA-256 and encodes lowercase hexadecimal.

Pretty JSON is never hashed. Every implementation reuses the same known vectors for
ASCII, Unicode, key order, decimals, and own-digest omission.

Every fractional threshold, metric, latency, FPS display, coordinate, and ranking
value uses `CanonicalDecimalV1`. Its full-match grammar is
`\A-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{0,8}[1-9])?\z`, or the target
language's equivalent full match. It allows at most 18 integer digits, 9 fractional
digits, and 29 bytes including sign and decimal point. Exponents, a
leading plus, a leading zero in a multi-digit integer part, a trailing fractional
zero, a bare decimal point, and over-scale or over-length tokens are invalid. Exact
numeric zero is valid only as `0`; `-0` is invalid. Negative nonzero values are valid
only where the field permits them, and every field fixes its sign and numeric range.
Exact comparison uses integer coefficient and scale or a trapping exact decimal
implementation, never string order, binary float, locale, or rounding. Shared known
vectors include valid `0`, `0.1`, `0.000000001`, `1.1`, and field-permitted `-0.1`;
numeric ordering for 2 versus 10 and -2 versus -10; and rejection of noncanonical
equivalents, excessive scale, and oversized tokens.

Latency and cold-start fields are nonnegative canonical decimal milliseconds named
`*_latency_ms` and `*_cold_start_ms`. Monotonic intervals and elapsed durations are
unsigned integer nanoseconds named `*_duration_ns`. Memory and sizes are unsigned
integers named `*_bytes`; counts are unsigned integers. Throughput gates use exact
processed-count and duration ratios. Any rounded decimal display is excluded from
digests, comparisons, and gates. When an optional display is emitted, round the exact
ratio to at most nine fractional places using round-to-nearest, ties-to-even, remove
trailing fractional zeros, and emit exact zero as `0`.

Persisted UTC instants use exact `YYYY-MM-DDTHH:MM:SSZ`: a valid Gregorian date,
two-digit time fields, no offset, fraction, leap second, or alternate spelling.
Validators parse and compare UTC instants, never their strings. Invalid calendar
dates, reversed intervals, forbidden future times, overflow, and a validity interval
one second over its bound are rejected. Durations use monotonic nanoseconds and are
never derived by subtracting wall-clock strings.

## Confirmed absence, probe uncertainty, and stable reason codes

A successful probe with `present=false` is confirmed absence. A probe with status
`unsupported` or `failed` and no value is unknown. Neither is converted to zero or a
passing fact, but they use different reason codes so an operator can distinguish a
missing device from an incomplete observation.

The following spellings are stable v1 `CandidateEvaluation` exclusion codes. The
32-code bound is per evaluation; it is not the size of this vocabulary.

| Group | Codes |
|---|---|
| Compatibility | `task_mismatch`, `prompt_mode_mismatch`, `class_vocabulary_mismatch`, `evaluation_dataset_mismatch`, `evaluation_vocabulary_mismatch`, `bundle_spec_mismatch`, `environment_mismatch`, `qualification_workload_mismatch`, `protocol_mismatch` |
| Lifecycle and trust | `test_only`, `bundle_disabled`, `bundle_revoked`, `maturity_disallowed`, `registry_untrusted`, `lifecycle_untrusted`, `screening_untrusted`, `screening_not_current_pass`, `support_profile_mismatch`, `support_profile_untrusted`, `support_profile_conflict`, `license_not_approved`, `license_not_allowed` |
| Execution | `network_required`, `isolation_required`, `isolation_unsupported`, `isolation_image_missing`, `isolation_policy_mismatch`, `unsafe_loader_on_host`, `compute_policy_mismatch`, `provider_not_allowed`, `precision_not_allowed`, `hardware_unavailable`, `hardware_probe_unknown`, `runtime_unavailable`, `runtime_probe_unknown` |
| Artifacts | `artifact_size_limit_exceeded`, `artifact_member_missing`, `artifact_member_mismatch`, `artifact_state_mismatch` |
| Evidence | `evidence_not_qualified`, `evidence_untrusted`, `evidence_inactive`, `evidence_revoked`, `evidence_expired`, `evidence_superseded`, `evidence_conflict`, `evidence_future_dated` |
| Metrics and gates | `requested_metric_unknown`, `ranking_metric_unknown`, `cold_start_unknown`, `cold_start_above_requirement`, `execution_mode_metric_mismatch`, `quality_gate_failed`, `repeat_throughput_gate_failed`, `sustained_fps_gate_failed`, `p95_latency_gate_failed`, `peak_rss_gate_failed`, `accelerator_memory_gate_failed` |
| Catalog | `catalog_only` |

`no_eligible_candidate` is the only top-level outcome code in v1; it is not a
`CandidateEvaluation` exclusion code. An abstained decision contains that top-level
code and keeps every explicit exclusion in its corresponding candidate evaluation.
A selected candidate has no hard-failure reason. Confidence probabilities are not
part of v1 because there is no defined calibration source.

## Catalog and site-qualified recommendations

Catalog metadata describes a candidate, public benchmark, or model card. It may be
shown as non-executable information with `catalog_only`, but it is not qualification
evidence and cannot satisfy a local SLO or quality gate.

A public YOLOZU recommendation requires `yolozu_managed` evidence and one exact
matching public support profile. A site-qualified recommendation uses reviewed
`site_managed` evidence for one exact site configuration. It may select locally with
`support_scope=site_qualified`, but it does not create a public support or endorsement
claim. Unseen private vocabularies or datasets require matching private site evidence;
public numbers are not substituted.

## Illustrative decisions

These are fictional policy examples, not schema fixtures, qualification evidence, or
current support claims.

A valid selected case can occur when the same fictional spec is pointed to by both a
matching Stable and matching Experimental channel. It is evaluated once and Stable
becomes the effective channel:

```json
{
  "status": "selected",
  "ranking_policy": "latency_first",
  "selected_bundle_id": "fixture_detector",
  "effective_channel": "Stable",
  "pointed_channels": ["Experimental", "Stable"],
  "matching_channels": ["Experimental", "Stable"],
  "support_scope": "public_qualified",
  "reason_codes": []
}
```

A text request with only fixed-class evidence abstains without choosing a fallback:

```json
{
  "status": "abstained",
  "ranking_policy": "latency_first",
  "selected_bundle_id": null,
  "support_scope": "none",
  "reason_codes": ["no_eligible_candidate"],
  "candidate_evaluations": [
    {
      "bundle_id": "fixture_fixed_detector",
      "rank_state": "excluded",
      "reason_codes": ["prompt_mode_mismatch"]
    }
  ]
}
```

The implemented `SelectionDecision` interface contract adds the complete bounded
digests, candidate evaluations, evidence identities, trace, and decision time. These
shortened policy examples are intentionally not valid schema instances.

## Non-goals

- No selector, model integration, model download, or benchmark run is made available
  by these interface contracts or the managed-output helper.
- No current algorithm is claimed to meet an accuracy, latency, throughput, memory,
  soft-real-time, or hardware-support objective.
- No implicit network access, dependency installation, model acquisition, training,
  TTA, TTT, video, streaming, tracking, OCR, or hard-real-time guarantee is added.
- The existing Stable predictions interface contract and validation/evaluation
  behavior are not changed.
