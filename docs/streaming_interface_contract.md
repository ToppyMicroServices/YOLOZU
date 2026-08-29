# Bounded local streaming interface contract

Status: contract-only P2 foundation. YOLOZU does not ship a stream decoder,
camera provider, stream loop, tracking implementation, qualified stream bundle,
or stream support claim.

The Python validators live in `yolozu/adaptive/streaming.py`. The canonical JSON
Schemas are:

- `docs/schemas/stream_job_spec.schema.json`
- `docs/schemas/stream_workload_profile.schema.json`
- `docs/schemas/frame_result.schema.json`
- `docs/schemas/stream_summary.schema.json`
- `docs/schemas/stream_qualification_report.schema.json`
- `docs/schemas/stream_selection_decision.schema.json`

Byte-identical packaged copies use the same basenames under
`yolozu/data/schemas/`. These records are separate from the static-image
`ImageJobSpec`, `QualificationReport`, and `SelectionDecision`. A static record is
not stream evidence and cannot be passed to a future stream executor.

## V1 source boundary

V1 describes only a workspace-confined local MP4 file or one capability request
using the reserved provider ID `contract_fixture_camera_v1`. Allowlisting this
contract-test ID does not implement a provider or establish camera availability
or support. The job and workload contain no filename, device identifier,
credential, eligible-device observation, or URL. RTSP and every other network
source are invalid.

The MP4 profile is H.264/AVC Constrained Baseline, Main, or High, level 3.0
through 5.1, 8-bit 4:2:0. The decoder-policy digest covers the exact backend and
hard process-tree enforcement plus the fixed reference-frame, decoded-picture
buffer, GOP, sample, NAL, probe-byte, atom-count, and nesting limits. A future
runtime must reject unsupported tracks, profiles, timestamps, edit lists,
fragmentation, encryption, external references, parser bombs, and an
unenforceable decoder memory limit. The interface contract does not itself open
or parse media.

A camera source binds only provider ID, pixel format, dimensions, and exact rate.
It never binds a physical identity. Preflight must enumerate the requested
capability and record the observed eligible count, exactly one, in the typed
local source-digest preimage. Execution must repeat the enumeration immediately
before opening the camera. It must fail if that observation is no longer exactly
one. Zero or multiple eligible devices is `camera_binding_ambiguous`.

The source rate is a reduced rational. Its numerator is 1 through 240,000, its
denominator is 1 through 1,001, and its exact value is 0.1 through 240 frames per
second. The schedule is:

```text
due(i) = start + i * source_rate_den / source_rate_num
```

Implementations compare the integer cross-products. They must not accumulate a
binary-float period. `max_frames` must cover the requested duration at the full
source rate; V1 never intentionally samples frames. Full-speed MP4 processing is
only a batch diagnostic and cannot produce stream qualification evidence.

## Admission and bounded output

The governed latency interval is
`stream_due_to_callback_enqueue_v1`. It begins at the rational due time, before
buffer acquisition or decode, and ends only after strict result validation,
bounded mask publication, and successful callback-buffer enqueue. A device time
may be retained in user-owned frame output as a diagnostic. It cannot replace the
governed start.

Before allocation, the worst-case decoded reservation is
`width * 4 * height`. The queue is bounded independently by frame count and
decoded bytes. `block` waits without allocating another decoded frame and cannot
record a drop. `drop_oldest` evicts the oldest source indices until both bounds
can admit the worst-case reservation. Processed, dropped, and
failed-unaccounted counts must cover every scheduled frame. Any
failed-unaccounted frame makes qualification ineligible.

The decoded stride policy is fixed to `width * 3` through `width * 4` bytes. A
camera job also binds a fixed caller-owned ingress pool. Its frame capacity must
equal the queue capacity, and its byte reservation must cover that many
worst-case decoded frames. These fields are part of the job and workload
digests; they do not claim that a camera pool exists today.

V1 also binds timeouts for source probe, camera open and first frame, runner probe
and load, each decode and predict operation, each output step, close, and the
cancellation grace period. Timeout or forced cancellation is a failed
termination. It cannot qualify. These are interface-contract deadlines only;
this module contains no cancellation or runtime implementation.

The authoritative drop fraction is the exact integer ratio:

```text
dropped / (processed + dropped)
```

A zero denominator is unknown. The configured maximum is a canonical reduced
rational and is compared by integer cross-multiplication. The optional display is
rounded to six decimal places with ties-to-even and is excluded from identity and
gating. `max_consecutive_drops` is a separate hard gate.

V1 caps the frame queue at 64 items and 512 MiB, the callback queue at 64 items
and 64 MiB, one stream at 864,000 scheduled frames, one frame at 1,000 task
results, all frames at 1,000,000 task results, masks at 10,000 files and 64 MiB
each, all declared files at 10,004, and output at 4 GiB. Mask data is written in
at most 1-MiB chunks before its small reference is placed in a FrameResult. These
are interface limits, not evidence that a current decoder or queue enforces them.

## FrameResult

One processed source index produces exactly one `FrameResult`, including an empty
`task_results` array. A dropped frame has no FrameResult. Array position is the
source-result index used by the tracking extension.

Each task result has an integral class ID, CanonicalDecimalV1 score, normalized
`x1,y1,x2,y2` box, and either a null mask or one bounded managed mask reference.
Object detection requires a null mask. Instance segmentation requires the pinned
`png_binary_mask_v1` reference with exact frame dimensions, byte size, and
SHA-256. This strict stream shape does not change the Stable predictions interface
contract.

`frame_result_digest` is SHA-256 over `canonical_json_v1` of the complete validated
record with only that own-digest field omitted. Validation creates one canonical
byte sequence. `FrameResult.canonical_bytes()` returns it, and
`FrameResult.canonical_line()` returns the same bytes followed by one LF for
`stream_results.jsonl`. A callback must receive those same canonical record bytes;
it must not reserialize a different object. Downstream
`source_frame_result_digest` means only this own digest.

`validate_stream_output_artifacts` validates a complete managed output without
opening or writing files. `stream_results.jsonl` has one canonical LF-terminated
FrameResult line per processed frame. `stream_summary.json`, `provenance.json`,
and every referenced mask must appear exactly once in byte-sorted
`checksums.json`. The manifest never lists itself. Declared paths, counts, sizes,
and SHA-256 values must match the supplied bytes exactly; missing, extra,
self-listed, duplicate, non-progress, and tampered inputs fail closed.

The summary's `output_file_count` includes `checksums.json`. Its `output_bytes`
is the exact sum of files listed by the manifest and therefore excludes the
manifest itself, avoiding a self-referential byte count. The aggregate validator
still applies the 4-GiB job cap to that sum plus the manifest bytes. It also
applies the job's result, mask, per-mask, and file caps.

Dropped frames have no FrameResult, so their count and maximum consecutive run
cannot be derived from JSONL rows. Aggregate validation therefore requires the
bounded internal `dropped_source_frame_indices` observation and recomputes both
values. It is deliberately absent from the summary, provenance, evidence, and
other public artifacts. The validator does not accept a caller's aggregate drop
claims as a substitute.

## Summary, evidence, and decisions

`StreamSummary` is aggregate user-owned output. It records complete frame
accounting, queue and callback high-water marks, latency percentiles, counts,
output limits, termination, and exact bundle/evidence identities. It contains no
raw frames or copied frame rows and is not qualification evidence.

`StreamQualificationReport` is a separate evidence kind. It binds the immutable
bundle and artifacts, environment, exact stream workload, protocol, admission and
decoder policies, governed latency interval, whole-job memory collector, optional
quality identity, and aggregate summary. Qualification requires at least 600
seconds of sustained measurement after the declared warm-up frame count. The
sustained section declares its source-index start, exclusive end, scheduled,
processed, dropped, and failed-unaccounted counts. It is an internally complete
subset of the run and excludes warm-up and post-section drain; its aggregate
counts and duration need not equal the whole-run summary. Authoritative FPS is
`processed_frames * 1_000_000_000 / sustained_duration_ns` and is gated by exact
integer arithmetic. Unknown memory coverage, a timeout, a partial interval, a
failed-unaccounted frame, or either failed drop gate cannot qualify.

For its half-open source window, the scheduled count is exactly
`ceil(source_rate_num * sustained_duration_ns /
(source_rate_den * 1_000_000_000))`. The exclusive end is the start plus that
count. A report cannot add or omit a cadence tick while retaining the same
duration.

When a quality requirement exists, the report repeats its exact metric,
direction, threshold, dataset, protocol, vocabulary, and task identity. The
validator derives `passed` or `failed` directly from CanonicalDecimalV1 measured
value and threshold comparison. A missing value derives `unknown`; a caller
cannot choose a different status.

Whole-job RSS covers the orchestrator, decoder, runner and descendants, camera and
frame pools, both queues, masks, and output staging from source start through
teardown. Accelerator memory covers every participating process and device.
Runner-only or incomplete observations remain unknown.

`StreamSelectionDecision` has `decision_kind=local_stream`. It contains the
sensitive local-only `stream_source_digest` and cannot be substituted with a
static or tracking decision. For MP4, a future preflight derives the digest from
source kind, exact byte length, and SHA-256 of bytes read from one pinned handle.
For camera, it covers only the provider/capability request and eligible count one.
It never includes a filename or persistent device ID.

The source digest may appear only in the returned local decision and user-owned
local provenance. It must not enter qualification reports, telemetry, logs,
support records, or public summaries.

`stream_source_preimage` is equally sensitive and local-only. For MP4 it exposes
the exact byte length and SHA-256 of the pinned file bytes. For camera it exposes
the requested provider/capability facts and the observed eligible count. It may
be retained only inside the returned local decision needed to verify the digest;
it must not enter qualification evidence, logs, telemetry, support records, or
public summaries. Local provenance retains the derived digest, not this preimage.

## Privacy and scope

Raw frames, recognized content, source credentials, filenames, camera identifiers,
and per-frame records must not enter aggregate summaries, logs, telemetry, or
public evidence. FrameResult rows are user-owned local output. Activity or fixture
success is not proof of adoption, performance, or support.

This contract defines a soft-real-time SLO. It is not a hard-real-time guarantee.
It adds no stream engine, camera access, codec, tracker, OCR path, network access,
model download, qualification result, selectable bundle, or Stable behavior.
