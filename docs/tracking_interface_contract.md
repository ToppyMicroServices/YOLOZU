# Tracking output interface contract

Status: contract-only P2 foundation. YOLOZU does not ship a tracker, tracking
adapter, tracking model, stream engine, qualified tracking bundle, or tracking
support claim.

Tracking output is separate from the existing predictions interface contract.
It does not add `track_id` or lifecycle fields to `predictions.json`. The
canonical schemas are:

- `docs/schemas/tracking_output_interface.schema.json`
- `docs/schemas/tracking_output_record.schema.json`

Their byte-identical packaged copies are under `yolozu/data/schemas/`. The
standard-library state and JSONL validators live in
`yolozu/contracts/tracking.py` and reuse the canonical Stream `FrameResult`
validator from `yolozu/adaptive/streaming.py`.

## Session and identity boundary

`track_id` is an unsigned JSON-safe integer from 1 through
9,007,199,254,740,991. It is only a lifecycle handle inside one stream session.
It is not a person, device, biometric, camera, or durable identity. Person
identification, face recognition, biometric inference, cross-camera or
cross-session identity linking, and persistent identity databases are outside
this interface contract. A reset ends the old namespace before a new session
starts, so a numeric ID may restart only in the new session.

The immutable output interface pins these exclusions and the task, detector
interval, per-frame result cap, and prediction/lost age limits. It also pins the
fixed caps of 1,000 active IDs, 1,000,000 unique IDs per session, and a
nonresetting 1,000,000-unit job budget. The job budget charges one unit for each
emitted tracking row and one additional unit when an ID first receives state in
its session. Ending or resetting releases active state but never refunds this
budget. Future tracking job, bundle, workload, and qualification/report records
must bind the exact `max_prediction_frames` and `max_lost_frames` values. This
foundation does not add those routing or evidence surfaces.

## Rows and lifecycle

`tracking_results.jsonl` contains canonical `tracking_result` rows and one
bounded `tracking_session_termination` record for every reset, EOF,
cancellation, or terminal failure. Rows for one frame are ordered by ascending
`track_id`. Every non-ended ID has exactly one row on every processed source
frame, and one ID cannot occur twice in a frame.

| Prior state | Allowed next row |
|---|---|
| New ID | `observed` |
| `observed` | `observed`, bounded `predicted`, bounded `lost`, or `ended` |
| `predicted` | `observed`, bounded `predicted`, bounded `lost`, or `ended` |
| `lost` | `observed`, bounded `lost`, or `ended` |
| `ended` | none in the same session |

An observed row has a source observation reference and an explicitly
tracker-adjusted estimate. Its optional observation copy is the complete
`class_id`, `score`, `bbox`, and `mask` tuple and must exactly equal the
referenced source result. A predicted row has only a tracker-prediction
estimate. Lost and ended rows are lifecycle-only and cannot carry detector
links, copied class/score/geometry/mask fields, or a track estimate. Detection,
estimate, and prediction confidence remain separate sources; no combined
confidence is inferred.

`max_prediction_frames=0` forbids predicted rows and `max_lost_frames=0`
forbids lost rows. Consecutive ages start at one and must match the state
history exactly. Ended IDs never reappear in that session. Lost IDs may reappear
as observed until explicitly ended or the session terminates.

## Source linkage and retained files

`source_frame_index` is job-wide and never reused. `session_index` changes only
after a reset, while `session_frame_index` starts at zero in each session and
increments for every processed source frame. Detector cadence uses only
`session_frame_index % detector_interval`.

The durable detector file retains every processed complete canonical
`FrameResult` exactly once, including empty and unmatched frames. A complete
tracking output declares exactly these regular files, plus referenced masks:

- `detector_frame_results.jsonl`
- `provenance.json`
- `stream_summary.json`
- `tracking_results.jsonl`
- zero or more `artifacts/masks/*.png` files

`checksums.json` is the fifth base file and is implicit rather than declared in
itself. No other regular output path is accepted.

The strict validator consumes the files in lockstep. It verifies a source
FrameResult's own digest and exact canonical bytes before validating any link to
that frame. `observation_ref.source_frame_result_digest` and
`source_result_index` then select exactly one ordered `task_results` item.
Dangling, cross-frame, duplicate, or mismatched-copy links fail. Individual
detector results may remain unreferenced. The aggregate validator parses the
exact canonical `stream_summary.json`, verifies its own digest, and derives the
processed-frame and detector-result counts from it. Either retained-file count
must match exactly, so removing an otherwise valid empty or unmatched frame is
not accepted.

FrameResult lines, contained detector results, tracking rows plus first state
allocations, and session terminations have independent accounting. The aggregate
validator takes the validated `StreamJobSpec`; it does not accept duplicate
caller-authored count or cap integers. That record supplies task, source rate,
decoded dimensions, the per-frame and total detector-result limits, and the
mask/file/byte limits. The tracking interface separately supplies the per-frame
tracking-row cap.

The legal configured maxima remain 10,000 mask artifacts, 10,004 files, and
4,294,967,296 bytes. They are independent ceilings. A tracking output uses five
base files, so its actual mask count is also limited by the file slots remaining
in that job. A job may still lawfully configure 10,000 masks and 10,004 files;
the validator does not require every configured maximum to fit simultaneously.
The incremental validator keeps one source frame, at most 1,000 tracking rows,
and at most the configured mask-reference set in memory. It does not construct a
whole-file DOM.

## Post-stage aggregate integrity

`validate_tracking_output_artifacts` accepts every declared regular output in
UTF-8 byte-sorted path order and accepts `checksums.json` separately. JSONL
iterators yield one complete canonical LF-terminated row at a time; the other
files can use arbitrary byte chunks. A finite empty file supplied directly as
`b""` is valid, including an empty detector file for a zero-frame terminated
session. An iterable that yields `b""` is rejected as a non-progress chunk.
Before accepting the output, the validator verifies:

- exact manifest fields, ordered paths, file count, per-file size and SHA-256,
  and total declared bytes;
- no manifest self-entry, duplicate, reordered, missing, extra, or undeclared
  file;
- the canonical Stream summary and its job task/source identity, plus every
  `FrameResult` task, source-rate schedule, decoded dimension, own digest, and
  per-frame/total detector-result cap;
- every tracking reference and state transition, and the configured tracking-row
  caps;
- exact agreement between all referenced masks and declared mask path, size,
  and SHA-256 metadata; and
- the configured shared mask, file, and byte caps before acceptance.

The checksum manifest covers every other declared regular file and only those
files. Its own bytes count toward the output-byte limit but are not listed in
`files`, `expected_paths`, `file_count`, or `total_bytes`. The function hashes
the supplied bytes and does not trust caller-reported checksums. The Stream
summary's `output_bytes` is the declared `total_bytes` excluding the checksum
manifest; the 4-GiB acceptance check includes the manifest bytes as well.

`provenance.json` is an exact canonical aggregate-only record. It contains no
frame, track, person, or device data. It binds
`tracking_output_interface_digest`, the `StreamJobSpec` digest, the Stream
summary digest, and the exact checksum-manifest SHA-256 values for both JSONL
files. It fixes `identity_scope=session_only` and
`contains_frame_or_identity_data=false`. Changing a source file and regenerating
only the checksum manifest therefore does not preserve the provenance binding.

This is a pure post-stage byte validator. It does not open paths, protect a
directory from links or time-of-check/time-of-use changes, make writes atomic,
or claim power-loss durability. A later runtime must use the repository's
managed-output transaction and feed the exact staged bytes to this validator
before publication.

## Atomicity

For each frame, all rows, transitions, references, ordering, timestamps, and
counter changes are checked before state is mutated. An over-limit or invalid
batch leaves the previous state unchanged. A later stream writer must append
only the returned normalized batch after this call succeeds; this module does
not write files. A session termination closes all
remaining IDs through one aggregate record rather than synthesizing up to 1,000
ended rows. Reset starts the next session; EOF, cancellation, and terminal
failure end the job. In the termination record, `active_track_count` counts
observed and predicted IDs, while `lost_track_count` counts lost IDs; their sum
cannot exceed 1,000.
