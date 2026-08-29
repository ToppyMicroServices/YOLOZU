# Schema governance

This document defines how YOLOZU evolves JSON artifact schemas without breaking comparability.

## Canonical schema location

The authoritative predictions JSON Schema lives at `docs/schemas/predictions.schema.json`
(JSON Schema draft 2020-12). The copies at `schemas/predictions.schema.json` and
`yolozu/data/schemas/predictions.schema.json` (packaged) must stay in sync - they are
overwritten from the canonical copy during release preparation.

The adaptive image routing v1 interface contracts use these canonical and packaged
byte-identical pairs. They do not have a third copy under `schemas/`:

- `docs/schemas/image_job_spec.schema.json` and
  `yolozu/data/schemas/image_job_spec.schema.json`
- `docs/schemas/qualification_workload_profile.schema.json` and
  `yolozu/data/schemas/qualification_workload_profile.schema.json`
- `docs/schemas/environment_profile.schema.json` and
  `yolozu/data/schemas/environment_profile.schema.json`
- `docs/schemas/algorithm_bundle_spec.schema.json` and
  `yolozu/data/schemas/algorithm_bundle_spec.schema.json`
- `docs/schemas/algorithm_bundle_registry.schema.json` and
  `yolozu/data/schemas/algorithm_bundle_registry.schema.json`
- `docs/schemas/algorithm_scout_sources.schema.json` and
  `yolozu/data/schemas/algorithm_scout_sources.schema.json`
- `docs/schemas/algorithm_scout_report.schema.json` and
  `yolozu/data/schemas/algorithm_scout_report.schema.json`
- `docs/schemas/bundle_lifecycle_record.schema.json` and
  `yolozu/data/schemas/bundle_lifecycle_record.schema.json`
- `docs/schemas/candidate_screening_record.schema.json` and
  `yolozu/data/schemas/candidate_screening_record.schema.json`
- `docs/schemas/candidate_isolation_probe.schema.json` and
  `yolozu/data/schemas/candidate_isolation_probe.schema.json`
- `docs/schemas/support_profile_spec.schema.json` and
  `yolozu/data/schemas/support_profile_spec.schema.json`
- `docs/schemas/support_profile_record.schema.json` and
  `yolozu/data/schemas/support_profile_record.schema.json`
- `docs/schemas/support_profile_set_proposal.schema.json` and
  `yolozu/data/schemas/support_profile_set_proposal.schema.json`
- `docs/schemas/lifecycle_rollback_bindings.schema.json` and
  `yolozu/data/schemas/lifecycle_rollback_bindings.schema.json`
- `docs/schemas/local_artifact_inventory.schema.json` and
  `yolozu/data/schemas/local_artifact_inventory.schema.json`
- `docs/schemas/qualification_report.schema.json` and
  `yolozu/data/schemas/qualification_report.schema.json`
- `docs/schemas/evidence_activation_record.schema.json` and
  `yolozu/data/schemas/evidence_activation_record.schema.json`
- `docs/schemas/screening_eligibility_observation.schema.json` and
  `yolozu/data/schemas/screening_eligibility_observation.schema.json`
- `docs/schemas/support_profile_eligibility_observation.schema.json` and
  `yolozu/data/schemas/support_profile_eligibility_observation.schema.json`
- `docs/schemas/selection_decision.schema.json` and
  `yolozu/data/schemas/selection_decision.schema.json`
- `docs/schemas/frame_result.schema.json` and
  `yolozu/data/schemas/frame_result.schema.json`
- `docs/schemas/stream_job_spec.schema.json` and
  `yolozu/data/schemas/stream_job_spec.schema.json`
- `docs/schemas/stream_workload_profile.schema.json` and
  `yolozu/data/schemas/stream_workload_profile.schema.json`
- `docs/schemas/stream_summary.schema.json` and
  `yolozu/data/schemas/stream_summary.schema.json`
- `docs/schemas/stream_qualification_report.schema.json` and
  `yolozu/data/schemas/stream_qualification_report.schema.json`
- `docs/schemas/stream_selection_decision.schema.json` and
  `yolozu/data/schemas/stream_selection_decision.schema.json`
- `docs/schemas/tracking_output_interface.schema.json` and
  `yolozu/data/schemas/tracking_output_interface.schema.json`
- `docs/schemas/tracking_output_record.schema.json` and
  `yolozu/data/schemas/tracking_output_record.schema.json`

The adaptive and streaming standard-library Python validators live under
`yolozu/adaptive/`; tracking validation lives under `yolozu/contracts/`. JSON Schema
fixes the transport shape and bounds. Python validation additionally performs the
normative semantic checks that JSON Schema cannot express directly, including NFKC
prompt normalization, aggregate UTF-8 limits, canonical digest verification, ordered
input indices, and privacy-safe environment fingerprint projection. These records
define the Experimental adaptive-routing interface. Their presence, or a read-only
recommendation response, is not model availability, execution, or qualification
evidence.

Candidate sdist/wheel and installed-surface tests assert that the packaged schemas,
empty adaptive SSOT, adaptive Python modules, and generated MCP reference are
present outside the checkout with `PYTHONPATH` cleared. The exact inventory and
default abstention/rejection outcomes are recorded in the
[installed-artifact verification report](../reports/adaptive_routing_installed_verification_2026-08-26.md).
This packaging check is not real bundle qualification.

The source-controlled adaptive registry instances have one copy under the packaged
data tree: `yolozu/data/adaptive_routing/bundle_specs.json`,
`yolozu/data/adaptive_routing/bundle_lifecycle.jsonl`,
`yolozu/data/adaptive_routing/candidate_screening.jsonl`,
`yolozu/data/adaptive_routing/support_profiles.jsonl`, repository-reviewed
reports under
`yolozu/data/adaptive_routing/qualification_reports/<report_id>/qualification_report.json`,
and `yolozu/data/adaptive_routing/evidence_activation.jsonl`. These files are
their own source and packaged form; there is no docs copy or serialized mutable
projection. A report ID is a validated ASCII component, never a path. Site-managed
reports and activation streams stay under an explicit site-confined root and are
not copied into the repository package. The default registry contains three
Candidate baselines matching the model zoo; the screening stream has two current
hold records, while the public support and activation streams remain empty and
there are no public qualification reports. Each baseline
has `execution_binding.status=unbound`, so the records are neither selectable nor
executable. Runtime projections must be
derived from validated immutable records and complete, strictly ordered event
chains.

`recommend_image_pipeline` consumes these governed records through a bounded,
read-only MCP service. It returns the SelectionDecision interface contract and
privacy-safe aggregate metadata. Custom registry, screening, and evidence roots
remain operator-asserted. A custom screening record cannot satisfy the managed-pass
gate even if it claims otherwise. The Candidate-only packaged registry produces abstention with
`maturity_disallowed`. The service
does not mutate any schema instance or source-controlled stream.

`process_images` consumes that complete selected decision rather than a caller-
supplied model identity. It rebuilds the governed records and stable digests,
retains no-follow input/artifact handles through execution, and defaults to a
runner-free no-write dry-run. Explicit execution publishes only the stable
predictions interface contract, local provenance, the code-owned checksum
manifest, and referenced masks through `ManagedOutputTransaction`.

`activate-qualification-evidence` derives trust from this retained workflow or
from an exact code-owned local qualifier output. It never accepts trust from a
report field. The default is dry-run; `--approve` is required for one atomic
append. Local output is limited to `site_managed` and `site_qualified`. Arbitrary
workspace JSON remains `operator_asserted` and nonselectable. Checksums protect
integrity after creation, not provenance against an adversarial local operator.
Repository-managed report directories retain tracked `public_inputs.json`,
`protocol.json`, `qualification_report.json`, `reproduce.txt`, and a
`checksums.json` that covers every other file exactly.

`review-image-pipeline-support-profiles` consumes the canonical
`docs/schemas/support_profile_set_proposal.schema.json` shape and requires exact
`canonical_json_v1` bytes plus LF. It may append only to
`yolozu/data/adaptive_routing/support_profiles.jsonl`; installed packages read those
same bytes. No mutable projection copy exists. A successful review records dormant
scope and does not change lifecycle-advertised support.

`update-image-pipeline-lifecycle` consumes the immutable registry, append-only
lifecycle and support-profile streams, and, for a non-`none` rollback, the canonical
`docs/schemas/lifecycle_rollback_bindings.schema.json` shape. The bindings must be
the complete ordered repository-managed activation set derived for the exact
historical advertised profiles. The command is dry-run by default. Approval can
append one reviewed lifecycle record only; it cannot rewrite earlier records,
bundle specs, artifacts, evidence, or support-profile definitions. Global revocation
is terminal. Channel rollback changes only the named Experimental or Stable pointer.

`promote-image-pipeline` reuses those canonical records and the complete ordered
managed activation-binding shape. Promotion lifecycle records audit-bind the exact
source/target pointers, explicit rollback readiness, and Stable comparator/drill
status. `SupportProfileSpec.advertised_constraints.max_p99_latency_ms` is an
optional backward-compatible absolute gate and is mandatory in the stricter Stable
promotion policy. This adds no derived registry or support projection and does not
change the schema version of existing v1 records.

`load_algorithm_bundle_registry` accepts only the exact packaged registry/lifecycle
pair above or an explicit workspace-confined directory containing those two exact
basenames. The packaged pair is `yolozu_managed`; a custom pair is always
`operator_asserted`, regardless of issuer claims inside its JSON. Loading validates
the complete bounded input and deterministic bundle order without fetching assets,
importing a model runtime, or making a bundle selectable.

Managed output trees use the code-owned `checksums.json` interface contract in
`yolozu/adaptive/managed_output.py`. The manifest lists every other declared file
in validated UTF-8 byte order and never lists itself. It has no caller-authored
schema location or mutable projection: `ManagedOutputTransaction` generates the
canonical bytes and the same helper validates them before publication, force
replacement, cleanup, or recovery.

## Scope

This governance applies to wrapped prediction-style payloads that include:

- `schema_version`
- `predictions`
- optional `meta`

Current version:

- `current_schema_version = 1`
- `minimum_supported_schema_version = 1`
- `current_entry_schema_version = 2`
- `minimum_supported_entry_schema_version = 1`

## Lifecycle rules

1. Backward-compatible changes do **not** bump `schema_version`.
   - Examples: adding optional fields, adding optional metadata keys, expanding accepted optional aliases.
2. Breaking changes **must** bump `schema_version`.
   - Examples: removing required fields, changing required field type/meaning, changing required coordinate conventions.
3. Validators reject unknown future versions.
   - If a payload declares `schema_version > current_schema_version`, validation fails until YOLOZU is upgraded.
4. Legacy wrapped payloads without `schema_version` are accepted with warning in compatibility mode.
5. The bundled YOLO-runtime, Detectron2, MMDetection, and YOLOX prediction
   exporters declare the current wrapper version on generated wrapped artifacts
   and the current version on every generated entry. Version omission is a
   compatibility path for older input artifacts, not the format for new output
   from these four exporters.

## Compatibility policy

- `schema_version` present:
  - must be integer
  - must satisfy `minimum_supported <= schema_version <= current`
- `schema_version` missing in wrapped payload:
  - accepted for backward compatibility
  - validator emits warning and treats payload as legacy mode
- entry-level `schema_version` missing:
  - treated as legacy entry v1
  - canonicalization migrates to entry v2

## Breaking-change process (checklist)

When proposing a schema-breaking change:

1. Open an RFC issue describing:
   - old vs new interface contract
   - expected migration cost
   - affected tools/adapters/protocols
2. Add/update migration utility for old artifacts.
3. Add golden test vectors for old and new versions.
4. Add CI gate coverage:
   - current version must pass
   - future/unsupported versions must fail
5. Update docs:
   - this governance doc
   - schema-specific docs (e.g., predictions schema pages)
   - release notes with migration steps
6. Update release/process metadata:
   - add a `Contract change` note in `CHANGELOG.md`
   - include baseline lifecycle impact (`dataset_hash`, `weights_hash`, baseline version path)
   - update PR checklist entries for baseline updates

See also: [RFC workflow + golden compatibility assets](rfc_workflow.md).

## Migration steps template

Use this template in release notes when introducing schema `N+1`:

1. Identify old artifacts (`schema_version == N` or missing).
2. Run migration tool to emit `schema_version == N+1` payloads (predictions entry v1->v2: `yolozu predictions migrate --from v1 --to v2 ...`).
3. Validate migrated artifacts with `yolozu validate ...` in CI.
4. Re-run evaluation protocol on migrated artifacts and compare metrics.
5. Remove compatibility mode only after deprecation window ends.

## CI enforcement

CI includes a schema compatibility gate that asserts:

- v1 wrapped payloads pass validation
- v2 wrapped payloads fail while current is v1
- entry v1 payloads are migrated to entry v2 during canonicalization/load paths
- YOLO-runtime, Detectron2, MMDetection, and YOLOX dry runs declare wrapper v1
  and entry v2 and pass strict validation without legacy-migration warnings

This prevents silent schema drift and guarantees interface-contract-first behavior.

In addition, golden compatibility assets are versioned under `baselines/golden/` and validated by:

- `python3 tools/check_golden_compatibility.py`

This gate pins protocol + golden artifact hashes and fails when schema/protocol behavior changes without coordinated golden updates.

## Schema Browser Coverage

Use this table as the maintained schema browser until a generated web page is
introduced. Each row points to the canonical schema or the closest current
schema surface for that artifact family.

| Artifact family | Canonical schema | Main docs | Notes |
|---|---|---|---|
| Predictions | `docs/schemas/predictions.schema.json` | [`predictions_schema.md`](predictions_schema.md) | Packaged copies live in `schemas/` and `yolozu/data/schemas/`. |
| Adaptive vision roadmap projection | `docs/schemas/adaptive_vision_roadmap.schema.json` | [`roadmap.md`](roadmap.md), [`../reports/adaptive_vision_roadmap.md`](../reports/adaptive_vision_roadmap.md) | The byte-identical packaged schema and JSON projection describe future scope, not implementation or qualification evidence. |
| Adaptive image request, workload, and environment | `docs/schemas/image_job_spec.schema.json`, `docs/schemas/qualification_workload_profile.schema.json`, `docs/schemas/environment_profile.schema.json` | [`adaptive_image_routing.md`](adaptive_image_routing.md), [`doctor_diagnostics.md`](doctor_diagnostics.md) | Byte-identical packaged schemas accompany standard-library validators. `doctor` produces EnvironmentProfile; recommendation and processing consume the typed request/workload records, but none advertises a selectable model. |
| Adaptive bundle, lifecycle, rollback, and support-profile records | `docs/schemas/algorithm_bundle_spec.schema.json`, `docs/schemas/algorithm_bundle_registry.schema.json`, `docs/schemas/bundle_lifecycle_record.schema.json`, `docs/schemas/lifecycle_rollback_bindings.schema.json`, `docs/schemas/support_profile_spec.schema.json`, `docs/schemas/support_profile_set_proposal.schema.json`, `docs/schemas/support_profile_record.schema.json` | [`adaptive_image_routing.md`](adaptive_image_routing.md) | Immutable bundle facts are separate from append-only lifecycle and reviewed dormant support scope. Review proposals and rollback bindings must cover their complete ordered sets. The canonical public lifecycle remains Candidate-only; implementation does not imply an available model. |
| Adaptive candidate screening | `docs/schemas/candidate_screening_record.schema.json` | [`adaptive_image_routing.md`](adaptive_image_routing.md), [`adaptive_candidate_screenings_2026-08-29.md`](adaptive_candidate_screenings_2026-08-29.md), [`algorithm_intake/README.md`](algorithm_intake/README.md) | Non-executing mechanical facts and human review produce pass, hold, or reject. The sole packaged stream has two current hold records and no pass; workspace input remains operator-asserted, and screening output is not a bundle registry. |
| Adaptive artifact and qualification evidence | `docs/schemas/local_artifact_inventory.schema.json`, `docs/schemas/qualification_report.schema.json`, `docs/schemas/evidence_activation_record.schema.json` | [`adaptive_image_routing.md`](adaptive_image_routing.md) | Inventory, measurement, and reviewed activation remain separate. The Experimental qualifier emits an unactivated report; the activation command defaults to dry-run and requires explicit review plus approval for an atomic append. The packaged Candidate baselines have unbound execution, and the runner map and public evidence storage remain empty, keeping the default nonselectable and non-executable. |
| Adaptive selection observations and decisions | `docs/schemas/screening_eligibility_observation.schema.json`, `docs/schemas/support_profile_eligibility_observation.schema.json`, `docs/schemas/selection_decision.schema.json` | [`adaptive_image_routing.md`](adaptive_image_routing.md) | File-free typed observations and complete selected/abstained records expose every candidate reason. The pure selector consumes only validated in-memory values. An unpointed excluded catalog entry uses an empty pointed-channel set and a null support observation instead of invented evidence. MCP recommendation returns this interface contract; pinned processing accepts only a complete selected record and repeats current-state validation. |
| OCR result boundary | `docs/schemas/ocr_bundle_interface.schema.json`, `docs/schemas/ocr_result.schema.json` | [`ocr_interface_contract.md`](ocr_interface_contract.md) | Contract-only model-free validation keeps recognized text out of detection labels, maps only bounded runner language/script indices, stamps pinned component provenance in core, and exposes no recognized content in summaries. No OCR adapter or support claim is included. |
| Streaming result and qualification boundary | `docs/schemas/frame_result.schema.json`, `docs/schemas/stream_job_spec.schema.json`, `docs/schemas/stream_workload_profile.schema.json`, `docs/schemas/stream_summary.schema.json`, `docs/schemas/stream_qualification_report.schema.json`, `docs/schemas/stream_selection_decision.schema.json` | [`streaming_interface_contract.md`](streaming_interface_contract.md) | Contract-only validators fix canonical per-frame output, privacy-bounded summaries, exact source-rate and drop accounting, an unactivated qualification-report record shape, and explicit selection or abstention. No decoder, live runner, schedule result, or support claim is included. |
| Tracking output boundary | `docs/schemas/tracking_output_interface.schema.json`, `docs/schemas/tracking_output_record.schema.json` | [`tracking_interface_contract.md`](tracking_interface_contract.md) | Contract-only state validation binds session-scoped track transitions to validated FrameResult records with explicit limits and canonical JSONL lockstep checks. No tracker, model, runner, or support claim is included. |
| Detection / COCO eval reports | `docs/schemas/coco_eval_report.schema.json`, `docs/schemas/eval_suite_report.schema.json` | [`python_api.md`](python_api.md), [`yolo26_eval_protocol.md`](yolo26_eval_protocol.md), [`evaluation_protocol_template.md`](evaluation_protocol_template.md) | The COCO report schema is also packaged at `yolozu/data/schemas/coco_eval_report.schema.json`; protocol hash must be recorded before fair comparison. |
| Segmentation dataset/eval | `docs/schemas/seg_dataset.schema.json`, `docs/schemas/seg_eval_report.schema.json` | [`predictions_schema.md`](predictions_schema.md) | Dataset and eval schemas are separate from predictions payloads. |
| Training handoff | `docs/schemas/training_run_summary.schema.json`, `docs/schemas/training_handoff.schema.json` | [`training_orchestration.md`](training_orchestration.md) | Handoff JSON carries next steps for resume/export/eval/parity. |
| Fine-tuning qualification | `docs/schemas/finetune_lane_qualification.schema.json` | [`training_capability_matrix.md`](training_capability_matrix.md), [`external_finetune_smoke.md`](external_finetune_smoke.md) | Separates actual training from projection and records dependency, provenance, checkpoint, metric-scope, and hold boundaries. |
| SynthGen sample | `schemas/synthgen_sample.schema.json` | [`synthgen_contract.md`](synthgen_contract.md), [`synthgen_repo_integration.md`](synthgen_repo_integration.md) | SynthGen intake remains an external generator handoff. |
| Research reports | `docs/schemas/research_lane_report.schema.json`, `docs/schemas/artifact_research_qualification.schema.json`, `docs/schemas/sdft_continual_qualification.schema.json`, `docs/schemas/bop19_tless_pose_qualification.schema.json`, `docs/schemas/compatible_host_external_runtime_qualification.schema.json`, `docs/schemas/research_note.schema.json` | [`research_lanes.md`](research_lanes.md), [`continual_learning.md`](continual_learning.md), [`bop_tless_protocol.md`](bop_tless_protocol.md), [`research_note_template.md`](research_note_template.md) | Research results stay separate from stable evaluation reports; artifact, SDFT-style continual, and official BOP qualifications require repeated evidence and an explicit promotion boundary. Compatible-host runtime qualification separately requires training, checkpoint, resource, and structural handoff evidence while retaining an availability-only boundary. |
