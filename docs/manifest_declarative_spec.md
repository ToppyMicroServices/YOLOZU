# Declarative tool manifest spec (phase 1)

This document freezes **required keys** and **rule boundaries** for `tools/manifest.json` in the YOLOZU declarative-manifest rollout.

## Scope

- Target file: `tools/manifest.json`
- Target entries: every item in `tools[]`
- Goal: make each tool entry self-descriptive for inputs, side effects, outputs, contracts, and runnable examples.

## Required fields for every tool entry

Each `tools[]` item MUST include:

- `id` (stable, lowercase identifier)
- `entrypoint` (repo-relative script path)
- `runner` (`python3` or `bash`)
- `summary` (human-readable purpose)
- `maturity` (`stable`, `experimental`, or `research`)
- `platform` object with `cpu_ok`, `gpu_required`, `macos_ok`, `linux_ok`
- `inputs` array (can be empty, but field must exist)
- `effects` object with `writes` and `fixed_writes` arrays
- `outputs` array (can be empty, but field must exist)
- `examples` with at least one runnable command

## Top-level AI surface sets

`ai_surfaces` is the machine-readable SSOT for public integration boundaries:

- `mcp_live`: exact canonical ids registered by the live MCP server
- `guaranteed_ai_safe`: deterministic lightweight guarantee
- `config_review`: in-process config generation/review subset
- `actions_public`: canonical operations shared with the Actions API

Each set declares ordered, unique `tool_ids` and a short `availability`
boundary. Generated MCP/Actions references and compact discovery must derive
these classifications from the source and packaged manifests.
Compact discovery exposes explicit `guaranteed_mcp_tools` and
`live_mcp_tools` fields. The compatibility field `supported_mcp_tools` retains
its historical meaning as the guaranteed subset.
In `ids_only` mode, `manifest_tools` is the filtered selection rather than the
entire registry, and expanded surface lists are replaced by bounded
`surface_counts`.
The generated MCP reference also supplies the exact JSON input schema and
function summary for every live id and is packaged with the wheel. Surface
membership is not a maturity claim: a live or guaranteed id without an
explicit matching `tools[].maturity` remains `maturity: null` with
`maturity_source: unclassified`. Maturity/tag filters match explicit metadata
only and report excluded unclassified counts in `filter_diagnostics`.

## Input declaration rules

For each `inputs[]` item:

- `name`: non-empty string
- `kind`: one of `file`, `dir`, `string`, `number`, `json`, `stdout`
- `required`: boolean
- `flag`: `--kebab-case` CLI flag when user-provided through CLI
- `default`: present when optional input has deterministic fallback behavior

Boundary:

- Internal-only values (derived from config/runtime) may omit `flag`, but must remain documented in `description`.

## Effects declaration rules

Every tool MUST declare side effects in `effects`:

- `effects.writes[]` for path(s) driven by input flags
  - required keys: `flag`, `kind`, `scope`, `description`
  - `kind`: `file` or `dir`
  - `scope`: `path` or `tree`
- `effects.fixed_writes[]` for deterministic writes not controlled by an output flag
  - required keys: `path`, `kind`, `scope`, `description`

Boundary:

- No undeclared write paths are allowed except explicitly approved cases using `effects.allow_unknown_flags=true`.

## Output declaration rules

For each `outputs[]` item:

- `name`: stable output identifier
- `kind`: one of `file`, `dir`, `string`, `number`, `json`, `stdout`
- `description`: what is produced
- `default`: required for deterministic default output paths

Boundary:

- If a tool writes multiple artifacts, each publishable artifact must appear in `outputs[]`.

## Contracts and docs

When applicable:

- `contracts.consumes[]` / `contracts.produces[]` must reference ids in top-level `contracts`.
- `docs[]` should point to repo-relative documentation files describing protocol/usage.
- `contract_outputs` should map produced contract ids to matching `outputs[].name`.

## Exact release-version examples

An `examples[].command` that pins `yolozu==VERSION` must declare how release automation treats it:

- `release_version_policy: current`: the version and matching underscore-form path token advance with the next release.
- `release_version_policy: historical`: the exact version remains fixed and `release_version_evidence` must identify a repo-relative evidence file (an optional `#anchor` is allowed).

Unclassified exact-version pins fail manifest and release metadata validation. Generic examples such as `yolozu==VERSION` are not exact pins and do not need a policy.

## Identifier and path constraints

- `id` matches `^[a-z0-9][a-z0-9_\-]*$`
- All manifest paths are repo-relative
- Paths must not include `..`
- Referenced files in `entrypoint`, `docs[]`, `contracts.*.schema` must exist

## Validation boundaries (phase split)

Phase 1 (this spec freeze):

- Document required fields and constraints
- Use `python3 tools/validate_tool_manifest.py` as baseline structure validation

Phase 2 (enforcement expansion):

- Strengthen validator to require presence of `inputs`, `effects`, `outputs`, `platform`, and at least one example for all tools
- Add regression tests for failure-path coverage
- Add CI gate to block non-compliant manifest updates

Current validator supports strict declarative checks with:

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
```

## Authoring checklist

For any new or modified tool entry:

1. Declare all CLI inputs in `inputs[]`.
2. Declare all write side effects in `effects.writes[]` / `effects.fixed_writes[]`.
3. Declare publishable artifacts in `outputs[]`.
4. Add at least one runnable command in `examples[]`.
5. Classify exact `yolozu==VERSION_NUMBER` examples as current or evidence-backed historical pins.
6. Add `contracts`/`contract_outputs` mappings where contracts exist.
7. Run `python3 tools/validate_tool_manifest.py --manifest tools/manifest.json`.
