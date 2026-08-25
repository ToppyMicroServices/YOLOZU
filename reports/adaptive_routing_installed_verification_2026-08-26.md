# Adaptive routing installed-artifact verification

Verification date: 2026-08-26 (Asia/Tokyo)

Bead: `YOLOZU-ll2.81.1.14`

## Current-state addendum

This report records the earlier foundation state verified for
`YOLOZU-ll2.81.1.14`. Later on 2026-08-26, `YOLOZU-ll2.81.2.1` registered the three
existing model-zoo records as non-promoted Candidate metadata. They remain
non-executable and unqualified because their execution binding is unbound and no
adaptive runner or evidence exists. The current installed default still abstains,
but it now evaluates three candidates and records `maturity_disallowed` rather than
returning zero candidates. See
[`adaptive_baseline_bundle_registry_2026-08-26.md`](adaptive_baseline_bundle_registry_2026-08-26.md).

## Result

At the time of this verification, the source, sdist, wheel, installed Python surface, and installed MCP discovery
agree on the Experimental adaptive-routing boundary. The packaged registry,
public evidence stream, and runner maps are empty. The installed recommendation
therefore abstains, and processing rejects the abstained decision with
`selection_required` without creating output.

This is interface and packaging verification. It does not qualify a real bundle,
measure model quality or speed, or demonstrate a selected public end-to-end run.

## Recorded outcome matrix

| Layer | Case | Exact expected outcome | Verification |
|---|---|---|---|
| Public installed default | Empty packaged registry | `status=abstained`, zero candidates | `tests.test_installed_ai_surface`, `tests.test_adaptive_routing_e2e` |
| Public workspace catalog | Valid caller catalog | `status=abstained`; `lifecycle_untrusted`, `registry_untrusted` | `tests.test_adaptive_routing_e2e` |
| Public processing | Abstained SelectionDecision | `ok=false`; `selection_required`; no output directory | `tests.test_installed_ai_surface`, `tests.test_adaptive_routing_e2e` |
| Public evidence input | Corrupt activation stream | `ok=false`; `invalid_evidence` | `tests.test_adaptive_routing_e2e` |
| Pure selector fixture | Complete `test_only=true` bundle | `status=abstained`; `test_only` | `tests.test_adaptive_routing_e2e` |
| Pure selector fixture | Complete qualified in-memory bundle | `status=selected`; empty reason list | `tests.test_adaptive_selector` |
| Environment profiles | CPU, Apple, NVIDIA, failed/unknown probes | Valid profile with exact `present`, `absent`, `unsupported`, or `failed` probe status | `tests.test_adaptive_environment_profile` |
| Evidence projection | Absent/inactive, revoked, superseded, expired, future, conflict, untrusted, not-qualified | Exact `evidence_*` reason matching the state | `tests.test_adaptive_selector`, `tests.test_adaptive_evidence_contracts` |
| Bundle and job gates | Composite artifacts, vocabulary, execution mode, provider, license, network | Selection or exact gate reason; network-required candidates yield `network_required` | Bundle, image-interface, selection, and selector suites |
| Executor-only fixture | Pinned fake runner below the public routing gate | Exact probe/load/predict/close sequence and bounded managed output | `tests.test_adaptive_processing` |
| Input and artifact safety | Swap, symlink, archive, byte/pixel/decode overflow | Fail before load or publication; no partial output | Inventory, qualification, processing, and managed-output suites |
| Output and force safety | Invalid result/mask/path/size/count or unrecognized destination | Fail closed; preserve an exact prior managed tree | `tests.test_adaptive_processing`, `tests.test_adaptive_managed_output` |
| Packaging | Git archive to sdist to wheel | Adaptive code, schemas, empty SSOT, and generated MCP reference present | `tests.test_candidate_artifact_ai_surface` |
| Installed MCP | Discovery and calls outside checkout with `PYTHONPATH` cleared | 27 live tools; recommendation abstains; processing returns `selection_required` | `tests.test_installed_ai_surface`, `tests.test_mcp_live_surface` |

Positive selector and executor fixtures are deliberately below the public
orchestration gate. No environment variable, MCP argument, registry field, or
packaged API enables their injection.

## Recorded verification

The following checks passed on 2026-08-26 in Asia/Tokyo:

- strict declarative manifest validation and generated MCP reference check;
- 7 packaged-manifest and documentation-reference tests;
- 158 adaptive tests, including the exact public/fixture outcome matrix;
- 23 candidate-artifact, installed-surface, AI-first, and live-MCP tests in a
  dependency-complete isolated environment;
- generated web-doc and adaptive-roadmap drift checks; and
- the full repository pre-push gate: lint, 153 focused tests, offline smoke, and
  real-image scenario preflight; and
- shell syntax and whitespace checks.

The installed and live-MCP tests emitted a dependency warning from
`pydantic-settings` about a third-party forward reference. The tests completed
successfully and no YOLOZU failure was observed.

## Verification commands

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 tools/generate_integration_tool_reference.py --check
python3 -m unittest tests.test_packaged_tools_manifest tests.test_manifest_docs_references
python3 -m unittest discover -s tests -p 'test_adaptive*.py'
python3 -m unittest tests.test_ai_first_mcp_surface tests.test_mcp_live_surface
/tmp/<dependency-venv>/bin/python -m unittest \
  tests.test_candidate_artifact_ai_surface \
  tests.test_installed_ai_surface \
  tests.test_ai_first_mcp_surface \
  tests.test_mcp_live_surface
YOLOZU_REQUIRE_REAL_COCO=1 python3 tools/generate_web_docs.py --check --json
python3 tools/generate_adaptive_vision_roadmap.py --check --json
bash scripts/pre_push.sh
git diff --check
```

The installed-artifact tests require the build backend and MCP dependencies.
They run in the hash-locked CI environment; missing optional local dependencies
produce an explicit skip rather than a fabricated pass.
