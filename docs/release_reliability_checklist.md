# Release reliability checklist

Use this checklist before publishing a GitHub Release. The goal is deterministic quality gates, explicit skip reasons, and reproducible artifacts.

Release trigger note:
- PyPI publish is triggered by `.github/workflows/publish.yml` on `release: published`.
- Tag push alone is insufficient for PyPI publish.
- Recommended operator helper: `bash release.sh`

## Auto bump policy

`tools/release.py` (wrapped by `release.sh`) classifies release size from git diff stats (`files changed`, `insertions+deletions`) since the latest semver tag.

- small: `X.Y.Z -> X.Y.(Z+1)` (`1.1.1+add` equivalent)
- medium: `X.Y.Z -> X.(Y+1).0` (`1.1+a.0` equivalent)
- large: `X.Y.Z -> (X+1).0.0` (`1+a.0.0` equivalent)

Dry-run preview:

```bash
bash release.sh --dry-run --allow-dirty --allow-non-main --output reports/release_report.dry_run.json
python3 tools/announce_release.py --event-json "$GITHUB_EVENT_PATH" --out-dir reports/announce_preview --x-max-len 280
```

## 1) Required local checks (must pass)

Run from repo root:

```bash
bash scripts/smoke.sh
# optional deeper walkthrough evidence (capability claims + deploy dry-run checks)
bash scripts/smoke.sh --profile deep
# CUDA machine: run TTT probe on GPU
bash scripts/smoke.sh --profile deep --torch-device cuda
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
python3 tools/check_schema_compatibility.py
python3 tools/check_golden_compatibility.py
python3 -m unittest tests.test_manifest_docs_references tests.test_tool_manifest tests.test_packaged_tools_manifest
python3 -m unittest tests.test_backend_shape_format_contracts tests.test_external_inference_templates_smoke tests.test_summarize_gpu_ngc_run_tool
python3 tools/check_mcp_settings.py --output reports/mcp_settings_check.release.json
```

DoD:
- `scripts/smoke.sh` writes `reports/smoke_coco_eval_dry_run.json`, SynthGen smoke artifacts (`reports/smoke_synthgen_summary.json`, `reports/smoke_synthgen_eval.json`, `reports/smoke_synthgen_overlay.png`), and instance-seg demo overlay PNGs under `reports/smoke_demo_instance_seg/overlays/` (unless `--skip-demo` is used).
- `scripts/smoke.sh --profile deep` additionally writes `reports/smoke_walkthrough_report.json` and backend dry-run exports (`reports/smoke_export_{onnxrt,trt,executorch}.json`). The deep profile uses `--torch-device` for the TTT probe device selection.
- Manifest validator returns `OK`.
- Schema compatibility gate passes.
- Golden compatibility check returns `ok=true`.
- Unit tests pass without unexpected failures.
- MCP settings check report shows `ok=true`.

## 2) Required CI workflows

- `.github/workflows/build_and_test.yml` (**required**): must be green on target commit.
- `.github/workflows/codeql.yml` (**required for security posture**): must be green on target commit or explicitly skipped for unsupported languages.
- `.github/workflows/manual_doi.yml` (**required when shipping manual update**): publishes `manual/build/yolozu_manual.pdf` to a separate Zenodo record and links it to software concept DOI.
- `.github/workflows/container.yml` (**optional publish**): expected to run for container-related changes on `main`; publishes only on tag/manual.
- `.github/workflows/announce_release.yml` (**optional announce**): posts GitHub Release announcement to LinkedIn/X/Reddit when secrets are configured; always uploads a post bundle artifact.
- `.github/workflows/ngc_test.yml` (**optional GPU smoke**): must produce deterministic `pass` or `skip` summary in `ci-logs/gpu-ngc`.
- `.github/workflows/gpu_zisn_pipeline.yml` (**optional GPU validation split**): manual machine-runner path for `YOLOZU-zisn.1/.2/.3` artifacts.
- Optional log-branch publish secret: `CI_LOGS_PUSH_TOKEN` (fine-grained PAT with repository contents write) for `ci-logs/*` branch updates from GPU workflows.

DoD:
- `ci` completed successfully.
- `container` failures are triaged only if release depends on image artifacts.
- `ci` includes schema compatibility, golden compatibility, and sdist/wheel package-content gates.
- GitHub Actions references are SHA-pinned and Python workflow installs use `tools/ci/install_with_hashes.py` so release automation is supply-chain hardened.
- `main` branch protection requires PR review (1 approval), includes administrators, dismisses stale reviews, requires conversation resolution, and blocks force-pushes.
- Manual DOI workflow produces `reports/manual_doi_publish.json` and a published (or explicit draft) Zenodo record.
- `gpu-ngc` produces `ci_logs/ci_gpu_ngc/dod_summary.json` and `dod_summary.md`.
- `gpu-zisn-pipeline` (when executed) produces `ci_logs/ci_gpu_zisn/dod_summary.json` and stage artifacts under `ci_logs/ci_gpu_zisn/zisn1|zisn2|zisn3/`.
- When `CI_LOGS_PUSH_TOKEN` is not configured, the workflows still upload artifacts but skip force-pushing the `ci-logs/*` branches.

## 3) GPU smoke interpretation (`gpu-ngc`)

Expected statuses:
- `pass`: GPU smoke executed and produced TRT parity/latency artifacts.
- `skip`: acceptable only with explicit reason (e.g., no idle runner or probe 403).
- `fail`: release blocker for GPU-related deliverables.

Known skip conditions:
- No idle self-hosted GPU runner (`has_runner=false`).
- Runner discovery API denied (`probe_status=http_403`): set `RUNNER_DISCOVERY_TOKEN`.
- GPU job skipped after runner detection (e.g., `NGC_API_KEY` not set).

DoD:
- Every run has a clear `dod_status` and guidance text in summary artifacts.
- No ambiguous “missing artifacts” state when `gpu_job_result=success`.

## 4) Contract and artifact sanity

- Validate representative predictions:
  ```bash
  python3 tools/validate_predictions.py data/smoke/predictions/predictions_dummy.json --strict
  ```
- Confirm docs/manifest sync:
  ```bash
  python3 -m unittest tests.test_manifest_docs_references
  ```
- Confirm backend contract guardrails:
  ```bash
  python3 -m unittest tests.test_backend_shape_format_contracts
  ```

DoD:
- Predictions schema validation passes in strict mode.
- Manifest docs references are synchronized and resolvable.
- Shape/format mismatch guardrails are covered by tests.

## 5) Release decision rule

Release is ready when all are true:
- Required CI (`ci`) is green.
- Local required checks pass.
- GPU workflow has either `pass` or justified `skip` with recorded reason.
- No unresolved `P0/P1` reliability issues in `bd list`.
