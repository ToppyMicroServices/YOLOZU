# Scorecard Governance Status

This page separates repository-local security hardening from Scorecard items that depend on GitHub settings, review history, or external programs.

For a local, snapshot-backed audit flow, see [`docs/repo_governance_audit.md`](repo_governance_audit.md).

## Already handled in the repository

- Workflow permissions are explicitly minimized in GitHub Actions.
- GitHub Actions references are SHA-pinned.
- Python install paths used by CI/container bootstrap are hash-locked via `tools/ci/install_with_hashes.py`.
- TensorRT/NGC container base images are pinned by digest.
- Code scanning uses a repo-managed `codeql.yml`.
- Fuzzing coverage is present through ClusterFuzzLite:
  - `.github/workflows/cflite_pr.yml`
  - `.github/workflows/cflite_batch.yml`
  - `.clusterfuzzlite/`
  - `fuzz/predictions_canonicalize_fuzzer.py`

## Repository settings currently expected

These items are configured outside the codebase and should be reviewed in GitHub settings when Scorecard findings are triaged.

- Branch protection on `main`
  - require pull request reviews
  - require conversation resolution
  - require linear history
  - disallow force pushes
  - include administrators
- Security scanning enabled
  - CodeQL workflow enabled
  - Scorecard workflow enabled
- Dependabot enabled
  - version updates
  - security updates

These settings can be checked locally by exporting GitHub snapshots and running:

```bash
python3 tools/check_repo_governance.py \
  --repo-json reports/github_governance/repo.json \
  --branch-protection-json reports/github_governance/branch_protection_main.json \
  --output reports/repo_governance_check.json
```

## Residual Scorecard items that are not solved by one commit

### `CodeReviewID`

This depends on reviewed pull request history, not just branch protection. Even with required review enabled, Scorecard can continue to report this until the repository accumulates enough reviewed PR activity.

Recommended operational rule:

- land changes through pull requests only
- keep `required_approving_review_count >= 1`
- avoid direct pushes to protected branches

### `MaintainedID`

This is partly time/history based. Active commits, issue triage, and release activity help, but there is no single repository file that clears it immediately.

Recommended operational rule:

- keep releases regular
- close or update stale P1/P2 issues
- keep CI green on default branch

### `CIIBestPracticesID`

This requires external OpenSSF Best Practices badge enrollment and completion. The repository can prepare evidence, but the finding will remain until the badge program state changes externally.

Recommended next step:

- enroll the repository in the OpenSSF Best Practices program
- map checklist evidence from `docs/release_reliability_checklist.md`
- keep the repository governance snapshot audit current in `docs/repo_governance_audit.md`

### `SASTID`

This depends on GitHub recognizing static-analysis coverage for the relevant
commits. The source-level control is the pinned CodeQL workflow; after that, the
signal may lag until CodeQL completes on the default branch and GitHub refreshes
code-scanning state.

Recommended operational rule:

- keep `.github/workflows/codeql.yml` enabled on `main` and on a schedule
- require the default `ci` workflow before merge
- verify the latest CodeQL run after security-sensitive workflow changes

### `CITestsID`

This depends on repository history and whether merged changes are associated
with CI-tested commits. The repository's default `ci` workflow now runs on pull
requests and `main`; the remaining risk is operational, such as admin merges or
manual pushes that bypass the normal review path.

Recommended operational rule:

- land changes through pull requests with passing `ci`
- avoid direct pushes to `main`
- keep workflow-only changes covered by the release/security regression fast path

### `VulnerabilitiesID`

This tracks dependency vulnerabilities and reachable package exposure. Some findings can be reduced in-code by raising minimum versions or locking installs, but others require dependency updates over time.

Recommended operational rule:

- keep Dependabot PRs flowing
- treat runtime package CVEs ahead of dev-only/test-only updates
- re-run release gates before tagging

## Practical interpretation

Use Scorecard as a posture dashboard:

- repository-local findings should usually be fixed in code/workflows
- governance/history findings should be documented, monitored, and handled through repository settings and maintainer process

This avoids spending time chasing findings that cannot be cleared by editing source files alone.

## 2026-04-02 remote audit snapshot

Latest remote code-scanning triage on `main` shows:

- source-level / workflow-level items that can be improved in-repo:
  - `VulnerabilitiesID` reported four `onnx` advisories on the default-branch dependency graph:
    - `GHSA-3r9x-f23j-gc73`
    - `GHSA-538c-55jv-c5g9`
    - `GHSA-cmw6-hcpp-c6jp`
    - `GHSA-p433-9wv8-28xj`
  - at the time of that run, the Scorecard workflow was still pinned to `ossf/scorecard-action` `v2.4.1` (since updated to `v2.4.3`; see follow-up below)
- repository-history / external-program items that still require operational follow-through:
  - `CodeReviewID`
  - `MaintainedID`
  - `SASTID`

The Scorecard action pin is updated in-repo, and the repository-side `onnx` minimum/lock
versions are raised to `1.21.0` so the next default-branch run can clear the patched
dependency-side vulnerability signal. The remaining three findings are still primarily
governed by reviewed-PR history, repository age/activity, and GitHub-side scan coverage
timing rather than by a single source edit.

## 2026-04-02 follow-up

The repository now carries these explicit source-level mitigations for Scorecard-related
security findings:

- `.clusterfuzzlite/Dockerfile` installs from a hash-locked requirements file with `pip --require-hashes`
- `osv-scanner.toml` documents a scoped ignore for `GHSA-hqmj-h5c6-369m`
- `.github/workflows/scorecard.yml` is pinned to `ossf/scorecard-action` `v2.4.3`
- `.github/workflows/build_and_test.yml` runs a workflow-only release/security regression fast path on pull requests, while `.github/workflows/scorecard.yml` uploads default-branch posture to code scanning after merge
- `pyproject.toml`, `requirements-test.txt`, and repository lockfiles pin `onnx>=1.21.0` / `onnx==1.21.0` to pick up fixes for:
  - `GHSA-3r9x-f23j-gc73`
  - `GHSA-538c-55jv-c5g9`
  - `GHSA-cmw6-hcpp-c6jp`
  - `GHSA-p433-9wv8-28xj`

Additional CI/CD hardening now enforced in-repo:

- workflow-only edits are no longer a blind spot: `.github/workflows/build_and_test.yml` runs release/security regression tests in `workflows_meta`
- `.github/workflows/container.yml` builds only on release tags or manual dispatch; container dependency/bootstrap validation should be manually triggered before releases that depend on image artifacts
- `.github/workflows/publish.yml` now fails fast when package version, release/manual trigger inputs, and `CHANGELOG.md` are not aligned

The `GHSA-hqmj-h5c6-369m` advisory currently has no fixed upstream `onnx` release. This
repository does not call `onnx.hub.load()`, which is the affected API surface, so the ignore
is limited to that advisory and should be revisited as soon as a patched `onnx` version exists.

## 2026-06-21 follow-up

The default-branch Scorecard alert set reported two source-actionable items:

- `VulnerabilitiesID` listed seven PyTorch advisories. YOLOZU keeps PyTorch out
  of the base runtime dependencies, but optional train/demo/test extras now
  require `torch>=2.10.0` and `torchvision>=0.25.0` so fixed PyTorch advisories
  do not remain reachable through broad lower bounds. Two PyTorch advisories
  still had no fixed upstream version in OSV on 2026-06-21:
  `GHSA-rrmf-rvhw-rf47` and `PYSEC-2026-139`. They are documented in
  `osv-scanner.toml` as temporary ignores because the dependency is optional and
  no patched torch release is available yet.
- `SASTID` reported that CodeQL was not visible across recent commits even
  though `.github/workflows/codeql.yml` was running successfully. The Scorecard
  job now grants `actions: read` and `contents: read` so the SAST check can read
  workflow/run metadata while still keeping the job permissions narrow.

## 2026-04-02 residual alert disposition

After the repository-local fixes above landed on `main`, the remaining open
code-scanning alerts were:

- `CodeReviewID`
- `MaintainedID`

These are not source vulnerabilities in this repository. They are residual
Scorecard governance signals driven by review history and repository age:

- `CodeReviewID` remains sensitive to recent reviewed history, not just the
  current branch-protection settings. This repository now requires pull-request
  review on `main`, but historical admin merges and bot-only change windows can
  still surface a `0/24 approved changesets` score until more reviewed PRs
  accumulate.
- `MaintainedID` remains sensitive to repository age. Scorecard explicitly keeps
  this signal red during the first 90 days, which cannot be changed by a source
  patch before that age window expires.

Operational handling:

- keep `required_approving_review_count >= 1`
- keep `require_last_push_approval=true`
- prefer reviewed PR merges over direct/admin merges whenever possible
- revisit `MaintainedID` automatically after the repository is older than 90 days

Because these alerts are operationally acknowledged and already documented here,
they can be dismissed in GitHub code scanning with a `won't fix` rationale that
points back to this governance note.
