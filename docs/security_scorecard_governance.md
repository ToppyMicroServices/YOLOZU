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

## 2026-03-28 remote audit snapshot

Latest remote code-scanning triage on `main` shows:

- actionable CodeQL items:
  - `py/clear-text-storage-sensitive-data` in `tools/check_repo_governance.py`
  - `py/empty-except` in `tools/export_predictions_trt.py`
- repository-history / external-program items that still require operational follow-through:
  - `PinnedDependenciesID` for `.clusterfuzzlite/Dockerfile`
  - `VulnerabilitiesID` (`GHSA-hqmj-h5c6-369m`)
  - `CodeReviewID`
  - `MaintainedID`
  - `SASTID`

The first two items are source-level and should be fixed in the repository.
The remaining items are primarily governed by dependency upgrades, reviewed-PR history,
or external program status rather than by a single source edit.
