# Repository Governance Audit

This page covers the local audit path for repository settings and governance posture that cannot be proven from source files alone.

Use it for:

- branch protection verification on `main`
- review-policy verification
- local evidence for fuzzing / Scorecard / Dependabot configuration
- collecting operator evidence before a release or security review

## What this does and does not prove

`tools/check_repo_governance.py` combines two kinds of evidence:

- local repository evidence
  - `codeql.yml`
  - `scorecard.yml`
  - `cflite_pr.yml`
  - `cflite_batch.yml`
  - `dependabot.yml`
  - governance/release docs
- exported GitHub settings snapshots
  - repository JSON
  - branch-protection JSON for `main`

It does **not** prove history-based or external-program findings by itself. In particular:

- `CodeReviewID` still depends on reviewed PR history
- `MaintainedID` still depends on release/activity cadence over time
- `CIIBestPracticesID` still depends on the OpenSSF Best Practices program state

## Snapshot collection

Collect the snapshots locally:

```bash
mkdir -p reports/github_governance
gh api repos/ToppyMicroServices/YOLOZU \
  > reports/github_governance/repo.json
gh api repos/ToppyMicroServices/YOLOZU/branches/main/protection \
  > reports/github_governance/branch_protection_main.json
```

Then run the audit:

```bash
python3 tools/check_repo_governance.py \
  --repo-json reports/github_governance/repo.json \
  --branch-protection-json reports/github_governance/branch_protection_main.json \
  --output reports/repo_governance_check.json
```

Typical output:

- `reports/repo_governance_check.json`
- stdout summary with required failures / missing evidence / manual followups

For a local-only pass/fail check of repository-side evidence, allow missing snapshots explicitly:

```bash
python3 tools/check_repo_governance.py \
  --repo-root . \
  --allow-missing-evidence \
  --output reports/repo_governance_check.local.json
```

## Current expected policy

The audit currently treats the following as required:

- default branch is `main`
- `main` requires at least one approving review
- stale reviews are dismissed
- approval is required after the last push
- conversation resolution is required
- linear history is enabled
- administrators are covered by protection
- force-pushes are disabled
- branch deletions are disabled
- Dependabot security updates are enabled

The audit currently treats the following as advisory:

- secret scanning enabled
- secret-scanning push protection enabled

## How to interpret failures

- `required_failed > 0`
  - a repository setting or local governance artifact is below the expected policy
- `missing_evidence > 0`
  - source files may be healthy, but the audit cannot verify live GitHub settings yet
- `manual_followups`
  - items that cannot be cleared by editing source files alone

## Why this exists

This repository keeps security hardening split into two layers:

- repository-local controls that can be versioned and tested
- repository settings / maintainer process that must be reviewed operationally

That split is important for security reviews, release preparation, and badge evidence because it makes the boundary explicit instead of hiding it in prose.
