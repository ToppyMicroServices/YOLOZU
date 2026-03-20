## Summary

- RFC issue (required for schema/protocol changes): <!-- e.g. YOLOZU-xxx / gh-123 -->

## Testing

- [ ] `python -m unittest`
- [ ] `ruff check .` (if available)
- [ ] `python tools/check_golden_compatibility.py`
- [ ] Added or updated automated tests for major new functionality / behavior changes, or explained why tests are not practical
- [ ] If this PR changes release automation, versioning, or packaging behavior, added or updated targeted regression tests

## Contract / Compatibility Checklist

- [ ] If this PR changes schema/protocol behavior, linked RFC is approved.
- [ ] Golden assets under `baselines/golden/` were updated (or confirmed unchanged).
- [ ] Compatibility gates pass in CI.
- [ ] Docs were updated when behavior, CLI, or operator workflow changed.
- [ ] Manifest/manual were updated when machine-readable tool docs or published operator guidance changed.
- [ ] If no docs/manual/manifest updates were needed, the PR description explains why.
- [ ] If reference adapter baselines changed, explain interface contract / canonicalization / metric / model/runtime deltas.
- [ ] If reference adapter baselines changed, include old-vs-new regression summary in PR description.
- [ ] If reference adapter baselines changed, explicitly state `dataset_hash` / `weights_hash` change status.

## Notes
