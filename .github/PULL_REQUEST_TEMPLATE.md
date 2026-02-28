## Summary

- RFC issue (required for schema/protocol changes): <!-- e.g. YOLOZU-xxx / gh-123 -->

## Testing

- [ ] `python -m unittest`
- [ ] `ruff check .` (if available)
- [ ] `python tools/check_golden_compatibility.py`

## Contract / Compatibility Checklist

- [ ] If this PR changes schema/protocol behavior, linked RFC is approved.
- [ ] Golden assets under `baselines/golden/` were updated (or confirmed unchanged).
- [ ] Compatibility gates pass in CI.
- [ ] If reference adapter baselines changed, explain interface contract / canonicalization / metric / model/runtime deltas.
- [ ] If reference adapter baselines changed, include old-vs-new regression summary in PR description.
- [ ] If reference adapter baselines changed, explicitly state `dataset_hash` / `weights_hash` change status.

## Notes
