# Portable release qualification packs

YOLOZU can turn one candidate predictions artifact into a portable release
decision. The same engine is available from the CLI, `yolozu.api`, and MCP.

Install the COCO extra before asking for a passing quality decision:

```bash
python3 -m pip install 'yolozu[coco]'
yolozu qualify-release create \
  --dataset /absolute/path/to/dataset \
  --predictions /absolute/path/to/candidate_predictions.json \
  --output-dir reports/candidate_qualification \
  --split val \
  --min-map50-95 0.40
```

The command exits `0` only for `decision: pass`. It exits `3` for `hold` and
`4` for a measured gate failure. A dry run validates the path and emits a pack,
but cannot pass because it has no official metrics:

```bash
yolozu qualify-release create \
  --dataset data/smoke \
  --predictions data/smoke/predictions/predictions_dummy.json \
  --output-dir reports/qualification_smoke \
  --split val \
  --max-images 2 \
  --dry-run
```

At least one quality threshold is required for `pass`. Use a baseline to gate
regression:

```bash
yolozu qualify-release create \
  --dataset /absolute/path/to/dataset \
  --predictions /absolute/path/to/candidate_predictions.json \
  --baseline-predictions /absolute/path/to/baseline_predictions.json \
  --output-dir reports/candidate_qualification \
  --max-map50-95-drop 0.01
```

## Pack contents

A pack contains only JSON evidence:

- `request.json`: normalized options, labels, input content digests, and the
  evaluation protocol fingerprint
- `candidate_evaluation.json`: path-redacted evaluation output
- `baseline_evaluation.json`: optional path-redacted baseline output
- `qualification.json`: `pass`, `hold`, or `fail` plus every gate result
- `checksums.json`: SHA-256 values for every evidence file and one pack digest

Dataset bytes, predictions, and model weights are not copied into the pack.
Input paths are reduced to labels before persistence. The dataset and
predictions are content-fingerprinted, so a later run can identify changed
inputs without disclosing their original absolute paths.

Verify checksums and recompute the embedded decision semantics before use:

```bash
yolozu qualify-release verify reports/candidate_qualification
```

Compare two packs only when their dataset and evaluation protocol fingerprints
match:

```bash
yolozu qualify-release diff \
  reports/baseline_qualification \
  reports/candidate_qualification
```

Pack checksums detect accidental modification and internal inconsistency. They
do not authenticate the publisher. Sign or externally pin the pack digest when
publisher identity matters.

## Stable Python API

```python
from yolozu.api import qualify_release, verify_qualification_pack

result = qualify_release(
    "/absolute/path/to/dataset",
    "/absolute/path/to/predictions.json",
    "/absolute/path/to/qualification_pack",
    min_map50_95=0.40,
)
assert result.passed
assert verify_qualification_pack(result.pack_dir).ok
```

The public types are `QualificationResult`, `PackVerificationResult`, and
`QualificationError`. `diff_qualification_packs` provides the same compatible
pack comparison used by the CLI.

## MCP

The live `qualify_release` MCP tool uses the same engine. Its input and output
paths must stay inside the MCP workspace. It defaults to `dry_run=true`, so an
agent must explicitly request real metrics and provide a threshold before a
pack can pass. The tool is Stable but is not in `guaranteed_ai_safe`: real
evaluation depends on local data and the optional COCO runtime, and it writes
the requested pack directory.
