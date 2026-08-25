# Adaptive baseline bundle registry

Recorded: 2026-08-26 (Asia/Tokyo)

Bead: `YOLOZU-ll2.81.2.1`

## Result

The three live records in `yolozu/data/manifest/model_zoo.json` are registered as
non-promoted Candidate `AlgorithmBundleSpec` records. The registry does not claim
that they can run through adaptive routing. Each record has
`execution_binding.status=unbound` and
`execution_binding.reason_code=runner_artifact_set_incomplete` because the model
zoo supplies a fetchable weight asset, not the complete ordered artifact set and
audited adapter needed by an adaptive runner.

The lifecycle contains one global registration and one Candidate registration per
bundle. It contains no Experimental or Stable assignment, support profile, or
qualification evidence. The installed recommendation therefore evaluates all
three records and abstains with `maturity_disallowed` before opening any artifact.

## Exact registered records

| Bundle | Source revision | Expected bytes | SHA-256 | License metadata | Execution | Lifecycle |
|---|---:|---:|---|---|---|---|
| `yolox-s-coco` | `0.1.1rc0` | 72,089,125 | `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` | `Apache-2.0`; pinned model-zoo review at `eba6c8d511033606eb2798dfa9f12747296db1fc` | unbound; fetchable model asset only | Candidate |
| `detectron2-faster-rcnn-r50-fpn-1x-coco` | `137257794` | 167,266,879 | `b6def88fe428c339d60e718c0e9b4d821580df6a92e02435ef439fea8a3a2395` | `Apache-2.0`; pinned model-zoo review at `eba6c8d511033606eb2798dfa9f12747296db1fc` | unbound; fetchable model asset only | Candidate |
| `mmdet-faster-rcnn-r50-fpn-1x-coco` | `20200130-047c8118` | 167,287,506 | `047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e` | `Apache-2.0`; pinned model-zoo review at `eba6c8d511033606eb2798dfa9f12747296db1fc` | unbound; fetchable model asset only | Candidate |

The source identifiers remain the existing official GitHub release or official
download URLs in the model-zoo SSOT. The bundle records reuse those exact source
identifiers, revisions, sizes, hashes, licenses, and deterministic cache keys. They
also inline the fixed COCO 80-class vocabulary. Unknown runtime compatibility,
memory, quality, and performance are not filled from model cards or inference.

## State boundaries

| State | Current meaning | Current result |
|---|---|---|
| Registered | A validated immutable record and append-only Candidate lifecycle entry exist. | Yes, for exactly three model-zoo records. |
| Fetchable | The existing explicit model-fetch path has an official source, pinned size and SHA-256, license gate, and cache identity. Fetch remains a separate user-authorized operation. | Metadata complete; no download was performed for this registration. |
| Executable | A complete runner-consumed artifact set, validated runtime/pipeline fields, and an audited adapter route are bound. | No. All three records are `unbound`; `runner_unavailable` is fail-closed. |
| Qualified | Active evidence matches the exact bundle, artifacts, environment, workload, protocol, and advertised constraints. | No. No qualification evidence or support profile exists. |
| Recommended | An Experimental or Stable assignment survives every gate and ranks first. | No. Candidate is nonselectable; the installed default abstains. |

These states are independent. In particular, registered or fetchable does not imply
executable, qualified, recommended, supported, or adopted.

## Deterministic exclusions

Focused fixtures assert these exact fail-closed outcomes:

- unavailable runtime: `runtime_unavailable`;
- missing artifact inventory: `artifact_member_missing`;
- unknown license review: `license_not_approved`;
- absent active evidence: `evidence_inactive`; and
- unbound adaptive runner: `runner_unavailable`.

The model-fetch tests also reject a size mismatch before publishing a cache entry or
metadata file. These checks are interface and security evidence only. They do not
measure inference, quality, latency, FPS, hardware support, or human adoption.

## SSOT and packaging

- Model acquisition metadata: `yolozu/data/manifest/model_zoo.json`
- Bundle registry: `yolozu/data/adaptive_routing/bundle_specs.json`
- Append-only lifecycle: `yolozu/data/adaptive_routing/bundle_lifecycle.jsonl`
- Bundle schema: `docs/schemas/algorithm_bundle_spec.schema.json`
- Packaged schema copy: `yolozu/data/schemas/algorithm_bundle_spec.schema.json`

The registry and lifecycle files already live in the packaged data tree, so there
is no second generated data copy. Source and packaged schema copies are required to
remain byte-identical.
