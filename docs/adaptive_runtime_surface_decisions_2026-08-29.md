# Third-party PyTorch runtime-surface decisions — 2026-08-29

Collected at 2026-08-28T23:40:26Z (UTC), corresponding to 2026-08-29 in
Asia/Tokyo. This decision reuses the immutable candidates and official primary
sources recorded in
[`adaptive_candidate_screenings_2026-08-29.md`](adaptive_candidate_screenings_2026-08-29.md).
No candidate code, package, model, weight, or dataset was downloaded or executed.
At 2026-08-29T00:12:19Z, the existing repository `.venv/bin/python`
environment was probed without mutation: Python 3.14.6, PyTorch
2.12.0.dev20260330, MPS built but unavailable, CUDA unavailable, and no
importable `groundingdino`, `sam2`, or `rfdetr` package.

The decision for both proposed runtime surfaces is `hold`. A hold is a completed
governance decision, not approval to add a dependency or an inference path. The
existing adapter and demanded-host qualification branch gates remain deferred.

## Grounding DINO plus SAM 2.1

Decision: `hold` for a shipped `groundingdino` + `sam2` + PyTorch provider.

| Required evidence | Current result |
|---|---|
| Legitimate product demand | `unknown`. The current public repository search found no consented workload that names this composite or fixes a demanded host. |
| Code and runtime licenses | Grounding DINO and SAM 2 code are Apache-2.0. SAM 2 states that its checkpoints are Apache-2.0. The inspected Grounding DINO material does not state a separate license for the fixed checkpoint, and the complete transitive runtime-license set has not been reviewed, so the composite license boundary remains `unknown`. |
| Immutable dependency and artifact integrity | The two source commits are fixed. Exact source-archive SHA-256 values, both complete weight SHA-256 values, and a complete transitive dependency lock and closure are unavailable in the reviewed record. |
| Real local inference | `unavailable`. The inspected local environment has PyTorch but no `groundingdino` or `sam2` package and no available accelerator provider. More importantly, no approved demanded host, complete artifact set, or supported isolated runner exists. No fixture or published benchmark substitutes for a real run. |
| Maintainable CI and release coverage | `unavailable`. The compiled PyTorch/CUDA paths are not shipped, installed-artifact-tested, or included in the release matrix. |
| Resource bounds | `unknown`. YOLOZU has no measured whole-job latency, peak memory, operation, or output-bound evidence for this exact composite. |
| Security review | `hold`. Compiled extensions, dependency resolution, artifact loading, and sandbox execution have not passed the repository isolation and supply-chain gates. |
| Support and maintenance policy | `unknown`. No repository owner or support policy accepts this fixed two-repository runtime surface. |

Reconsider only when one consented workload fixes the required task and host,
all source and weight bytes have reviewed licenses and SHA-256 values, the
isolation decision supports the exact backend, and one bounded real inference
plus maintainable CI/release coverage exists.

## RF-DETR Nano 1.9.4

Decision: `hold` for a shipped `rfdetr` + PyTorch provider.

| Required evidence | Current result |
|---|---|
| Legitimate product demand | `unknown`. The current public repository search found no consented low-latency workload or demanded host. |
| Code and runtime licenses | The fixed open-source package and Nano weight are identified as Apache-2.0; all Plus/XL/2XL components remain excluded. The transitive runtime/license set has not been reviewed as a shippable YOLOZU surface. |
| Immutable dependency and artifact integrity | Release 1.9.4 and its source commit are fixed. The official registry supplies an MD5 for the Nano weight, not the required SHA-256. Source-archive and weight SHA-256 values and a complete transitive dependency lock and closure remain unavailable because acquisition was not authorized. |
| Real local inference | `unavailable`. The inspected local environment has PyTorch but no `rfdetr` package and no available accelerator provider. More importantly, no approved demanded host, complete artifact set, or supported isolated runner exists. Published TensorRT/T4 figures are not local evidence. |
| Maintainable CI and release coverage | `unavailable`. YOLOZU has separate optional PyTorch and torchvision coverage, but the exact `rfdetr` provider and its fully resolved dependency and export surface are not qualified by the installed-artifact and release matrices. |
| Resource bounds | `unknown`. YOLOZU has no measured whole-job latency, sustained throughput, peak memory, or output-bound evidence for the fixed Nano variant. |
| Security review | `hold`. Transitive dependencies, artifact deserialization, optional export paths, and isolated execution have not passed the repository supply-chain gates. |
| Support and maintenance policy | The [official 1.9.4 release](https://github.com/roboflow/rf-detr/releases/tag/1.9.4) was published on 2026-08-24. This is release activity, not a YOLOZU maintenance commitment; YOLOZU has no repository owner or support policy for this provider surface. |

Reconsider only when a consented workload fixes the host and SLO, the exact
source and weight bytes have reviewed SHA-256 values, the full dependency/license
set and isolation backend pass review, and one bounded real run plus release CI
coverage is available.

## Product boundary

These holds do not reopen `YOLOZU-ll2.13`. They add no runtime dependency,
adapter, registry binding, model availability, qualification, selection,
promotion, or support claim. Unavailable evidence stays `unknown` rather than
being inferred from repository activity, fixtures, or upstream benchmark tables.
