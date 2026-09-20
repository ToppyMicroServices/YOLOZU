# First image-service CNN: review status

Status on 2026-09-20: **hold; not registered, qualified, activated, or promoted**.
The bounded MCP gateway works, but its default model selection still abstains.
This is the remaining work in step 1 of the service plan, tracked as `YOLOZU-0rp4`.

## Exact local candidate

The [machine-readable review proposal](../reports/image_service_candidate_review_2026-09-20.json)
contains an `AlgorithmBundleSpec` that passed the strict schema/digest validator.
It is outside the managed registry and has no effect on selection. Its proposed
code-owned execution class is subject to repository review, not an audit claim.

The candidate is Torchvision Mask R-CNN ResNet-50 FPN v2, `COCO_V1`, CPU FP32,
object detection only, with the exact 80-class mapping. The report pins Torch
`2.12.0.dev20260330`, Torchvision `0.27.0.dev20260330`, Pillow `12.2.0`,
safetensors `0.8.0`, adapter bytes, and all three component digests. This installed
nightly runtime was used for a local smoke; no other runtime is qualified by it.

The cached source checkpoint was rehashed before a restricted `weights_only=True`
load. Conversion sorted the state keys and serialized contiguous tensors with
`safetensors.torch.save`. The converted SHA-256 is
`59f39b25a05a7130ebb754f82451e44b791c684601a15cefc64f79462e1e8187`
(185,728,220 bytes). No weights were downloaded or redistributed. In the draft,
`source_id` identifies the original checkpoint, not a downloadable safetensors
file; any later acquisition workflow must preserve that conversion distinction.

The revised adapter loaded those bytes and retained 10 detections on the existing
smoke image, matching the earlier run's first prediction. Load plus prediction
took 5.690 seconds in this single run. This is not a latency percentile, accuracy
measurement, supported resolution range, or full qualification. The draft input
shape describes only that image. No OS network isolation was measured or claimed.

## Why activation remains blocked

Torchvision's [code license](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE)
is BSD-3-Clause, but its [pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license)
asks users to determine the permissions for their use case, including terms
derived from training data. The inspected sources do not settle the permission
for this proposed service and weight redistribution. The draft therefore keeps
`license_expression=unknown` and `license_review=unreviewed`. This is an unresolved
review question, not a finding that use is prohibited. The upstream sources were
checked on 2026-09-20; the exact installed source-revision pages could not be
retrieved in that check.

Repository prerequisites still missing:

- A reviewed disposition for the exact weights, runtime dependency set, and
  evaluation images, with retained references and intended service/redistribution scope.
- Managed candidate screening and registration with reviewed license state.
- A frozen quality workload, evaluator, threshold, and measured qualification.
  The one-image smoke must not supply those values.
- Reviewed support profiles, retained evidence activation, and the existing
  failure-drill/promotion gates. No approval identifier was invented or reused.

The current qualifier also requires an enabled, license-approved channel target
before measurement. A new candidate's initial qualification entry path needs to
be resolved under the existing governance policy before its first promotion;
assigning a channel just to get past preflight would not establish eligibility.
The exact proposal's current bundle lookup was checked and returned
`bundle_not_found`; no full workload was started and no lifecycle record changed.

Once these inputs and reviews exist, run the frozen workload, retain its actual
results even on failure, and promote only if every gate passes. Neither this
proposal nor explicit `execute=true` can bypass those checks.

## Steps 2–4

The repository implements the five-tool surface, authenticated Streamable HTTP,
private tenant directories, image/capacity and HTTP-upload limits, per-minute request limits,
idle-time expiry, queued cancellation, and bounded runner execution. The
authenticated local HTTP-protocol test covers upload, submit, status, and cancel;
without a registered model it verifies abstention, not successful CNN selection.
Provider request settings were checked against the official OpenAI and Claude
documentation; no provider API call was made.

Public operation still needs a deployment target, TLS/DNS, a real credential,
gateway request limits, and OS/container isolation. None has been provisioned.
See [the service guide](image_service_mcp.md) for exact limits and setup.
