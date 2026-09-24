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

## Approved local preparation boundary

On 2026-09-25, the release reviewer approved a narrower acquisition design. YOLOZU
does not download or redistribute the source checkpoint or converted artifact.
The user obtains the exact upstream checkpoint, reviews the upstream pretrained-model
notice, and runs:

```bash
yolozu prepare-torchvision-maskrcnn \
  --checkpoint /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --accept-upstream-terms
```

The command accepts only the pinned 185,828,065-byte source with SHA-256
`73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`.
It performs no network request, loads only after verification with
`weights_only=True` and read-only memory mapping, converts locally with sorted
contiguous tensors, and accepts only the 185,728,220-byte safetensors output with SHA-256
`59f39b25a05a7130ebb754f82451e44b791c684601a15cefc64f79462e1e8187`.
The adjacent provenance records the upstream URL and notice, runtime versions,
explicit acknowledgement, `NOASSERTION`, and false redistribution fields without
recording the user's source path. Existing cache bytes are never overwritten.

This resolves the acquisition and redistribution design question. It does not
establish that the checkpoint is Apache-2.0. It also does not register the draft,
approve a license review, define a quality workload, or satisfy qualification.

The implementation was exercised on 2026-09-25 with the cached source checkpoint.
The temporary conversion produced the exact expected size and SHA-256, and its
provenance recorded Torch `2.12.0.dev20260330` and safetensors `0.8.0`. The
temporary artifact was then removed. Installation into the default user cache was
attempted but failed cleanly because the host filesystem had less usable free space
than the 185,728,220-byte artifact. No partial artifact remains. Persistent local
installation is therefore still incomplete on this host; the conversion result is
not qualification evidence.

## Why activation remains blocked

Torchvision's [code license](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE)
is BSD-3-Clause, but its [pretrained-model notice](https://github.com/pytorch/vision#pre-trained-model-license)
asks users to determine the permissions for their use case, including terms
derived from training data. The local preparation record therefore uses
`NOASSERTION`; it does not copy Apache-2.0 from YOLOZU onto the checkpoint. The
historical 2026-09-20 draft remains unchanged with `license_expression=unknown`
and `license_review=unreviewed`. A future registered spec must use the reviewed
`NOASSERTION` disposition unless independent rights evidence supports a different
expression.

Repository prerequisites still missing:

- A managed artifact-license review that records `NOASSERTION`, the approved
  user-acquired/non-redistributed scope, and retained upstream references.
- Reviewed dispositions for the runtime dependency set and frozen evaluation data.
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
