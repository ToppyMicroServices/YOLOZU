# First image-service CNN: review status

Status on 2026-09-26: **measured hold; Candidate only**.

The exact Torchvision Mask R-CNN bundle is registered and executable only on the
qualification-only Candidate channel. Ordinary routing still cannot select it.
The first full qualification completed, but its preregistered latency gate did
not pass. No evidence activation, support-profile assignment, lifecycle
promotion, failure-drill claim, or successful five-tool CNN execution followed.
This is tracked as `YOLOZU-0rp4`.

## Artifact and redistribution boundary

YOLOZU source remains Apache-2.0. That license is not copied onto the source
checkpoint or the locally converted safetensors artifact. The bundle records
the weight license as `NOASSERTION` and retains the upstream pretrained-model
notice at the pinned Torchvision revision.

The user obtains the exact upstream checkpoint and runs:

```bash
yolozu prepare-torchvision-maskrcnn \
  --checkpoint /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --accept-upstream-terms
```

The command performs no download. It accepts only the 185,828,065-byte source
with SHA-256
`73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`,
then writes only the 185,728,220-byte local safetensors artifact with SHA-256
`59f39b25a05a7130ebb754f82451e44b791c684601a15cefc64f79462e1e8187`.
Both files remain outside the repository, packages, reports, and release
artifacts. The local artifact and path-free provenance were rechecked at mode
`0600` before qualification.

This design avoids YOLOZU redistribution. It is not a legal conclusion that the
checkpoint is Apache-2.0 or that every use is permitted.

## Frozen quality run

The reviewed bundle is
`torchvision-maskrcnn-r50-fpn-v2-coco-cpu@2026-09-26-local1`, with bundle-spec
digest `76ab14c7d5a2f50824491d6859b1ecfd2dd40cecfd0ac951f2d8ed1c23784f2d`.
The job definition is
[`torchvision_maskrcnn_coco16_cpu.json`](../configs/qualification/torchvision_maskrcnn_coco16_cpu.json).
It fixed the 16-image local COCO input, exact annotation hash, code-owned
`coco_bbox_simple_map_v1` evaluator, quality threshold `0.2`, p95 limit
`5000 ms`, cold-start limit `30000 ms`, and minimum repeat throughput
`0.1 fps` before the completed measurement.

The evaluator excludes `iscrowd=1`, binds every selected image basename and
dimension, binds the exact 80-class bundle vocabulary, and uses YOLOZU's simple
bbox mAP at IoU thresholds 0.50 through 0.95. It is not official COCOeval and
must not be compared directly with an upstream COCO metric.

The full protocol ran one fresh cold start, 20 warm-up iterations, three repeats
of 200 timed inferences, and one quality prediction for each of the 16 images.
The exact retained report is
[`qualification_report.json`](../yolozu/data/adaptive_routing/qualification_reports/qualification-20260925T190537Z-76ab14c7d5a2/qualification_report.json).

Observed results:

- status: `hold`
- failure: `max_p95_latency_exceeded`
- cold start: `6376.550708 ms`
- conservative p50: `5101.565541 ms`
- conservative p95: `7633.534416 ms`
- conservative p99: `9520.114875 ms`
- conservative repeat throughput: 200 images in 1,052,487,852,667 ns,
  approximately `0.190 fps`
- simple bbox mAP@50:95: `0.53920414`, above the fixed `0.2` threshold
- report digest:
  `ba3850695b760b059b4d61d6cc39e66c0c2f3cbff1326568a34208a579862b12`

The quality and throughput gates passed, but the report as a whole did not.
The `5000 ms` p95 limit was not raised after seeing the result. The exact report,
protocol, public input IDs, reproduction command, and checksums are retained;
model bytes, images, and annotations are not.

## Remaining gate

The Candidate stays unavailable to normal jobs. The separately frozen
[`torchvision_maskrcnn_service_single_cpu.json`](../configs/qualification/torchvision_maskrcnn_service_single_cpu.json)
matches the workload constructed by the MCP service, but it was not measured
after the blocking quality-profile result. It carries no support or performance
claim.

Resume only after an explicit, prospective decision about the intended p95
objective or a real performance improvement. A new threshold or implementation
must be reviewed before a new measurement; the retained hold report must not be
relabelled. A future qualified service workload would still need evidence
activation, complete support-profile review, Candidate-to-Experimental
promotion, and successful execution through all five MCP tools.

The local HTTP-protocol tests cover the five-tool surface and abstention. Public
operation still needs a deployment target, TLS/DNS, credentials, gateway request
limits, and OS/container isolation. None is claimed here.
