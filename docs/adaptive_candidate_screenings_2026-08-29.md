# Adaptive candidate screenings — 2026-08-29

Collected at 2026-08-28T18:26:28Z (UTC), corresponding to 2026-08-29 in
Asia/Tokyo. Only the official primary sources linked below were used. No code,
weight, model, dataset, or package was downloaded or executed.

These are metadata/design decisions. Published benchmark values are catalog
context only. They are not YOLOZU measurements, qualification, support,
selection, promotion, or proof of human adoption.

## Grounding DINO plus SAM 2.1

Decision: `hold`.

The screened composite fixes Grounding DINO commit
[`856dde20aee659246248e20734ef9ba5214f5e44`](https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44)
and SAM 2.1 commit
[`2b90b9f5ceec907a1c18123530e92e794ad901a4`](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4).
Both repositories identify their code license as Apache-2.0. SAM 2 explicitly
states that its model checkpoints are Apache-2.0 in the
[official license section](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license).
The Grounding DINO release provides the exact
[`groundingdino_swint_ogc.pth`](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha)
asset, but the inspected official material does not state a separate weight
license or SHA-256 for it. The composite weight-license result is therefore
`unknown`, even though the code license is known.

SAM 2.1 requires Python 3.10+, PyTorch 2.5.1+, torchvision 0.20.1+, and its
official install path may compile a CUDA extension. Its official checkpoint
list includes the
[`sam2.1_hiera_tiny.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
variant. Grounding DINO's official installation also uses a compiled local
PyTorch/CUDA path. This combined runtime is not a shipped and qualified YOLOZU
adaptive runner/provider surface. It remains `runtime_provider_new_surface_hold`.

The proposed predictions interface contract mapping is:

- Grounding DINO's selected text similarity becomes `score`; the matched prompt
  token/phrase becomes the bounded requested `label`; and the returned box is
  validated as the detection `bbox`.
- The exact selected box is the only SAM prompt. A successfully validated SAM
  mask becomes the referenced mask artifact for that same instance. A boxes-only
  or failed SAM result cannot synthesize a mask or claim instance segmentation.
- Published Grounding DINO behavior returns 900 boxes by default with word-level
  similarities. Prompt-token grouping and all result caps must be frozen in a
  later adapter interface before execution.

The code archive SHA-256, both exact weight SHA-256 values, evaluation-dataset
license, measured compute/peak memory, maintenance decision for the fixed 2024
commits, and supply-chain review remain unknown. The adapter and demanded-host
qualification Beads stay deferred. Reconsider only after an exact runtime and
isolation decision passes, all source/weight SHA-256 values and license evidence
are reviewed, the dataset/protocol is preregistered, and local resource bounds
are measured.

Canonical record:
`screening-groundingdino-sam21-20260829` / digest
`888d12bd564508033c7fe1b6186e0ba10e6821084a2924e5fa4cd099b7d0da55`.

## RF-DETR Nano 1.9.4

Decision: `hold`.

The screened detector fixes official release
[`1.9.4`](https://github.com/roboflow/rf-detr/releases/tag/1.9.4) and its immutable
commit
[`9b009fa928d6218320439803d1da01869a85c072`](https://github.com/roboflow/rf-detr/tree/9b009fa928d6218320439803d1da01869a85c072).
The official README separates the Apache-2.0 open-source package and
Apache-designated Nano/Small weights from PML-1.0 Plus components in its
[license section](https://github.com/roboflow/rf-detr/blob/1.9.4/README.md#license).
This screening excludes every Plus/XL/2XL component.

The official 1.9.4 weight registry fixes the Nano URL and an MD5 value in
[`model_weights.py`](https://github.com/roboflow/rf-detr/blob/1.9.4/src/rfdetr/assets/model_weights.py).
It does not provide the SHA-256 required by the YOLOZU artifact interface
contract. A later acquisition must fetch only that exact URL, verify the
published MD5 as an upstream check, compute SHA-256, and review the resulting
size/hash before registry entry. This task did not download the weight.

The fixed-class predictions mapping is direct: validated `xyxy` becomes `bbox`,
confidence becomes `score`, and the checkpoint-bound COCO class index maps
through an immutable vocabulary to `label`. No mask is synthesized for this
detection-only candidate.

The official table reports Nano at 384x384 and places all latency values in the
specific NVIDIA T4, TensorRT FP16, batch-1 benchmark context described in the
[official benchmark section](https://github.com/roboflow/rf-detr/blob/1.9.4/README.md#benchmarks).
Those numbers are not YOLOZU evidence and do not establish a low-latency result
on any demanded host. The package metadata requires PyTorch 2.2+ and
torchvision 0.17+, with separate optional export/runtime dependencies. The
`rfdetr` adapter/runtime is not a shipped and qualified YOLOZU adaptive surface,
so it remains `runtime_provider_new_surface_hold`.

The source archive SHA-256, weight SHA-256, evaluation-dataset license review,
measured operations/whole-run peak memory, and supply-chain review remain
unknown. The adapter and demanded-host qualification Beads stay deferred.
Reconsider only after an exact runtime/isolation decision passes, SHA-256 and
dataset-license evidence are reviewed, and the candidate is measured under the
same frozen YOLOZU protocol as existing baselines.

Canonical record: `screening-rfdetr-nano-1.9.4-20260829` / digest
`1e9daf2d959f1f1c8d847d9cf02f85e69b226b457c974ee8b5676750f62cd777`.

## Alternatives and branch gates

Newer SAM-family, YOLOE, and other efficient detector options were not silently
substituted. They require their own immutable screening records. A valid outcome
for both current candidates is no adapter.

The exact-title Beads queries resolve one adapter and one demanded-host
qualification Bead for each P1 decision. Because both current records are hold,
all four remain deferred to 2099-12-31 with the reconsideration triggers above.
