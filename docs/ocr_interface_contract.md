# OCR result interface contract

Status: contract-only P2 foundation. YOLOZU does not ship an OCR model, adapter,
weight, document parser, remote service, or support claim.

`OCRResult` is separate from the existing predictions interface contract. OCR
text is not a detection class label. The canonical schemas are:

- `docs/schemas/ocr_bundle_interface.schema.json`
- `docs/schemas/ocr_result.schema.json`

The byte-identical packaged copies are under `yolozu/data/schemas/`. Python
validation and runner mapping live in `yolozu/contracts/ocr.py`.

## Trust boundary

The runner may return recognized text, geometry, the two confidence values, and
either an in-range language/script index or `unknown`. It may not choose a
language ID, script ID, component ID, model ID, path, URL, or setting. Core maps
the indices through the pinned immutable bundle interface and stamps the exact
detector and recognizer provenance after all output validation succeeds.

Every item has `content_trust=untrusted_input_derived`. Text is preserved as
valid UTF-8 without semantic normalization, but it is inert user-owned output.
It must not be interpreted as an instruction or copied into selection decisions,
telemetry, benchmark summaries, generic errors, support logs, or downstream tool
arguments. The contract exposes only a count-based privacy-safe summary.

## Result boundary

V1 requires exactly one ordered `top_left`, `top_right`, `bottom_right`,
`bottom_left` quadrilateral. Python validation rejects duplicate, collinear,
concave, self-intersecting, wrongly ordered, wrongly wound, noncanonical, and
out-of-image coordinates. Detection and recognition confidence remain separate
CanonicalDecimalV1 values in the inclusive range 0 through 1. No combined
confidence is inferred.

The bundle declares 1..256 language IDs, 1..128 script IDs, and exactly two
components in detector/recognizer order. IDs are bounded ASCII tokens. Component
or model provenance is never accepted from runner output.

Text, geometry, per-image, and whole-job limits are checked before a successful
result is assembled. There is no truncation. Empty success is valid only when
both pinned components succeeded and the detector returned zero regions. A
timeout, crash, missing component, or invalid output raises a stable failure and
publishes no partial result.

## Input and feature scope

The input media preflight accepts bounded JPEG, PNG, or single-frame WebP only.
PDF, TIFF, animated or multipage images, archives, office documents, and video
are invalid in v1. `logical_page_reference` is display-only and cannot influence
input indexing or paths.

Translation, PII classification, semantic document understanding, handwriting
support, and remote OCR are outside this contract. Each needs separate screening
and evidence before implementation.
