# Provenance and license status

This tracked fixture exists to test dataset interfaces and lossless field
preservation. It is not evidence of model accuracy.

## Sources

- Images and instance annotations were selected from the repository's local
  COCO validation subset.
- `data/coco/annotations/instances_val2017.json` has SHA-256
  `ff849eb4cf125c4dd268e42eecb84e32d401a82ed61bc577d43aa43e9f114d0b`.
- `prepare_summary.json` records the derivation method for every label family.
  Bounding boxes and polygon masks use the local COCO annotations. Keypoints,
  depth, and object 6DoF pose are deterministic heuristics.

## License boundary

The local annotations file contains `images`, `annotations`, and `categories`,
but no `licenses` lookup table. The two validation images used by the qualified
round trip retain numeric license IDs `1` and `2`; those IDs cannot be mapped
to license text from the tracked metadata alone.

Consequently, this fixture does not make a complete upstream redistribution
license claim. Before publishing the fixture as a separate download, restore
and verify the authoritative per-image license mapping and review the
[official COCO terms of use](https://cocodataset.org/#termsofuse). YOLOZU's
Apache-2.0 code license does not replace dataset-specific terms.
