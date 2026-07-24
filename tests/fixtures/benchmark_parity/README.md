# Benchmark Artifact-Parity Fixtures

These deterministic JSON files exercise the shipped classification and OBB
artifact interface contracts. They are regression fixtures, not model-quality
or backend-runtime evidence:

- `backend_inference_executed` is `false`.
- `torch` and `onnx` identify benchmark artifact slots only.
- The benchmark command records the source-file SHA-256 and compares the
  normalized artifacts without claiming it ran either backend.
