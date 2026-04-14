# Roadmap

The **source of truth** for planned / in-progress work is the Beads (bd) issue tracker:

```bash
bd list
bd ready
```

This `docs/` folder keeps longer-form planning notes for context:

- `docs/roadmaps/pytorch_trt.md` — PyTorch/ONNX/TensorRT implementation notes (historical)
- `docs/roadmaps/yolo26_competition.md` — YOLO26 toolchain goals (historical; see bd epic)
- `docs/roadmaps/symmetry_commonsense_realtime.md` — Symmetry/commonsense constraint plan (historical)
- `docs/yolo26_size_buckets.md` — YOLO26 n/s/m/l/x size bucket envelopes (params/FLOPs)

If you are working on RunPod and `bd list` looks stale, see:

- `deploy/runpod/README.md` (“Beads (bd) sync on RunPod”)
- `deploy/runpod/refresh_beads_sync.sh`
