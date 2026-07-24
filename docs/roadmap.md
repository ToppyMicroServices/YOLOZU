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

If `bd list` looks stale, refresh the local database from the exported
`beads-sync` snapshot:

```bash
bash refresh_beads_sync.sh
```

For RunPod and two-machine details, see:

- `docs/beads_github_workflow.md`
- `deploy/runpod/README.md` (“Beads (bd) refresh on RunPod”)
- `deploy/runpod/refresh_beads_sync.sh`
