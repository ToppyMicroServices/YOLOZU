# Roadmap

The **source of truth** for planned / in-progress work is the Beads (bd) issue tracker:

```bash
bd list
bd ready
```

This `docs/` folder keeps longer-form planning notes for context:

- `yolozu/data/manifest/adaptive_vision_roadmap.json` — dated, packaged public projection of the adaptive local-vision scope and guardrails
- `reports/adaptive_vision_roadmap.md` — human-readable report generated from that projection
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

## Adaptive local vision projection

The live task graph for environment-qualified local vision starts at Bead `YOLOZU-ll2.81`. Beads remains authoritative for issue status, dependencies, ownership, and completion.

The packaged [`adaptive_vision_roadmap.json`](../yolozu/data/manifest/adaptive_vision_roadmap.json) is a dated public projection of product scope, target maturity, and safety boundaries. It intentionally omits live status and personal ownership data. The generated [`adaptive_vision_roadmap.md`](../reports/adaptive_vision_roadmap.md) is a report, not qualification evidence.

Regenerate or verify the report with:

```bash
python3 tools/generate_adaptive_vision_roadmap.py --check --json
```
