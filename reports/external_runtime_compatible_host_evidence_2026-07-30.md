# Compatible-Host External Runtime Qualification — 2026-07-30

Status: **Linux/CUDA qualification workflow dispatched; final per-lane result
is recorded from the completed immutable workflow run**.

## Scope

The compatible-host workflow supplements, rather than rewrites, the macOS CPU
availability result in
[`external_runtime_evidence_2026-07-30.md`](external_runtime_evidence_2026-07-30.md).
It pins and invokes non-dry training for:

- YOLOX at source commit
  `6ddff4824372906469a7fae2dc3206c7aa4bbaee`;
- MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.3.0, MMPose 1.3.2, and MMSeg 1.2.2;
- NVIDIA TAO Toolkit 5.5.0 in its vendor container.

The repository-owned preparation step creates bounded detection, keypoint, and
segmentation layouts from the tracked real-image fixture. Its generated labels
are marked runtime-only and are not task-quality ground truth.

## Reproduction command

```bash
bash scripts/run_external_runtime_gpu_qualification.sh \
  --output-dir reports/compatible_host_external_runtimes \
  --dataset-root data/real_multitask_fewshot
```

The self-hosted workflow run is
[30536643127](https://github.com/ToppyMicroServices/YOLOZU/actions/runs/30536643127),
at source commit `9ab7a4f04d8f568b1c50b907a5309295b95d43c7`.

## Evidence boundary

A zero exit status is not enough. Each lane must record non-dry training,
runtime/config/dataset/checkpoint hashes, resource use, and a structured
outcome. A runtime availability pass does not establish training quality.
Optional third-party runtimes and their licenses remain outside YOLOZU.
