# Runpod / Linux GPU quickstart (TensorRT pipeline)

This folder is a minimal, repo-local entrypoint for running the **YOLO26 TensorRT pipeline** on a Linux + NVIDIA GPU
machine (e.g., Runpod). It is designed so macOS users can still work on CPU-only tooling, and defer GPU work to Runpod.

It also includes a one-command runner for the **RTDETRPose backend suite** (PyTorch → ONNX → TensorRT → parity/benchmark).

## Inputs

- **YOLO-format COCO dataset root** (images + labels):
  - `.../images/<split>/*.{jpg,jpeg,png,bmp,tif,tiff,webp,gif}`
  - `.../labels/<split>/*.txt`
- **ONNX models** per bucket (file names are flexible; use a template):
  - `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`

## Outputs (default paths)

- TensorRT engines: `engines/{bucket}_{precision}.plan`
- Engine build metadata: `reports/trt_engine_{bucket}_{precision}.json`
- ONNXRuntime predictions: `reports/pred_onnxrt_{bucket}.json`
- TensorRT predictions: `reports/pred_trt_{bucket}.json`
- Parity reports: `reports/parity_{bucket}.json`
- COCO suite report: `reports/eval_suite_trt.json`
- Per-engine latency: `reports/latency_{bucket}.json`
- Latency summary report: `reports/benchmark_latency.json` (+ optional `reports/benchmark_latency.jsonl`)

## One-command pipeline

Run the wrapper script (it calls `tools/run_trt_pipeline.py`):

```bash
bash deploy/runpod/run_trt_pipeline.sh \
  --dataset /data/coco-yolo \
  --onnx-template /data/models/{bucket}.onnx \
  --precision fp16 \
  --combined-output output0 \
  --boxes-scale abs \
  --min-score 0.0 \
  --topk 300 \
  --max-images 500
```

Notes:
- `--combined-output output0` matches common Ultralytics ONNX exports. If your ONNX has different output names/layout,
  adjust the exporter flags.
- The pipeline supports `--dry-run` to generate schema-correct artifacts without requiring TensorRT/onnxruntime/cv2.

## RTDETRPose backend suite (PyTorch → ONNX → TRT)

Run the wrapper script (it calls `tools/run_rtdetr_pose_backend_suite.py`):

```bash
bash deploy/runpod/run_rtdetr_pose_backend_suite.sh \
  --config rtdetr_pose/configs/base.json \
  --checkpoint /data/checkpoints/rtdetr_pose.pt \
  --device cuda \
  --precision fp16 \
  --dynamic-hw \
  --export-image-size 320 \
  --suite-image-size 640
```

Outputs (default paths):
- Run folder: `runs/rtdetr_pose_backend_suite/<run_id>/`
- Suite report: `runs/rtdetr_pose_backend_suite/<run_id>/backend_suite.json`
- Pipeline record: `runs/rtdetr_pose_backend_suite/<run_id>/pipeline.json`
- ONNX + meta: `runs/rtdetr_pose_backend_suite/<run_id>/model.onnx` (+ `.meta.json`)
- Engine + meta: `runs/rtdetr_pose_backend_suite/<run_id>/model_fp16.plan` (+ `.meta.json`)

## Docker skeleton

TensorRT is not installed via pip; use an NVIDIA/TensorRT base image and add Python deps for YOLOZU’s tooling.

## Prebuilt images (GHCR)

`.github/workflows/container.yml` builds these images on release-tag pushes (`vX.Y.Z`) and manual workflow runs.
Pull requests do not build container images by default; run the workflow manually when a container change needs validation before release.
Release-tag pushes and manual workflow runs publish images to GHCR.
All published images also receive the `latest` tag on release-tag builds.

CPU images:

```bash
docker pull ghcr.io/toppymicroservices/yolozu:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu-demo:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu:latest
docker pull ghcr.io/toppymicroservices/yolozu-demo:latest
```

Optional RunPod GPU images are published to GHCR and mirrored to NVIDIA NGC (requires GitHub secret `NGC_API_KEY`):

```bash
docker pull ghcr.io/toppymicroservices/yolozu-trt:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu-rtdetr-pose:X.Y.Z
docker pull ghcr.io/toppymicroservices/yolozu-trt:latest
docker pull ghcr.io/toppymicroservices/yolozu-rtdetr-pose:latest
docker pull nvcr.io/yolozu/yolozu-trt:X.Y.Z
docker pull nvcr.io/yolozu/yolozu-rtdetr-pose:X.Y.Z
docker pull nvcr.io/yolozu/yolozu-trt:latest
docker pull nvcr.io/yolozu/yolozu-rtdetr-pose:latest
```

Notes:
- Pull-request validation only checks that the Dockerfiles still build; it does not push images.
- Manual republish should use **Actions → container → Run workflow** with `release_tag=vX.Y.Z` so GHCR and NGC receive the same release tags.
- GPU image publication still requires `NGC_API_KEY`, even when the workflow is triggered manually.

Build (example):

```bash
docker build --pull -f deploy/runpod/Dockerfile -t yolozu-trt:local .
```

For RTDETRPose (installs torch + onnxruntime-gpu), use:

```bash
docker build --pull -f deploy/runpod/Dockerfile.rtdetr_pose -t yolozu-rtdetr-pose:local .
```

Run (example):

```bash
docker run --gpus all -it --rm \
  -v "$PWD:/workspace/YOLOZU" \
  -v "/data:/data" \
  yolozu-trt:local
```

Then run the pipeline command above inside the container.

## Compliance artifacts (doctor + dependency licenses)

Inside the container, you can emit reproducibility/compliance artifacts:

```bash
bash deploy/runpod/run_compliance.sh
```

This writes:
- `reports/doctor.json`
- `reports/dependency_licenses.json`

## Beads (bd) sync on RunPod (fix for stale `bd list`)

If the RunPod repo was cloned with `git clone --single-branch`, `git fetch origin beads-sync` may update only
`FETCH_HEAD` and *not* update `refs/remotes/origin/beads-sync`, leaving `bd list` stale.

Use the helper (recommended):

```bash
bash deploy/runpod/refresh_beads_sync.sh
```

One-time alternative: teach `origin` to track `beads-sync`, then `git fetch origin` will keep it fresh:

```bash
git remote set-branches --add origin beads-sync
git fetch origin
```

## COCO val2017 bootstrap (optional)

To download COCO val2017 + annotations and prepare a YOLO-format dataset (without writing large artifacts into
`/workspace`), run:

```bash
bash deploy/runpod/bootstrap_coco_val2017.sh
```

This produces:
- raw COCO under `/tmp/coco`
- YOLO dataset under `/tmp/coco-yolo` (use this path with `--dataset`)
