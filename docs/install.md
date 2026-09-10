# Install

## Pip (default stable path)

Requires Python 3.10 or newer. A virtual environment avoids modifying a
system-managed Python installation. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install yolozu
yolozu --help
yolozu doctor --output -
```

On Windows PowerShell, activation is optional when you call the environment's
Python directly:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install yolozu
.\.venv\Scripts\python.exe -m yolozu --help
.\.venv\Scripts\python.exe -m yolozu doctor --output -
```

If `yolozu` is not found after installation, use `python3 -m yolozu` from the
same environment, or reactivate it. Do not install optional model runtimes just
to make every `doctor` capability available; core validation and the synthetic
demo do not need Torch, CUDA, or MPS.

Try a local demo without a dataset or model download:

```bash
yolozu demo instance-seg --background synthetic --inference none --run-dir reports/quickstart_instance_seg --progress
```

The report and PNG overlays under `reports/quickstart_instance_seg/` verify
the workflow using generated data. They do not measure a real model's accuracy.
For real object-detection metrics, install `yolozu[coco]` and follow the
[README evaluation steps](../README.md#install-and-evaluate-your-predictions).
If you already have predictions, no model runtime or repository checkout is
needed for that evaluation path.

## Optional extras

Install only what you need:

```bash
python3 -m pip install 'yolozu[demo]'     # torch/torchvision demos (CPU OK; includes timm + opencv-contrib + transformers for depth demo)
python3 -m pip install 'yolozu[onnxrt]'   # ONNX Runtime tooling
python3 -m pip install 'yolozu[train]'    # RT-DETR pose reference trainer
python3 -m pip install 'yolozu[coco]'     # COCOeval support (pycocotools)
python3 -m pip install 'yolozu[mcp]'      # MCP server integration
python3 -m pip install 'yolozu[actions]'  # Actions/OpenAPI integration (FastAPI)
python3 -m pip install 'yolozu[full]'     # everything above
```

If you are running from a source checkout (editable install), install extras like:

```bash
python3 -m pip install -e '.[demo]'
```

Note: PyTorch wheels are platform-dependent. If `pip install 'yolozu[demo]'` fails, follow the official PyTorch install selector for your platform, then re-install `yolozu[demo]`.

## macOS / Apple Silicon beta scope

`Torch backend on macOS/MPS` is a qualification path in this repo, not a blanket production-ready claim.
`torch.backends.mps.is_available()` reports whether this process can use MPS.
A true result does not qualify every model, operator, precision, or workload.

- good fit: `yolozu demo`, `yolozu export --backend torch`, small `rtdetr_pose/tools/train_minimal.py` smoke runs
- not in scope: TensorRT engine build/run paths (`trtexec`, CUDA-only workflows)

Recommended environment hint when an op is not yet implemented on MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 yolozu doctor --output -
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 rtdetr_pose/tools/train_minimal.py --device mps --help
```

Training-device notes:

- `--device auto` now resolves in `cuda -> mps -> cpu` order
- `--device mps` is allowed for the reference trainer
- `--amp fp16|bf16` on MPS is best-effort beta; if autocast is unavailable, the trainer warns and falls back to fp32
- post-train ONNX export is attempted on CPU by default, even when training itself ran on MPS/CUDA

## macOS / Apple Silicon MPS workflow

Use `venv` and `pip` for a new source-checkout environment. Miniforge/conda is
optional. PyTorch stopped publishing to its official Conda channel starting
with [PyTorch 2.6](https://pytorch.org/blog/pytorch2-6/), so the old
`conda install ... -c pytorch` route is not a source for the `torch>=2.10`
required by YOLOZU's training extra. Do not bypass dependency checks with
`--no-deps` during a fresh install.

MPS availability depends on the runtime and the process's access to the device.
If a probe fails inside a restricted runner, repeat the same probe with the same
Python interpreter in a normal terminal before replacing the environment.

An earlier `macOS 26.3.1 arm64` check recorded:

- `pip` PyTorch wheels: `mps_built=true`, `mps_available=false`
- Miniforge/conda PyTorch: `mps_built=true`, `mps_available=true`
- `rtdetr_pose/tools/train_minimal.py --device mps --dry-run`: completed on MPS

Those observations do not isolate packaging as the cause and are not a current
support matrix. Follow the current [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
when diagnosing your environment.

Source-checkout setup:

```bash
git clone https://github.com/ToppyMicroServices/YOLOZU.git
cd YOLOZU
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[train]'
```

Verify MPS before longer runs:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps_built", torch.backends.mps.is_built())
print("mps_available", torch.backends.mps.is_available())
print(torch.ones(2, device="mps"))
PY
yolozu doctor --output -
```

Interpret the probe separately from workload qualification:

- `torch.backends.mps.is_available() == true` plus a successful tensor allocation confirms device access for this process
- `macos_ok: true` in the manifest only means the CLI can run on macOS; it does not guarantee MPS availability
- if `mps_available=false`, stay on `cpu` or `--device auto`

Small training smoke:

Use the committed `data/smoke` fixture; a fresh checkout does not contain a
downloaded `data/coco128` dataset. The trainer's `--dry-run` performs one
optimization step, unlike evaluation `--dry-run`, which skips metric computation.
This checks execution, not model quality.

```bash
PYTHONPATH="$PWD:$PWD/rtdetr_pose" \
python rtdetr_pose/tools/train_minimal.py \
  --device mps \
  --amp none \
  --dry-run \
  --dataset-root data/smoke \
  --split val \
  --config rtdetr_pose/configs/base.json \
  --run-dir runs/mps_train_smoke
```

Expected signals:

- `yolozu doctor --output -` shows `runtime_capabilities.torch.mps_available: true`
- `runs/mps_train_smoke/run_record.json` records `args.device: "mps"` and `hardware.accelerator.mps.available: true`
- if ONNX export warns about missing `onnx`, training still succeeded; install `onnx` only if you need post-train export in the same env

If MPS still stays unavailable:

- `PYTORCH_ENABLE_MPS_FALLBACK=1` can handle unsupported operations after MPS is available; it does not make an unavailable device visible
- use `cpu` or `--device auto` to continue without MPS
- compare the interpreter, Torch version, architecture, and execution context before changing packages

## CI dependency tiers

YOLOZU CI uses three install tiers to reduce optional-extras combinatorial noise:

- `core`: `pip install .` only (packaging + CLI smoke)
- `recommended`: pinned lock install via `requirements-locks/requirements-ci.lock` (interface contract/behavior regression gates)
- `full`: GPU/backends (`tensorrt`, CUDA providers) in optional manual workflows

For deterministic CI reproduction, install the same lock file locally:

```bash
python3 -m pip install -r requirements-locks/requirements-ci.lock
```

Detailed mapping (jobs/workflows + rationale): [`ci_dependency_tiers.md`](ci_dependency_tiers.md).  
Source metadata is tracked in `pyproject.toml` under `[tool.yolozu.ci_tiers.*]` and `[tool.yolozu.optional_extras_rationale]`.

## CPU demos (quick sanity checks)

These demos are optional and intended as fast end-to-end smoke checks.
They typically require `pip install 'yolozu[demo]'`.

```bash
yolozu demo  # runs a small demo suite (prefers COCO instances if available)
yolozu demo instance-seg  # short path: uses COCO instances if present, otherwise falls back to a synthetic demo
yolozu demo keypoints  # Keypoint R-CNN inference on a sample image
yolozu demo pose  # 6D pose demo (chessboard + OpenCV solvePnP)
yolozu demo pose --backend aruco  # ArUco marker pose (requires opencv-contrib; cached sample in demo_output/pose/_samples)
yolozu demo pose --backend densefusion  # heavy: CUDA + large downloads
yolozu demo depth  # monocular depth inference (default: Depth Anything; downloads weights on first run)
yolozu demo depth --compare  # compare Depth Anything + MiDaS + DPT in one run
yolozu demo train  # MNIST fine-tune demo (bounded by --max-steps; downloads ResNet18 on first run)
yolozu demo continual --method ewc_replay
yolozu demo continual --compare --markdown
```

More practical continual example (vision backbone):

```bash
yolozu demo continual --problem mnist_rotate --method ewc
```

> **Data placement:** See [training_inference_export.md § Canonical COCO data placement](training_inference_export.md#canonical-coco-data-placement) for the full directory standard and copy-paste setup commands.

## COCO instances (polygon) mask demo

The download helper below requires a source checkout and downloads COCO assets.
The explicit synthetic demo above works from a pip install without those files.

If you don't have COCO instances data yet, you can download a tiny subset (2 images) locally:

```bash
python3 scripts/download_coco_instances_tiny.py --num-images 2
python3 scripts/download_coco_instances_tiny.py --help
```

This writes:

- `data/coco/annotations/instances_val2017.json`
- `data/coco/images/val2017/` (a few JPEGs)

Then `yolozu demo` will auto-detect it and run the polygon-mask instance-seg demo.

To run *real* instance segmentation inference (Mask R-CNN via `torchvision`) on those images, use:

```bash
yolozu demo instance-seg \
	--inference torchvision
```

To run a visible raw-vs-TTA compare on a corrupted real COCO image, use:

```bash
yolozu demo instance-seg-tta \
	--run-dir reports/demo_instance_seg_tta
```

This scans a small set of COCO polygon-mask images, applies a deterministic corruption, compares raw Mask R-CNN predictions against augmentation-based TTA (brightness-lift + hflip for brightness corruption; hflip otherwise), and writes:

- `reports/demo_instance_seg_tta/selected/overlay_raw.png`
- `reports/demo_instance_seg_tta/selected/overlay_tta.png`
- `reports/demo_instance_seg_tta/selected/overlay_delta.png`
- `reports/demo_instance_seg_tta/instance_seg_tta_demo_report.json`

If your COCO data is under the default paths, you can omit `--coco-instances-json` and `--coco-images-dir`.
Defaults:

- `data/coco/annotations/instances_val2017.json`
- `data/coco/images/val2017`

For `--background coco-instances`, if you omit `--inference`, it defaults to `auto` (real inference when available).
To force the lightweight GT-derived fallback (no torch needed), use:

```bash
yolozu demo instance-seg --inference none
```

To run the fully synthetic variant explicitly:

```bash
yolozu demo instance-seg --background synthetic
```

To run a YOLO-style bbox dataset variant (real images, pseudo masks derived from YOLO labels):

```bash
yolozu demo instance-seg --background yolo-bbox --yolo-root /path/to/yolo_dataset --yolo-split val --inference none
```

If you have a COCO-style instances annotations JSON (polygons) and the matching images directory, you can run:

```bash
yolozu demo instance-seg \
	--background coco-instances \
	--coco-instances-json /path/to/annotations/instances_val2017.json \
	--coco-images-dir /path/to/images/val2017
```

To run the demo suite (no subcommand) but still include the COCO instances polygon demo:

```bash
yolozu demo \
	--coco-instances-json /path/to/annotations/instances_val2017.json \
	--coco-images-dir /path/to/images/val2017
```

Demo outputs are written under `demo_output/` by default.

## Repository checkout (dev path)

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pip install -e .
python3 -m unittest -q
```

To enable demos in a source checkout:

```bash
python3 -m pip install -e '.[demo]'
python3 scripts/download_coco_instances_tiny.py  # optional: enables coco-instances background without long flags
yolozu demo instance-seg
```
