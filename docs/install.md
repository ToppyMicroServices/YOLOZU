# Install

## Pip (default stable path)

```bash
python3 -m pip install yolozu
yolozu --help
yolozu doctor --output -
```

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
MPS is supported when `torch.backends.mps.is_available()` is `true`.

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

## macOS / Apple Silicon Miniforge/MPS workflow

You can build the environment with plain Python tooling (`venv` + `pip`) first. Miniforge/conda is not a hard requirement for YOLOZU or for Python itself. The fallback exists because MPS availability is decided by the installed Torch binary/runtime combination, not by whether Python can create the environment.

If a `pip`-installed PyTorch build reports `mps_available=false` on a compatible Apple Silicon Mac, try a Miniforge/conda PyTorch build before giving up on MPS.

The repo was verified on `macOS 26.3.1 arm64` with:

- `pip` PyTorch wheels: `mps_built=true`, `mps_available=false`
- Miniforge/conda PyTorch: `mps_built=true`, `mps_available=true`
- `rtdetr_pose/tools/train_minimal.py --device mps --dry-run`: completed on MPS

Suggested setup:

```bash
git clone https://github.com/ToppyMicroServices/YOLOZU.git
cd YOLOZU
curl -L -o /tmp/Miniforge3-MacOSX-arm64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash /tmp/Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/bin/activate"
conda create -y -n yolozu-mps python=3.11 pytorch torchvision -c pytorch
conda activate yolozu-mps
python -m pip install -e '.[train]' --no-deps
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

Treat this as the qualification gate:

- `torch.backends.mps.is_available() == true` means the MPS path is actually usable on this machine
- `macos_ok: true` in the manifest only means the CLI can run on macOS; it does not guarantee MPS availability
- if `mps_available=false`, stay on `cpu` or `--device auto`

Small training smoke:

```bash
PYTHONPATH="$PWD:$PWD/rtdetr_pose" \
python rtdetr_pose/tools/train_minimal.py \
  --device mps \
  --amp none \
  --dry-run \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --run-dir runs/mps_train_smoke
```

Expected signals:

- `yolozu doctor --output -` shows `runtime_capabilities.torch.mps_available: true`
- `runs/mps_train_smoke/run_record.json` records `args.device: "mps"` and `hardware.accelerator.mps.available: true`
- if ONNX export warns about missing `onnx`, training still succeeded; install `onnx` only if you need post-train export in the same env

If MPS still stays unavailable:

- keep `PYTORCH_ENABLE_MPS_FALLBACK=1` for partial CPU fallback when an op is unsupported
- prefer `--device auto` for day-to-day safety
- treat Miniforge/conda as a workaround for Torch packaging/runtime mismatches, not as a requirement for Python environment creation
- compare `pip` and `conda` outputs with `yolozu doctor --output -` to isolate whether the blocker is the Torch build or the repo config

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

# COCO instances (polygon) mask demo

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
