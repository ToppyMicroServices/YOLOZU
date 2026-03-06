# Install

## Pip (recommended)

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
python3 -m pip install 'yolozu[train]'    # RT-DETR pose training scaffold
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

## CI dependency tiers

YOLOZU CI uses three install tiers to reduce optional-extras combinatorial noise:

- `core`: `pip install .` only (packaging + CLI smoke)
- `recommended`: pinned lock install via `requirements-ci.lock` (interface contract/behavior regression gates)
- `full`: GPU/backends (`tensorrt`, CUDA providers) in optional/nightly/manual workflows

For deterministic CI reproduction, install the same lock file locally:

```bash
python3 -m pip install -r requirements-ci.lock
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
