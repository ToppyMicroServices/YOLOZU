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
python3 -m pip install 'yolozu[demo]'     # torch/torchvision demos (CPU OK)
python3 -m pip install 'yolozu[onnxrt]'   # ONNX Runtime tooling
python3 -m pip install 'yolozu[train]'    # RT-DETR pose training scaffold
python3 -m pip install 'yolozu[coco]'     # COCOeval support (pycocotools)
python3 -m pip install 'yolozu[mcp]'      # MCP server integration
python3 -m pip install 'yolozu[actions]'  # Actions/OpenAPI integration (FastAPI)
python3 -m pip install 'yolozu[full]'     # everything above
```

Note: PyTorch wheels are platform-dependent. If `pip install 'yolozu[demo]'` fails, follow the official PyTorch install selector for your platform, then re-install `yolozu[demo]`.

## CPU demos (quick sanity checks)

These demos are optional and intended as fast end-to-end smoke checks.
They typically require `pip install 'yolozu[demo]'`.

```bash
yolozu demo  # runs a small demo suite (prefers COCO instances if available)
yolozu demo instance-seg  # defaults to --background coco-instances and --inference auto (requires torch/torchvision)
yolozu demo continual --method ewc_replay
yolozu demo continual --compare --markdown
```

More practical continual example (vision backbone):

```bash
yolozu demo continual --problem mnist_rotate --method ewc
```

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

