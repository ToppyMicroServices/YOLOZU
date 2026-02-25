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
python3 -m pip install 'yolozu[demo]'     # torch demos (CPU OK)
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
yolozu demo  # runs a small demo suite (instance-seg synthetic + coco128 + continual if torch is available)
yolozu demo instance-seg
yolozu demo continual --method ewc_replay
yolozu demo continual --compare --markdown

# COCO instances (polygon) mask demo

If you have a COCO-style instances annotations JSON (polygons) and the matching images directory, you can run:

```bash
yolozu demo instance-seg \
	--background coco-instances \
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

