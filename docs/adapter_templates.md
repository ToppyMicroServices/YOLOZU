# Adapter templates and onboarding

This page defines the required adapter starter routes for parity-ready onboarding:

- `mmdet`
- `detectron2`
- `yolo_runtime`
- `rtdetr`
- `opencv_dnn`
- `custom_cpp`

Starter files live under `examples/adapter_starters/`.

## Packaged SDK quick start

Installed packages can advertise plugins through the
`yolozu.adapters.v1` Python entry-point group. Listing reads package metadata
without importing provider code:

```bash
yolozu adapter list
```

After reviewing the selected distribution, explicitly load its metadata or run
the predictions interface conformance check:

```bash
yolozu adapter doctor my_adapter --allow-plugin-load
yolozu adapter test my_adapter --allow-plugin-load
```

`adapter test` uses a packaged 8x8 network-free image by default. A provider
that needs framework-specific input can accept `--config config.json` and
`--records records.json`. The entry point must resolve to an object or no-arg
class with these fields:

- `adapter_id`: the exact entry-point name
- `interface_version`: integer `1`
- `summary`: nonempty text
- `capabilities`: JSON-serializable object
- `create(config)`: returns an object implementing `predict(records)`

Python callers can use `discover_adapters`, `load_adapter_plugin`, and
`test_adapter_plugin` from `yolozu.adapter_sdk`. Loading always requires
`allow_plugin_load=True`; discovery never imports third-party code.

## Quick start

1. Pick a starter matching your framework.
2. Run your framework inference and emit YOLOZU predictions JSON (`{image,detections}`).
3. Validate schema:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

4. Add your output to parity suite:

```bash
python3 tools/adapter_parity_suite.py \
  --adapter-predictions rtdetr=/path/to/reference.json \
  --adapter-predictions mmdet=/path/to/mmdet.json \
  --adapter-predictions detectron2=/path/to/detectron2.json \
  --adapter-predictions yolo_runtime=/path/to/yolo_runtime.json \
  --adapter-predictions opencv_dnn=/path/to/opencv_dnn.json \
  --adapter-predictions custom_cpp=/path/to/custom_cpp.json \
  --reference-adapter rtdetr \
  --output reports/adapter_parity_suite.json
```

## Onboarding checklist for a new adapter

1. Implement inference wrapper from a starter file.
2. Produce at least one smoke predictions artifact and validate it.
3. Add the adapter output into `tools/adapter_parity_suite.py` run and confirm parity report generation.
4. Document any framework-specific install/runtime notes in the adapter-specific starter file.
