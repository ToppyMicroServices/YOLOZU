# Version Compatibility

This page separates install floors, repository-pinned validation environments,
protocol choices, and GPU environment evidence. They are not interchangeable:
an install floor is not a claim that every newer version was qualified, and a
configured GPU container is not proof that a particular host completed a run.

## Package and CPU validation envelope

The candidate labeled-sample workflow builds a wheel from the tested revision,
installs it non-editably into a clean environment, and checks its installed file
hashes against that exact wheel. Version strings alone are insufficient: a local
candidate and a published package may both report `4.7.0` but contain different
code. `pip check` is run against the installed distribution, not stale editable
metadata from a development environment.

The configured matrix in `.github/workflows/sample_compatibility.yml` covers:

| Environment | Dependencies | Check |
|---|---|---|
| Linux, Python 3.10–3.14 | Latest resolvable core + `coco` | Generate labels, strict validation, official COCO evaluation, then repeat after relocation |
| Linux, Python 3.10 | NumPy 1.24.0, PyYAML 6.0, Pillow 12.2.0, typing_extensions 4.8.0, pycocotools 2.0.7 | Same check at the declared core/COCO floors |
| macOS, Python 3.14; Windows, Python 3.12 | Latest resolvable core + `coco` | Same installed-wheel check |

Configuration is not a test result. Each run uploads `compatibility-report.json`
with the wheel hash, source revision, actual runtime/dependency versions, metrics,
and sample hashes. Re-run it for a new wheel or environment:

```bash
python3 tools/ci/check_sample_compatibility.py \
  --wheel /path/to/yolozu-candidate.whl \
  --output-dir reports/sample_compatibility
```

Run the helper with the Python interpreter where that wheel and its `coco` extra
are already installed. It refuses an existing output directory. See the
[labeled sample](labeled_sample.md) for user-facing commands. These checks do not
exercise Torch, ONNX, GPU inference, or external training runtimes.

| Component | Declared install floor | Current repository-pinned evidence | Scope |
|---|---|---|---|
| PyTorch | `torch>=2.10.0` in the Torch-backed extras | `torch==2.10.0+cpu` in `requirements-locks/requirements-ci.lock`; `torch==2.10.0` in the demo and RT-DETR locks | The CI pin qualifies repository CPU tests. Device, accelerator, and custom-wheel behavior remain environment-specific. |
| Torchvision | `torchvision>=0.25.0` in the Torch-backed extras | `torchvision==0.25.0` in `requirements-locks/requirements-demo-extra.lock` | Applies to the demo/runtime bundle, not the dependency-free validation/evaluation core. |
| ONNX | `onnx>=1.21.0` in ONNX-backed extras | `onnx==1.21.0` in the CI, TensorRT-tool, and RT-DETR locks | Regenerate and recheck exported artifacts when the runtime or exporter changes. |
| ONNX Runtime | `onnxruntime>=1.17` in ONNX-backed extras | CPU `onnxruntime==1.24.2` in the CI and TensorRT-tool locks; task-specific RT-DETR locks use `onnxruntime==1.24.3` or `onnxruntime-gpu==1.24.4` | The floor is packaging compatibility. Backend evidence must record the runtime actually used. |

The exact lock files, rather than this summary, remain the machine-consumed
source for each test environment.

## ONNX opset boundary

`tools/export_trt.py` defaults to ONNX opset `18`. The current TensorRT/YOLO
examples and GPU smoke workflow explicitly pin opset `17`; that is a recorded
protocol choice, not the tool default. Keep the selected opset in export
metadata and compare artifacts only under the same preprocessing, decode, and
opset protocol.

## TensorRT and CUDA qualification

YOLOZU does not declare one universally qualified TensorRT/CUDA pair. The
self-hosted GPU workflow is configured with
`nvcr.io/nvidia/tensorrt:24.08-py3`, but it runs only when the required runner
and credentials are available. A successful run must retain:

- the exact container image or installed package identity;
- `nvidia-smi` GPU, driver, and reported CUDA context;
- `trtexec --version` or the TensorRT Python package version;
- the generated engine metadata and parity/latency reports.

Without those run artifacts, describe TensorRT/CUDA support as
environment-qualified rather than attaching a static version claim.

See [`production_readiness.md`](production_readiness.md) for the production
readiness matrix and [`evaluation_protocol_template.md`](evaluation_protocol_template.md)
for the reusable evaluation protocol template.
