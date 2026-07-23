# PyInstaller / PyArmor packaging

This repo’s installed CLI entrypoint is `yolozu` (`yolozu.cli:main`).

Packaging deep-learning stacks into a single executable is possible, but the result can be large (especially with `torch`,
`onnxruntime`, and CUDA/TensorRT libraries). This document provides a **supported path** for bundling the CLI without
depending on repo-only scripts under `tools/`.

## 1) Minimal (CPU-only) binary

Use this when you only need lightweight commands (doctor/validate/resources/eval with precomputed predictions).

```bash
python3 -m pip install -U pip
python3 -m pip install yolozu pyinstaller

pyinstaller -y -F -n yolozu \
  deploy/pyinstaller/yolozu_entrypoint.py \
  --additional-hooks-dir deploy/pyinstaller \
  --collect-data yolozu.data

./dist/yolozu --help
./dist/yolozu doctor --output -
```

## 2) Training-enabled binary (`yolozu train/test`)

This bundles the RT-DETR pose reference trainer. Install the extra deps first:

```bash
python3 -m pip install -U pip
python3 -m pip install 'yolozu[train]' pyinstaller
```

Build:

```bash
pyinstaller -y -F -n yolozu \
  deploy/pyinstaller/yolozu_entrypoint.py \
  --additional-hooks-dir deploy/pyinstaller \
  --collect-data yolozu.data \
  --collect-data rtdetr_pose
```

Quick smoke:

```bash
./dist/yolozu --help
./dist/yolozu test configs/examples/test_setting.yaml --max-images 2
```

Notes:
- `deploy/pyinstaller/hook-yolozu.py` registers modules loaded through the legacy lazy-alias map. Keep `--additional-hooks-dir deploy/pyinstaller` in every supported build.
- `rtdetr_pose` uses `importlib.resources` to load builtin JSON configs. `--collect-data rtdetr_pose` is required.
- ONNX export/parity during training additionally needs `onnx` / `onnxruntime` to be installed in the build env.

## 3) PyArmor (obfuscation) + PyInstaller

PyArmor can be layered on top of the entrypoint before PyInstaller bundling.
Keep the entrypoint stable (`deploy/pyinstaller/yolozu_entrypoint.py`) and ensure that data files are still collected
(`--collect-data yolozu.data --collect-data rtdetr_pose`) and the YOLOZU hook directory is enabled
(`--additional-hooks-dir deploy/pyinstaller`).

Recommended approach:
1) Obfuscate your application code (per your org’s PyArmor policy).
2) Run PyInstaller against the obfuscated entrypoint.
3) Validate `yolozu resources list` and `yolozu train/test` in the resulting `dist/` binary.

## Troubleshooting

- Missing packaged JSON/config resources:
  - ensure `--collect-data yolozu.data` and (if training) `--collect-data rtdetr_pose`
- Missing modules loaded through a legacy `yolozu.*` alias:
  - ensure `--additional-hooks-dir deploy/pyinstaller` is present
- Optional deps not present in the build environment:
  - install the extras you need *before* running PyInstaller, e.g. `pip install 'yolozu[full]'`

## CI verification

- GitHub Actions runs a PyInstaller smoke build on every full CI run with the YOLOZU hook directory and packaged data enabled.
- PyArmor integration is optional (it may require a license). To enable the CI smoke, set a repository Actions variable:
  - `PYARMOR_SMOKE=1`
