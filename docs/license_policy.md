# License Policy (Apache-2.0 / No Copyleft Code)

This repository is intended to be **Apache-2.0** code only.

## Repository policy: permissive code, no built-in telemetry

- Shipped repository code is intended to remain under **Apache-2.0**.
- This repository does **not** document or ship a built-in relicensing path for the repository code.
- Shipped YOLOZU tooling does **not** include built-in telemetry, usage analytics, or phone-home data collection.
- Quality control is handled through explicit checks, manifests, CI gates, provenance reports, and documented workflows instead of silent data collection.
- This remains a **best-effort engineering policy**, not a legal guarantee for third-party artifacts or deployment environments.

## Rules

- Do **not** vendor or depend on GPL/AGPL code in this repository.
- To compare against external baselines (e.g., YOLO26), run them in a separate environment and **only import predictions JSON** into this repo for evaluation.
- Keep datasets and model weights out of git.
- The packaged model registry for `yolozu fetch` is curated to Apache-friendly licenses only. Use a custom registry + `--allow-non-apache` only when you explicitly accept the risk and boundary the baseline environment.

## External training boundary

When YOLOZU needs a YOLO-style training lane outside the in-repo RT-DETR reference trainer:

- prefer the **YOLOX external lane** first because YOLOX is Apache-2.0-friendly
- keep the training loop in the external repo/runtime
- let YOLOZU own dataset resolution, reports, and the predictions interface contract
- treat the **Ultralytics bridge** as optional and review it under its own license terms before commercial use
- treat the **HF DETR bridge** as optional and review its runtime/dependency boundary separately

In practice this means:

- `python3 -m yolozu train --external-backend yolox ...` is the recommended external YOLO-style path
- `python3 -m yolozu train --external-backend ultralytics ...` is available only as an optional bridge
- `python3 -m yolozu train --external-backend hf-detr ...` is available only as an optional bridge
- the repository must not vendor Ultralytics or other copyleft implementation code

Optional bridge reports include `runtime_license_boundary` metadata. That field records the
external runtime name, that the runtime is not bundled with YOLOZU, that it is not a default
install dependency, and that runtime/model license review remains required before deployment.

## Company release policy (naming + provenance)

- Use consistent product/repo naming in release artifacts: `YOLOZU` (`ToppyMicroServices/YOLOZU`).
- Keep `LICENSE`, `NOTICE`, `COPYRIGHT`, and `SECURITY.md` at repository root and included in distribution artifacts.
- Release notes should state the contract boundary (`docs/release_1_0_stability.md`) and any non-contract experimental areas.

## COCO / coco128

The `coco128` helper dataset is fetched from **official COCO hosting** and converted to YOLO-format labels locally.
Datasets have their own licenses; using them does not change the license of this repository.

## Quick Checks

- Run `python3 tools/check_license_policy.py` before pushing.
- The unit test `python3 -m unittest tests/test_license_policy.py` enforces basic guardrails (e.g., no branded external-runtime fetch URL, presence of `LICENSE`).
- CI runs `tools/check_license_policy.py`, `ruff`, and `python -m unittest` on push/PR.

## Commercial-use due diligence (best-effort, not legal advice)

This repo keeps its **code** Apache-2.0-only, but commercial usage risk can still come from:
- **Dependencies** (Python packages, CUDA/TensorRT/system libs, Docker base images)
- **Datasets** (image licenses vary; some datasets are research-only)
- **Model weights** (separate licenses; keep out of git)
- **Deployment/runtime integrations** (telemetry defaults, cloud logging, or vendor-side reporting in third-party systems)

To help audit Python dependencies, generate a license report from the *current environment*:

```bash
python3 tools/report_dependency_licenses.py --output reports/dependency_licenses.json
```

To use it as a guardrail (fail if copyleft-suspect licenses are detected):

```bash
python3 tools/report_dependency_licenses.py --fail-on-copyleft
```

Notes:
- This is **best-effort**: it only inspects installed Python distributions' metadata.
- It does **not** audit CUDA/TensorRT/system libraries, datasets, or model weights.
- For real commercial deployment, you should also do a formal review with legal counsel.

## Third-party dependencies policy (release-time)

- Python packages: tracked via lock/report tooling (`tools/report_dependency_licenses.py`).
- System/runtime dependencies (CUDA, cuDNN, TensorRT, OpenCV builds): deployment-team responsibility; validate in target runtime and keep SBOM/license evidence outside this repository when required by policy.
- Datasets/weights are separate artifacts with independent license terms; they are never implicitly covered by this repo's Apache-2.0 license.
- Telemetry/logging controls in third-party platforms, hosted runtimes, or cloud services are outside the YOLOZU repository boundary; review and disable them in the target environment if your policy requires that.
