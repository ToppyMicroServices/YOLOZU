YOLOZU is an Apache-2.0, interface-contract-first evaluation harness for detection / segmentation / pose.

**Why it matters (3 lines):**
- Run inference anywhere (PyTorch / ONNXRuntime / TensorRT / OpenCV-DNN / C++ / Rust).
- Export only the results into a stable `predictions.json` **predictions interface contract** (+ pinned `export_settings`).
- Validate + score with the same evaluator → apples-to-apples backend comparisons and reproducible reports.

**Quickstart (3 commands):**
```bash
python3 -m pip install yolozu==1.0.0
yolozu doctor --output -
bash scripts/smoke.sh
```

**Artifacts**
- PyPI: https://pypi.org/project/yolozu/1.0.0/
- Zenodo (software): https://doi.org/10.5281/zenodo.18744757 (concept DOI: https://doi.org/10.5281/zenodo.18744756)
- Zenodo (manual PDF): https://doi.org/10.5281/zenodo.18744927 (concept DOI: https://doi.org/10.5281/zenodo.18744926)

**For experts (keywords):** Safe TTT (Tent/MIM), LoRA/norm-only scopes, Hessian diagnostics, continual learning hooks, run interface contract.

More copy/paste templates: https://github.com/ToppyMicroServices/YOLOZU/blob/main/docs/announcements/v1_0_0.md
