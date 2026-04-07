# YOLOZU (萬) — 日本語README

English: [`README.md`](README.md) | 中文: [`Readme_zh.md`](Readme_zh.md)

YOLOZU は、workflow を単一の training framework に lock-in したくないチーム向けの Apache-2.0 vision evaluation toolkit です。

Bring your own inference.
Export once.
Evaluate fairly.

YOLOZU が基準にするのは、安定した predictions interface contract です。
中身は wrapped `predictions.json` と、その中の protocol-pinned な `meta.export_settings` です。

## 1分デモ

```bash
python3 -m pip install -U yolozu
yolozu demo overview
```

出力: `demo_output/overview/<utc>/demo_overview_report.json`

```mermaid
flowchart LR
    A["Ultralytics"] --> D["wrapped predictions.json"]
    B["RT-DETR"] --> D
    C["Detectron2 / MMDetection / custom"] --> D
    D --> E["validate"]
    E --> F["evaluate"]
    F --> G["comparable report"]
```

[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://pypi.org/project/yolozu/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/build_and_test.yml)

## 最初に読む3本

- [`docs/README.md`](docs/README.md): docs 全体の入口と最短の使い方
- [`docs/predictions_schema.md`](docs/predictions_schema.md): predictions interface contract
- [`docs/install.md`](docs/install.md): install、`doctor`、環境確認

## Primary Focus

- 主戦場: 既存 predictions を framework / runtime をまたいで公平に評価すること
- 次のレイヤ: 同じ predictions interface contract へつなぐ export / train scaffold
- 発展レイヤ: continual learning、TTT、SynthGen、backend parity の research path

## Capability Maturity

- Stable: prediction validation/evaluation、wrapped `predictions.json`、repo smoke/demo path、install/doctor
- Experimental: backend parity、benchmark orchestration、SynthGen intake/handoff、macOS/MPS evaluation path
- Research: continual learning、self-distillation、TTT、Hessian refinement

## Production Readiness

- いま production-ready と言いやすいもの: prediction validation/evaluation と predictions interface contract
- 環境ごとの検証が必要なもの: backend parity、benchmark orchestration、SynthGen handoff、macOS/MPS path
- research-oriented なもの: continual learning、self-distillation、TTT、Hessian refinement
- 詳細: [`docs/production_readiness.md`](docs/production_readiness.md)

## 次に見る場所

- 既存 predictions を評価する: [`docs/external_inference.md`](docs/external_inference.md)
- train → export → eval を試す: [`docs/training_inference_export.md`](docs/training_inference_export.md)
- backend 比較や benchmark を見る: [`docs/backend_parity_matrix.md`](docs/backend_parity_matrix.md), [`docs/benchmark_mode.md`](docs/benchmark_mode.md)
- YOLOZU-synthgen 連携を準備する: [`docs/synthgen_repo_integration.md`](docs/synthgen_repo_integration.md)
- tool / manifest の参照先: [`docs/tools_index.md`](docs/tools_index.md), [`tools/manifest.json`](tools/manifest.json)

## すぐ分かること

- 既に predictions があり、framework をまたいで公平比較したいなら向いています。
- commercial / internal use で Apache-2.0-friendly な tooling が欲しい場合に向いています。
- one-click default の end-to-end training framework が欲しいなら別の道具の方が合う可能性があります。
- GPU は必須ではありません。最初の demo path は CPU-friendly です。
- YOLOZU の中で学習しなくても使えます。
- 主な artifact は wrapped `predictions.json` と `meta.export_settings` です。

## Demo と発展資料

- 30秒 demo: `yolozu demo overview`
- 実画像 showcase: [`docs/assets/readme_multitask_showcase.png`](docs/assets/readme_multitask_showcase.png)
- 学習系の全体像: [`docs/learning_features.md`](docs/learning_features.md)
- Distillation: [`docs/distillation.md`](docs/distillation.md)
- Continual learning: [`docs/continual_learning.md`](docs/continual_learning.md)
- Hessian refinement: [`docs/hessian_solver.md`](docs/hessian_solver.md)

## repo checkout で使う場合

```bash
python3 -m pip install -e .
bash scripts/smoke.sh
```

詳しくは次を見てください。

- docs index: [`docs/README.md`](docs/README.md)
- install 詳細: [`docs/install.md`](docs/install.md)
- manual source: [`manual/README.md`](manual/README.md)

## Support と License

- Support: [`docs/support.md`](docs/support.md)
- License policy: [`docs/license_policy.md`](docs/license_policy.md)
- Apache-2.0 license: [`LICENSE`](LICENSE)
- Latest release: [GitHub Releases](https://github.com/ToppyMicroServices/YOLOZU/releases)
- Zenodo software DOI: [10.5281/zenodo.18744756](https://doi.org/10.5281/zenodo.18744756)
- Zenodo manual DOI: [10.5281/zenodo.18744926](https://doi.org/10.5281/zenodo.18744926)
