# YOLOZU (萬) — 日本語README

English README: [`README.md`](README.md)

[![PyPI](https://img.shields.io/pypi/v/yolozu?logo=pypi&logoColor=white)](https://pypi.org/project/yolozu/)
[![Zenodo (software DOI)](https://zenodo.org/badge/DOI/10.5281/zenodo.18744756.svg)](https://doi.org/10.5281/zenodo.18744756)
[![Zenodo (manual DOI)](https://zenodo.org/badge/DOI/10.5281/zenodo.18744926.svg)](https://doi.org/10.5281/zenodo.18744926)
[![CI (required)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml/badge.svg)](https://github.com/ToppyMicroServices/YOLOZU/actions/workflows/ci.yml)

視覚モデル評価のためのフレームワーク非依存ツールキット YOLOZU は、
ドメインシフト下における継続学習およびテスト時適応（TTT: test-time adaptation/training）を
再現可能に扱うことを目的として設計している。

特徴:

1. 破滅的忘却の緩和手法の採用
自己蒸留、リプレイ、パラメータ効率更新（PEFT）といった学習による破壊的忘却の緩和を可能にする．(忘却の完全な解消を保証するものではない)
TTT(Test time training)による推論時の重み調整による継続的な学習を通じたドメインシフト対策を備えている．

2. 予測をインタフェース契約として扱う
  推論結果を共通フォーマット predictions.json（＋export_settings）として保存し、
  モデル実装や推論基盤に依存しない評価を行う。
  これにより、継続学習やテスト時学習の結果を、フレームワークや実行環境を跨いで比較・再実行・CI検証できる。

3. タスク横断の評価に対応
  物体検出、セグメンテーション、キーポイント推定、単眼深度推定、6DoF姿勢推定の評価ワークフローをサポートする。
  学習実装は必須ではなく、評価系と分離して運用できる。

4. 実運用を想定した成果物管理
  各実験はバージョン付きアーティファクトを出力し、
  CI 上での差分比較や回帰検知を前提とした運用を可能にする。

5. 高度な最適化・評価手法を組み込み可能
  Hessian を用いた損失最適化や、不均衡データに対する FRACtal 系手法など、
  精度安定性や評価の信頼性を高めるための多様な手法を取り込める設計としている。


推論バックエンド（PyTorch / ONNXRuntime / TensorRT / ExecuTorch / C++ / Rust など）は自由に選び、
**同一の `predictions.json` interface contract** に落として評価・比較できることを最重視します。

対象:
- リアルタイム単眼 RGB **検出**
- 単眼 **depth + 6DoF pose**（RT-DETRベースの最小学習スキャフォールド）
- **セマンティックセグ**（データ準備 + mIoU評価）
- **インスタンスセグ**（PNG mask interface contract + mask mAP評価）

推奨デプロイ（標準パス）: **PyTorch → ONNX → TensorRT**

---

## Quickstart（コピペ1行 / repo checkout）

```bash
bash scripts/smoke.sh
```

この 1 行で以下を順に実行します（ネットワーク不要）:
- `yolozu doctor --output -`
- `yolozu validate dataset data/smoke`
- `yolozu validate predictions data/smoke/predictions/predictions_dummy.json --strict`
- `yolozu eval-coco --dataset data/smoke --split val \
  --predictions data/smoke/predictions/predictions_dummy.json --dry-run`

出力: `reports/smoke_coco_eval_dry_run.json`

pip だけで始める場合:

```bash
python3 -m pip install yolozu
yolozu doctor --output -
```

追加機能（必要なものだけ）:

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos（CPU可）
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[train]'   # 学習スキャフォールド（torch+onnxrt等）
python3 -m pip install 'yolozu[full]'
```

ドキュメント入口: [`docs/README.md`](docs/README.md)

学習系ドキュメント（継続学習 / TTT / distillation / long-tail recipe の PyTorch plugin 選択肢）: [`docs/learning_features.md`](docs/learning_features.md)

---

## 何が“売り”か（設計の中心）

- **Bring-your-own inference + interface-contract-first evaluation**  
  推論はどこで回してもよく、評価は `predictions.json` に統一して **公平に比較**できます。
- **再現性/運用性（Run interface contract / Run Contract）**  
  `yolozu train` の run interface contract で、成果物の置き場・run_meta・resume・export/parity を固定（`docs/run_contract.md`）。
- **Continual learning（反忘却: self-distillation + replay + LoRA）**  
  タスク/ドメイン列の継続微調整と、忘却の評価/抑制のための runner と成果物を提供（`docs/continual_learning.md`）。
- **Safe TTT（test-time training）**  
  Tent / MIM / CoTTA / EATA / SAR のプリセット・ガード・リセットポリシーを用意（`docs/ttt_protocol.md`）。
- **Prediction distillation（準・学習: offline）**  
  teacher/student の `predictions.json` をブレンドしてアブレーションを高速化（`docs/distillation.md`）。
- **Hessian-based refinement（準・学習: post-inference）**  
  `predictions.json` に対する per-detection の局所 refinement（engine外の後処理; `docs/hessian_solver.md`）。

---

## CLIの使い分け（pip vs ソースチェックアウト）

### モジュールパスについて（重要）
- canonical な Python モジュールはカテゴリ別パッケージ配下（`yolozu/core`, `yolozu/datasets`, `yolozu/eval`, `yolozu/inference`, `yolozu/predictions`, `yolozu/training`, `yolozu/geometry`）に配置されています。
- 旧 import（例: `from yolozu.dataset import build_manifest`）は `yolozu/__init__.py` の package-level alias により互換維持されます。

### pip: `yolozu`（インストール安全・CPU中心）
- `yolozu doctor`（環境診断）
- `yolozu validate dataset|predictions|instance-seg`（成果物検証）
- `yolozu eval-coco` / `yolozu eval-instance-seg`（評価）
- `yolozu onnxrt export`（ONNXRuntime推論→predictions出力、要 `yolozu[onnxrt]`）
- `yolozu onnxrt quantize`（ONNXRuntime dynamic quantize、要 `yolozu[onnxrt]`）
- `yolozu train`（RT-DETR pose 学習、要 `yolozu[train]`）
- `yolozu test`（シナリオスイート実行）

### repo: `python3 tools/yolozu.py`（研究/評価の“全部盛り”）
- `export --backend {torch,onnxrt,trt}`（統一引数 + キャッシュ + runメタ）
- TTA/TTT など、重いワークフローをまとめて扱うためのツール群が `tools/` にあります。

---

## 予測JSON（predictions interface contract）

評価の中心は `predictions.json` です
（スキーマ: [`schemas/predictions.schema.json`](schemas/predictions.schema.json)
/ 解説: [`docs/predictions_schema.md`](docs/predictions_schema.md)）。

推論は PyTorch / ONNXRuntime / TensorRT / C++ など好きな環境で実行し、結果だけを共通形式 `predictions.json` に保存します。
YOLOZU はその JSON を同じ評価器で採点するので、バックエンド差を「同一条件」で比較でき、再現も容易です。

- どのバックエンドでも **同じスキーマ**で出力
- 変換・評価・差分（parity）を統一

---

## Dataset 形式（YOLO + 任意メタデータ）

基本:
- 画像: `images/<split>/*.(jpg|png|...)`
- ラベル: `labels/<split>/*.txt`（`class cx cy w h` 正規化）

任意メタデータ（JSON）: `labels/<split>/<image>.json`
- Mask/Seg: `mask_path` / `mask` / `M`
- Depth: `depth_path` / `depth` / `D_obj`
- Pose: `R_gt` / `t_gt`（または `pose`）
- Intrinsics: `K_gt` / `intrinsics`

検証:
```bash
yolozu validate dataset data/smoke --strict
```

### 互換（YOLOv8 / YOLO11 / YOLOX）

- Ultralytics YOLOv8 / YOLO11:
  - `images/train` + `labels/train`（および `images/val` + `labels/val`）ならそのままOK
  - Ultralytics の `data.yaml` も `--dataset` に渡せます
    （`path:` + `train:`/`val:` が `images/<split>` を指す想定）
- YOLOX:
  - COCO JSON（`instances_*.json`）が多いので、
    `tools/prepare_coco_yolo.py` で YOLO形式へ一度変換するのが最短です

### Keypoints 形式サポート（明示）

- 直接対応: `auto`, `yolo_pose`, `coco`, `cvat_xml`
- 直接未対応（変換してから利用）: `detectron2_dataset_dict`, `labelme_keypoints`
- 形式一覧（CLI）:

```bash
python3 tools/yolozu.py prepare-keypoints-dataset --list-formats --source . --out .
```

- 最小CVAT XMLスモークテスト:

```bash
python3 -m pytest -q tests/test_prepare_keypoints_dataset_cvat_xml.py
```

詳細な復旧手順: [`docs/cvat_keypoints_recovery.md`](docs/cvat_keypoints_recovery.md)

---

## TTA / TTT（Test-Time Adaptation / Training）

- TTA: 予測の後処理で軽量に揺らす（`--tta`）
- TTT: **推論前**にモデルパラメータを少し更新（Tent / MIM、torch backendのみ）

TTTは repo 側のエクスポータで使うのが基本です（`docs/ttt_protocol.md`）:

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset data/smoke \
  --checkpoint runs/smoke/checkpoints/best.pt \
  --device cuda \
  --ttt --ttt-preset safe --ttt-reset sample \
  --ttt-log-out reports/ttt_log_safe.json \
  --output reports/predictions_ttt_safe.json
```

注意:
- TTT は torch backend 限定です（ONNXRuntime/TensorRT は TTA か precomputed predictions を推奨）

---

## Training scaffold（RT-DETR pose）+ Run interface contract（本番級の再現性）

実装: `rtdetr_pose/rtdetr_pose/train_minimal.py`（ラッパ: `rtdetr_pose/tools/train_minimal.py`）

### 最短（ソースチェックアウト）
```bash
python3 -m pip install -r requirements-test.txt
# 任意: CI推奨tier（固定依存）をローカル再現
python3 -m pip install -r requirements-ci.lock
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/smoke \
  --config rtdetr_pose/configs/base.json \
  --max-steps 50 \
  --run-dir runs/smoke
```

### 反復運用（Run Contract 推奨）

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01

# 完全resume（model/optim/sched/AMP scaler/EMA/progress + RNG）
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume

# 配線スモーク（最初のoptimizer stepで止め、保存/export/parityまで通す）
yolozu train configs/examples/train_contract.yaml --run-id exp01 --dry-run
```

run interface contract で固定された成果物（固定パス）:
- `runs/<run_id>/checkpoints/{last,best}.pt`
- `runs/<run_id>/reports/{train_metrics.jsonl,val_metrics.jsonl,config_resolved.yaml,run_meta.json,onnx_parity.json}`
- `runs/<run_id>/exports/model.onnx`（+ meta）

実装済みの“壊れない”学習ループ要件:
- Resume（完全復帰）
- NaN/Inf guard（skip + LR decay + stop）
- Grad clip（推奨）
- AMP / EMA / DDP（torchrun）
- Validation cadence（epoch/step）+ early stop

拡張（任意）:
- フォトメトリックAug
  （`--hsv-*`, `--gray-prob`, `--gaussian-noise-*`, `--blur-*`）  
  ※実画像を使う場合は `--real-images` を併用
  （スキャフォールドはデフォルトで合成画像）。
- 推論加速オプション: `--infer-batch-size`, `--torch-compile*`, `--torch-amp`, `--torch-channels-last`, `--torch-inference-mode`

### Depthモード（RT-DETR pose 学習スキャフォールド）

`rtdetr_pose/tools/train_minimal.py` では、backbone交換境界（`[P3,P4,P5]`）を
維持したまま深度を段階的に有効化できます。

- `--depth-mode none`（既定）: 深度を使わない互換パス
- `--depth-mode sidecar`: `depth_path` / `depth` のサイドカー深度を読み込み、`depth_valid` を付与
- `--depth-mode fuse_mid`: サイドカー深度を projector 後に軽量融合
  （backbone外）、`--depth-dropout` で modality dropout 可能

安全動作:

- `--depth-unit` は `unspecified|relative|metric`（既定: `unspecified`）
- 絶対深度コスト（`cost_z` / `cost_t`）は `metric` のときのみ有効化
- `--depth-scale` でサイドカー深度のスケール補正を適用

### 不均衡対策 / backbone override / strict data 検証

`train_minimal.py` では次を追加サポート:

- クラス不均衡対策: `--imbalance-strategy class_balanced`（DDPでも利用可能）
  （`--imbalance-gamma`, `--imbalance-min-weight`, `--imbalance-max-weight`, `--imbalance-aggregate`）
- backbone 明示上書き: `--backbone-name`, `--backbone-norm`, `--backbone-args`
- 実データ厳格検証: `--strict-task-data`（bbox/keypoints/depth/poseの教師情報不足を即時エラー）

### 実画像 few-shot の多タスク finetune デモ

```bash
# 事前にライセンスを確認した上で実画像をDL
python3 scripts/download_coco_instances_tiny.py \
  --out-root data/coco --split val2017 --num-images 8 --seed 0 --force

# 実画像データセットを準備（COCO画像 + annotation由来 sidecar）
python3 tools/prepare_real_multitask_fewshot.py \
  --instances-json data/coco/annotations/instances_val2017.json \
  --images-dir data/coco/images/val2017 \
  --out data/real_multitask_fewshot \
  --train-images 6 --val-images 2 \
  --strict-provenance --force

# bbox -> segmentation -> keypoints -> depth -> pose6d を段階実行
python3 tools/run_real_multitask_finetune_demo.py \
  --dataset-root data/real_multitask_fewshot \
  --out reports/real_multitask_finetune_demo \
  --device cpu \
  --epochs 1 --max-steps 1 --batch-size 2 --image-size 96 \
  --strict-provenance --force
```

結果レポート:
`reports/real_multitask_finetune_demo/multitask_finetune_demo_report.json`
（`prepare_summary.json` に annotation由来ラベルの provenance も記録）

### Reference adapter回帰ゲート（実画像baseline）

`RTDETRPoseAdapter` を reference adapter として固定し、
`predict(records) -> entries` の interface contract を CI で回帰監視できます。

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --repro-policy relaxed \
  --runtime-lock requirements-ci.lock \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression.json
```

interface contractのみ（hard gate）:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --score-gate-mode off \
  --perf-gate-mode off \
  --runtime-lock requirements-ci.lock \
  --enforce-runtime-lock \
  --enforce-weights-hash \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_contract.json
```

Behaviorのみ（warn gate）:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --schema-gate-mode off \
  --consistency-gate-mode off \
  --score-gate-mode warn \
  --perf-gate-mode warn \
  --runtime-lock requirements-ci.lock \
  --enforce-runtime-lock \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --output reports/reference_adapter_regression_behavior.json
```

baseline更新（意図的変更時のみ）:

```bash
python3 tools/run_reference_adapter_regression.py \
  --dataset data/smoke \
  --split val \
  --max-images 2 \
  --runtime-lock requirements-ci.lock \
  --baseline baselines/reference_adapter/rtdetr_pose_smoke_val.json \
  --write-baseline \
  --output reports/reference_adapter_regression_baseline_write.json
```

Run Contract仕様: [`docs/run_contract.md`](docs/run_contract.md)

---

## ONNX 量子化（低コスト）

ONNXRuntime の dynamic quantize で int8-ish の ONNX を生成できます（CPU向け、校正データ不要）:

```bash
yolozu onnxrt quantize --onnx model.onnx --output model_int8.onnx --weight-type qint8
```

---

## 対称性（Symmetry）チェック

- 設定: `configs/runtime/symmetry.json`（ローダ: `yolozu.config.load_symmetry_map`）
- 実装: `yolozu/symmetry.py`（`none`, `Cn`/`C2`/`C4`, `Cinf`）
- テンプレ検証: `yolozu/template_verification.py`

---

## 実行ファイル化（PyInstaller / PyArmor）

手順: [`deploy/pyinstaller/README.md`](deploy/pyinstaller/README.md)

---

## 開発者向け（ローカル検証）

```bash
.venv/bin/ruff check .
.venv/bin/python -m unittest
```
