# CVAT keypoints復旧手順（COCO / YOLO / CVAT XML）

この手順は、**データを失った状態から最短で学習再開できること**を目的にしています。
対象は **custom keypoints（person以外を含む）** です。

## 0) 先に結論（最短ルート）

- 最短: **CVATで YOLO Pose 形式で再エクスポート** → そのままYOLOZUへ。
- 次点: CVAT COCO keypoints を使う場合は `tools/prepare_coco_keypoints_yolozu.py` で YOLOZU形式へ変換。
- CVAT XML も `prepare-keypoints-dataset` で直接取り込み可能（bbox欠損時は可視キーポイントから推定）。

ワンコマンド化（推奨）:

```bash
python3 tools/prepare_keypoints_dataset.py --source <INPUT_PATH> --format auto --out <OUTPUT_DATASET_ROOT>
```

このラッパーは `YOLO Pose` と `COCO keypoints` を自動判定して dataset 化します。

## 対応/未対応形式（明示）

### 直接対応（そのまま入力可能）

- `auto`（`images/`+`labels/` / `annotations/` / `annotations.xml` から自動判定）
- `yolo_pose`（YOLO Pose レイアウト）
- `coco`（COCO keypoints JSON）
- `cvat_xml`（CVAT XML。`--source` はXMLファイルまたは `annotations.xml` を含むディレクトリ）

### 未対応（直接入力は不可、変換してから）

- `detectron2_dataset_dict`（Python dict直入力）
  - 変換先: COCO keypoints JSON
- `labelme_keypoints`
  - 変換先: COCO keypoints JSON

CLIで一覧確認:

```bash
python3 tools/prepare_keypoints_dataset.py --list-formats
```

または

```bash
python3 tools/prepare_keypoints_dataset.py --list-formats
```

---

## 1) CVAT export = YOLO Pose（推奨）

### 1-1. 期待ディレクトリ

```text
<dataset_root>/
  images/
    train2017/*.jpg
    val2017/*.jpg
  labels/
    train2017/*.txt
    val2017/*.txt
```

ラベル1行は次の形:

```text
class cx cy w h x1 y1 v1 x2 y2 v2 ...
```

### 1-2. 学習実行（例）

まず dataset 化（wrapper生成）:

```bash
python3 tools/prepare_keypoints_dataset.py \
  --source /path/to/cvat_yolo_pose_dataset \
  --format yolo_pose \
  --out /path/to/yolozu_kp_dataset
```

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root /path/to/yolozu_kp_dataset \
  --split train2017 \
  --real-images \
  --num-classes <N_CLASSES> \
  --num-keypoints <K> \
  --batch-size 2 \
  --epochs 1 \
  --run-dir runs/kp_recover_smoke
```

### 1-3. まず通すチェック

```bash
.venv/bin/python -m pytest -q tests/test_dataset_keypoints.py
```

---

## 2) CVAT export = COCO keypoints（custom category対応）

### 2-1. 変換

```bash
python3 tools/prepare_keypoints_dataset.py \
  --source /path/to/coco_root \
  --format coco \
  --out /path/to/yolozu_kp_dataset \
  --annotations annotations/person_keypoints_val2017.json \
  --images-dir val2017 \
  --out-split val2017 \
  --category-name <YOUR_CATEGORY_NAME> \
  --class-id 0 \
  --min-kps 1
```

`--category-id` でも指定可能です。

### 2-2. 生成物

- `labels/<split>/*.txt`（YOLO pose形式）
- `labels/<split>/classes.json`（`keypoint_names`, `skeleton` を含む）
- `dataset.json`

### 2-3. 学習

```bash
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root /path/to/yolozu_kp_dataset \
  --split val2017 \
  --real-images \
  --num-classes 1 \
  --num-keypoints <K> \
  --batch-size 2 \
  --epochs 1 \
  --run-dir runs/kp_coco_recover_smoke
```

---

## 3) CVAT export = CVAT XML

CVAT XML も直接取り込みできます。

```bash
python3 tools/prepare_keypoints_dataset.py \
  --source /path/to/cvat_export_or_annotations.xml \
  --format cvat_xml \
  --out /path/to/yolozu_kp_dataset \
  --class-id 0
```

画像ルートが自動推定と異なる場合は `--cvat-images-dir` を指定します。

```bash
python3 tools/prepare_keypoints_dataset.py \
  --source /path/to/annotations.xml \
  --format cvat_xml \
  --cvat-images-dir /path/to/images_root \
  --out /path/to/yolozu_kp_dataset
```

注記:
- group対応bboxがない場合、可視キーポイント外接矩形からbboxを自動生成します。
- 可能なら引き続き YOLO Pose 再エクスポートが最も安定です。

---

## 4) Detectron2でやっていたことの移植観点

- Detectron2由来の COCO keypoints JSON は、上記の COCO変換フローで移植可能。
- ただし YOLOZU学習は最終的に YOLO poseラベル（`class cx cy w h + keypoints`）で回すのが安定。
- `skeleton` / `keypoint_names` は `classes.json` または `dataset.json` から読み取られ、学習設定に利用可能。

---

## 5) 復旧時の最小ゲート

- `--num-keypoints` がデータ定義と一致している
- 最初の1epoch smoke で `loss_kp` が有限値で推移
- 推論後に `tools/eval_keypoints.py` で PCK/OKS が計算できる

```bash
python3 tools/eval_keypoints.py \
  --dataset /path/to/yolozu_kp_dataset \
  --split val2017 \
  --predictions /path/to/predictions.json \
  --output reports/keypoints_eval.json
```
