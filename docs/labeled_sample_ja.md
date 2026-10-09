# モデルなしで最初の評価reportを作る

[English](labeled_sample.md)

YOLOZUを試すために、自分のモデルやdatasetを用意する必要はありません。
この手順では画像と正解ラベルを生成し、予測を検証してCOCOevalのreportを作ります。
その後、自分のモデルの予測結果にも同じ評価手順を使えます。

サンプルは320×240の合成画像8枚です。trainとvalに各4枚を配置し、
class IDは`0 = circle`、`1 = rectangle`とします。valの最後の1枚は空画像です。
モデル、GPU、画像や外部datasetのダウンロードは不要です。

## インストールして評価する

Python 3.10以上が必要です。macOS/Linuxでは、新しい作業ディレクトリで実行します。
以下は公開済みの4.11.0に固定した手順です。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'yolozu[coco]==4.11.0'
```

Windows PowerShellでは`py -m venv .venv`、
`.\.venv\Scripts\Activate.ps1`で仮想環境を作成・有効化し、
以降の`python3`を`python`に置き換えます。
環境設定で詰まった場合は[インストール手順](install.md)を参照してください。

```bash
python3 -m yolozu demo dataset --run-dir reports/labeled_sample --seed 0
python3 -m yolozu validate dataset reports/labeled_sample --split train --strict
python3 -m yolozu validate dataset reports/labeled_sample --split val --strict
python3 -m yolozu validate predictions reports/labeled_sample/predictions.json --strict
python3 -m yolozu eval-coco --dataset reports/labeled_sample --split val \
  --predictions reports/labeled_sample/predictions.json \
  --output reports/labeled_sample_coco.json
```

サンプル生成にはcore依存だけを使います。COCOevalには`coco` extraが必要です。
生成先が既に存在すると処理を拒否するため、再実行時は新しい`--run-dir`を選び、
後続のコマンドのパスも揃えてください。

## 結果を確認する

`reports/labeled_sample/labeled_preview.png`を開くと、全8枚の画像とラベルを確認できます。
元画像にはラベルを描き込んでいません。評価reportは
`reports/labeled_sample_coco.json`です。ターミナルでも確認できます。

```bash
python3 -m json.tool reports/labeled_sample_coco.json
```

変更していないseed `0`のサンプルでは、次の結果になります。

| フィールド | 期待値 |
|---|---:|
| `dry_run` | `false` |
| `counts.images` | 4 |
| `counts.detections` | 6 |
| `metrics.map50_95` | 約1.0 |
| `metrics.map50` | 約1.0 |

空のval画像も評価に含まれます。予測は正解ラベルから作った既知の値なので、
このスコアはデータ処理と評価の動作確認用です。実モデルの精度を示すものではありません。
`--dry-run`のreportはmetricsが`null`になり、この評価の完了には数えません。
`pycocotools`が見つからない場合は、有効な仮想環境に`coco` extraをインストールしてください。

## release gateも試す

4.11.0では、同じサンプルから閾値判定とJSONのevidence packを作れます。

```bash
python3 -m yolozu qualify-release create \
  --dataset reports/labeled_sample --split val \
  --predictions reports/labeled_sample/predictions.json \
  --min-map50-95 0.99 --output-dir reports/labeled_sample_qualification
python3 -m yolozu qualify-release verify reports/labeled_sample_qualification
```

サンプルを変更していなければ、作成結果は`decision: pass`、検証結果は`ok: true`です。
`0.99`はこのfixtureの確認値で、本番用の推奨閾値ではありません。
自分のモデルでは、評価前に用途に合う閾値と比較用baselineを決めてください。
YAML spec、回帰チェック、再利用可能なGitHub Actionは
[release qualificationの説明](release_qualification.md)にあります。

## 自分の予測結果に進む

自分の画像、YOLO形式の正解ラベル、wrapped `predictions.json`にパスを置き換えます。
予測boxは正規化した`cx, cy, w, h`、class IDは正解ラベルと一致する0始まりの値です。
同じdataset、split、class mapping、評価条件で比較してください。
[predictions interface contract](predictions_schema.md)と
[framework別のexport手順](byop_quickstarts.md)を参照できます。

サンプルを他の環境に移す場合は、`classes.json`と`data.yaml`を含むディレクトリ全体をコピーします。
元サンプルの`sample_manifest.json`は、その画像とラベルの生成条件・hashを記録しています。
変更したデータのprovenanceとして流用しないでください。

初回実行で詰まったら[First-run failure](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=first_run_failure.yml)、
評価の疑問は[Evaluation question](https://github.com/ToppyMicroServices/YOLOZU/issues/new?template=evaluation_question.yml)へ。
回答は任意で、集計への利用可否もフォームで選べます。
公開投稿にdataset、重み、非公開の予測、認証情報、個人情報を添付しないでください。
詳しい受付方針は[support](support.md)にあります。
