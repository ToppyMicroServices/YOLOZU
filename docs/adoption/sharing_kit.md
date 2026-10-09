# Sharing YOLOZU's first evaluation path

Use the Japanese walkthrough when introducing YOLOZU to Japanese readers and
the English walkthrough elsewhere. Both start with a released package and
locally generated inputs, then lead to a COCO evaluation report and an optional
release qualification pack.

- [Japanese walkthrough](../labeled_sample_ja.md)
- [English walkthrough](../labeled_sample.md)
- [Release qualification and CI](../release_qualification.md)
- [Real eager/TorchScript comparison](../case_studies/maskrcnn_eager_torchscript.md)

The copy below is prepared for publication. It is not a record of a social post,
an article submission, a community endorsement, or an external user trial.

## Short Japanese post

> 既存の物体検出モデルの予測結果を、同じ条件で検証・評価するYOLOZU。
> 無料で使え、コードはApache-2.0です。
> モデルやdatasetを用意せず、画像・ラベルの生成から最初のCOCO評価reportまで試せる手順を公開しました。
> 合成サンプルのスコアは動作確認用です。
> [日本語の手順](https://github.com/ToppyMicroServices/YOLOZU/blob/main/docs/labeled_sample_ja.md)

## Short English post

> Compare existing object-detection predictions with YOLOZU, a free evaluation
> toolkit with Apache-2.0 code. Keep your inference stack and use the same labels
> and evaluation settings. Try the first report with locally generated images;
> no model download is needed. The synthetic scores check the workflow.
> [Walkthrough](https://github.com/ToppyMicroServices/YOLOZU/blob/main/docs/labeled_sample.md)

## Japanese article introduction

### 既存の予測結果から評価reportを作る：YOLOZUをモデルなしで試す

物体検出モデルを比較するとき、推論だけでなく、正解ラベル、class mapping、
評価条件も揃える必要があります。YOLOZUは、既存の予測結果を
predictions interface contractで検証し、正解ラベルと照合してJSONの評価reportを作ります。
モデルや推論環境を置き換える必要はありません。

初回は、自分のdatasetを準備せずに試せます。公開済みの`yolozu[coco]==4.11.0`で
合成画像8枚とラベルを生成し、strict validationを通してCOCOevalを実行します。
reportの見方まで、[日本語の手順](https://github.com/ToppyMicroServices/YOLOZU/blob/main/docs/labeled_sample_ja.md)
にまとめました。予測を正解ラベルから作るため、このスコアは動作確認用です。
実モデルの精度を示すものではありません。

同じサンプルから、閾値判定と検証可能なrelease qualification packも作れます。
自分の予測結果に移行した後は、[YAML specとGitHub Action](https://github.com/ToppyMicroServices/YOLOZU/blob/main/docs/release_qualification.md)
で評価をCIに組み込めます。YOLOZUはToppyMicroServices OÜが開発する無料の商用プロダクトで、
リポジトリのコードはApache-2.0です。

## Use the copy for a specific audience

For engineers comparing existing model outputs, lead with the evaluation
walkthrough and the checked framework export routes. For engineers reviewing
model releases in CI, lead with the release qualification pack and Action.
Use the documented real-output comparison when a reader asks for evidence
beyond the synthetic first run. Its results apply only to its stated protocol.

Publish one complete tutorial before linking it from shorter posts. Use the
walkthrough URL as the main call to action. Describe the actual problem solved;
do not present candidate inference, Research methods, popularity, or accuracy
gains as established product results.

If a reader asks for help, link the [support routes](../support.md). Participation
and aggregate use are voluntary. Keep private artifacts out of public threads.
For a consented observation, use the existing
[observation kit](design_partner_observation_kit.md), including its correction
and deletion window. Record first evaluation and repeat use only when there is
consented external evidence; the maintainer check below does not supply it.

## Maintainer verification — 2026-10-08

The published PyPI package `yolozu[coco]==4.11.0` was installed in a fresh,
non-editable virtual environment outside the source checkout. `pip check`
passed. The check used macOS arm64 and Python 3.14.6 with NumPy 2.5.3,
Pillow 12.3.0, PyYAML 6.0.3, and pycocotools 2.0.11.

| Check | Observed result |
|---|---|
| Seed `0` sample generation | 8 images; labeled preview visually inspected |
| Strict train/val dataset and predictions validation | All commands exited `0` |
| COCOeval on the val split | `dry_run: false`; 4 images; 6 detections; `map50_95 = 1.0`, `map50 = 1.0` |
| Release gate with fixture threshold `0.99` | `decision: pass` |
| Pack verification | `ok: true`, no errors |

This is a local maintainer check on synthetic inputs. It establishes neither
external adoption nor real-model performance. Windows instructions follow the
existing installation guide; this check did not run Windows.

The [weekly snapshot](2026-10-08-baseline.md) separately records public activity
signals. Future reviews should inspect consented completion and blocker
categories under the [measurement policy](README.md), rather than interpreting
traffic or downloads as users.
