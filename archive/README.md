# archive

このディレクトリは、現在の標準入口から外した旧実験コードを残す場所です。

方針:

- 実行の標準入口は `scripts/`, `configs/`, `src/` の現役ファイルに置く。
- Notebook や比較用コードは、再確認できるように削除ではなくここへ退避する。
- ここにあるファイルは原則として新規実装の土台にしない。必要な処理は Python スクリプトへ移して使う。

## notebooks/FF_AFF_gakusyu

`src/FF/AFF/gakusyu/` にあった旧 Notebook 群です。

現在の標準入口:

```text
src/FF/AFF/gakusyu/cbam_full_from_images.py
```

## legacy_models/deeplerning_model_notebooks

`src/deeplerning_model/` にあった旧モデル比較 Notebook 群です。

重み、CSV、画像などの重い生成物は `src/deeplerning_model/` に残している。必要性が確認できるまでは削除しない。

## misc

一時コード、研究外メモ、単発確認コードを退避する場所です。

- `not_kenkyu/`
- `tekito/`

## recovered

復元された Notebook や復旧作業由来のファイルを退避する場所です。

- `hukkyu/`

## samples

小さな確認用サンプル画像などを退避する場所です。

- `test_hozon/`
