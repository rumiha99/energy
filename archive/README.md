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

