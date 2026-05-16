# Cleanup Plan

Codex に継続して任せるための整理方針。

## 完了

- `src/FF/AFF/gakusyu/` の標準入口を `cbam_full_from_images.py` に集約。
- 同ディレクトリの旧 Notebook を `archive/notebooks/FF_AFF_gakusyu/` に退避。
- データ、認証情報、学習結果は削除せず、Git 管理外のまま保持。

## 次にやる

1. `src/deeplerning_model/`, `src/not_kenkyu/`, `src/tekito/` の中身を確認し、必要なものだけ `archive/` へ退避する。
2. `YOLO/` 配下の実験ディレクトリを、現役・旧実験・外部変換ツールに分ける。
3. `hukkyu/` と `test_hozon/` が不要な復元物か確認する。
4. 最新実験は `configs/` + `scripts/` + `runs/` の形へ移す。
5. パラメータ探索は `configs/` を入力、`runs/` を出力にして自動化する。

## 削除してよい可能性が高いもの

まだ削除はしていない。Git 管理外のものは GitHub から復元できないため、削除前に確認する。

- `__pycache__/`
- `.ipynb_checkpoints/`
- 明らかな一時出力
- 再生成可能な古い評価出力

## 削除しないもの

- `data/`
- `data_org/`
- `kaggle/KAGGLE_CONFIG_DIR/`
- `.nested_git_backups/`
- 学習済み重みや実験結果のうち、再生成条件が不明なもの

