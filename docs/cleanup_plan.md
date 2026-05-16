# Cleanup Plan

Codex に継続して任せるための整理方針。

## 完了

- `src/FF/AFF/gakusyu/` の標準入口を `cbam_full_from_images.py` に集約。
- 同ディレクトリの旧 Notebook を `archive/notebooks/FF_AFF_gakusyu/` に退避。
- `src/not_kenkyu/`, `src/tekito/`, `hukkyu/`, `test_hozon/` を削除せず `archive/` に退避。
- `src/deeplerning_model/` の旧 Notebook を `archive/legacy_models/deeplerning_model_notebooks/` に退避。
- データ、認証情報、学習結果は削除せず、Git 管理外のまま保持。

## 次にやる

1. `src/deeplerning_model/` に残っている 8GB 程度の重み・CSV・画像を、残す結果と削除可能な結果に分ける。
2. `YOLO/` 配下の実験ディレクトリを、現役・旧実験・外部変換ツールに分ける。
3. 最新実験は `configs/` + `scripts/` + `runs/` の形へ移す。
4. パラメータ探索は `configs/` を入力、`runs/` を出力にして自動化する。

## 削除してよい可能性が高いもの

まだ削除はしていない。Git 管理外のものは GitHub から復元できないため、削除前に確認する。

- `__pycache__/`
- `.ipynb_checkpoints/`
- 明らかな一時出力
- 再生成可能な古い評価出力
- `src/deeplerning_model/` に残る古い `.pth`, `.csv`, `.png`

## 削除しないもの

- `data/`
- `data_org/`
- `kaggle/KAGGLE_CONFIG_DIR/`
- `.nested_git_backups/`
- 学習済み重みや実験結果のうち、再生成条件が不明なもの

## 現在保留している大きい領域

- `src/deeplerning_model/`: 約 8GB。Notebook は archive 済み。重み・評価画像・CSV は削除判断待ち。
