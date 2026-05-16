# Directory Inventory

最終更新: 2026-05-17

`/home/hiromu/energy` の整理用メモ。削除判断を急がず、まず「現役」「外部依存」「データ」「旧実験」に分ける。

## 現役

| Path | 役割 | 方針 |
| --- | --- | --- |
| `configs/` | 実験設定 | 維持。パラメータ調整はここへ集約する。 |
| `scripts/` | コマンド入口 | 維持。Notebook ではなくここから実行できる形へ増やす。 |
| `src/FF/AFF/gakusyu/cbam_full_from_images.py` | CBAM/ConvNeXt 完全版 | 最新入口として維持。画像から特徴量を再計算する。 |
| `docker/energy-yolo12/` | 現在使う Docker 環境 | 維持。 |

## データ・生成物

| Path | 役割 | 方針 |
| --- | --- | --- |
| `data/` | 研究データ | Git 管理外。削除しない。 |
| `data_org/` | 元データ | Git 管理外。削除しない。 |
| `runs/`, `models/`, `artifacts/` | 実験結果・重み・中間生成物 | Git 管理外。必要に応じて再生成する。 |
| `.history/` | エディタ/復元系の履歴と思われる | Git 管理外。容量は小さいが、不要確認後に削除候補。 |
| `.nested_git_backups/` | 旧サブリポジトリの `.git` 退避 | 復元用。すぐには削除しない。 |
| `kaggle/KAGGLE_CONFIG_DIR/` | Kaggle 認証情報 | Git 管理外。移動・削除しない。 |

## 外部コード・ツール

| Path | 役割 | 方針 |
| --- | --- | --- |
| `sam2/` | SAM2 関連 | 依存元として扱う。変更は最小限。 |
| `segment-anything/` | Segment Anything 関連 | 依存元として扱う。変更は最小限。 |
| `yolov12/` | YOLOv12 関連 | 依存元として扱う。変更は最小限。 |
| `coco-annotator/` | アノテーション/変換ツール | 必要性確認後に維持または外部参照化。 |
| `YOLO/` | YOLO 実験・変換コード | 実験履歴が多いため、段階的に整理する。 |

## 旧実験・比較候補

| Path | 役割 | 方針 |
| --- | --- | --- |
| `archive/notebooks/FF_AFF_gakusyu/` | `gakusyu` 直下から退避した旧 Notebook | 比較・復元用に保持。 |
| `archive/recovered/hukkyu/` | 復元 Notebook 群 | 比較・復元用に保持。 |
| `archive/samples/test_hozon/` | 一時保存サンプル画像 | 小さいため保持。 |
| `archive/misc/not_kenkyu/` | 研究外メモ | 比較・復元用に保持。 |
| `archive/misc/tekito/` | 一時コード | 比較・復元用に保持。 |
| `archive/legacy_models/deeplerning_model_notebooks/` | 旧モデル比較 Notebook | 比較・復元用に保持。 |
| `src/deeplerning_model/` | 旧モデル比較の重み・結果画像・CSV | 約 8GB。削除判断待ち。 |
