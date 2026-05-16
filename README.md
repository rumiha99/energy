# energy

研究用の画像処理・特徴量抽出・学習実験ワークスペース。

このリポジトリは Codex に継続的に作業を任せやすくするため、次の方針で整理する。

- 実行入口は `scripts/` に置く
- 実験で変える値は `configs/` に置く
- 実験結果は `runs/` に出す
- 学習済み重み、画像、動画、CSV、pickle、ZIP などは GitHub に入れない
- Notebook は試行錯誤用として残し、再利用する処理は Python スクリプトへ移す

## Docker

```bash
cd /home/hiromu/energy/docker/energy-yolo12
docker compose up -d --build
docker exec -it yolo12 bash
```

コンテナ内の作業ディレクトリ:

```text
/home/hiromu/energy
```

## Git

正式な GitHub リポジトリ:

```text
https://github.com/rumiha99/energy
```

通常の作業:

```bash
cd /home/hiromu/energy
git status
git add <変更したファイル>
git commit -m "message"
git push
```

## 現在の標準入口

### 環境確認

```bash
python scripts/check_environment.py
```

### YOLO 前処理

設定:

```text
configs/preprocess_yolo.json
```

設定だけ確認:

```bash
python scripts/run_preprocess_yolo.py --dry-run
```

実行:

```bash
python scripts/run_preprocess_yolo.py --config configs/preprocess_yolo.json
```

実行結果のメタデータは `runs/preprocess_yolo_YYYYMMDD_HHMMSS/` に保存する。

### CBAM/ConvNeXt 完全版

元になった Notebook 実装は archive に退避済み。

```text
archive/notebooks/FF_AFF_gakusyu/CBAM_check.ipynb
```

その「CNNで使用するデータ直し」以降を、画像から特徴量まで再計算する形でまとめた完全版:

```text
src/FF/AFF/gakusyu/cbam_full_from_images.py
```

設定:

```text
configs/cbam_full_from_images.json
```

画像から特徴量を 1 件だけ計算して確認:

```bash
python src/FF/AFF/gakusyu/cbam_full_from_images.py --mode dry-run --limit 1
```

学習:

```bash
python src/FF/AFF/gakusyu/cbam_full_from_images.py --mode train
```

評価:

```bash
python src/FF/AFF/gakusyu/cbam_full_from_images.py --mode eval
```

特徴量だけ CSV に書き出す:

```bash
python src/FF/AFF/gakusyu/cbam_full_from_images.py --mode export-features
```

この完全版では、CSV は `filename`, `category`, `Label` による分割リストとして使い、
`h0`-`h7`, `size_count`, `R` は `mask/` と `combined/` の画像から再計算する。

## 推奨ディレクトリ

```text
configs/      実験設定
scripts/      コマンド実行する入口
runs/         実験結果、ログ、metrics。Git 管理しない
models/       学習済み重み。Git 管理しない
artifacts/    中間生成物。Git 管理しない
src/          既存研究コード
archive/      旧実験コード、旧 Notebook
docs/         整理方針、棚卸しメモ
docker/       Docker 環境
```

## Codex に依頼するときの例

```text
/home/hiromu/energy を対象に、scripts/run_preprocess_yolo.py と
configs/preprocess_yolo.json の形式に合わせて、特徴量抽出とSVM予測も
scripts/run_feature_predict.py と configs/feature_predict.json に分離して。
既存 Notebook は変更しないで。
```

```text
AREA_THRESHOLD, hida の n/T, SVM の C/gamma を Optuna で探索する
scripts/tune_optuna.py を作って。
各 trial の config と metrics を runs/ に保存して。
評価指標は macro F1 を最大化にして。
```

## GitHub に入れないもの

`.gitignore` で主に以下を除外している。

- `data/`, `data_org/`
- `runs/`, `models/`, `artifacts/`, `outputs/`
- `.pt`, `.pth`, `.pkl`, `.csv`
- 画像、動画、ZIP
- `.env`, Kaggle 認証情報
