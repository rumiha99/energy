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

## 推奨ディレクトリ

```text
configs/      実験設定
scripts/      コマンド実行する入口
runs/         実験結果、ログ、metrics。Git 管理しない
models/       学習済み重み。Git 管理しない
artifacts/    中間生成物。Git 管理しない
src/          既存研究コード
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
