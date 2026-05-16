# Codex / Docker / GitHub 運用メモ

## 現在の前提

- Docker 作業ディレクトリ: `/home/hiromu/energy`
- Docker Compose: `/home/hiromu/energy/docker/energy-yolo12/docker-compose.yml`
- GitHub 連携済みリポジトリ: `/home/hiromu/energy/src`
- GitHub remote: `https://github.com/rumiha99/src.git`

## 起動

```bash
cd /home/hiromu/energy/docker/energy-yolo12
docker compose up -d --build
docker exec -it yolo12 bash
```

コンテナ内の作業場所は `/home/hiromu/energy`。

## GitHub と連携する範囲

現状では `/home/hiromu/energy` 全体ではなく、`/home/hiromu/energy/src` が Git リポジトリ。
大きな画像、動画、学習済み重み、CSV、Notebook 出力は Git に入れない。

```bash
cd /home/hiromu/energy/src
git status
git add <変更したファイル>
git commit -m "message"
git push
```

## Codex に任せやすい実験ループの形

Codex に自動調整を任せる場合、Notebook ではなく Python スクリプト化する。
最低限、次の 3 つをコマンドで実行できる状態にする。

```bash
python train.py --config configs/experiment.yaml
python evaluate.py --run-dir runs/exp001
python tune.py --config configs/search_space.yaml
```

推奨構成:

```text
src/
  configs/
    train.yaml
    search_space.yaml
  scripts/
    train.py
    evaluate.py
    tune_optuna.py
  runs/              # Git 管理しない
  artifacts/         # Git 管理しない
```

`tune_optuna.py` が行う処理:

1. パラメータ候補を Optuna で生成する
2. そのパラメータで学習する
3. 検証データで評価する
4. 評価指標を Optuna に返す
5. best params と best score を保存する

## Codex に依頼するときの例

```text
/home/hiromu/energy/src の all_fast/pipeline_full.py を読んで、
AREA_THRESHOLD, hida の n, T, SVM の C/gamma を Optuna で探索できる
scripts/tune_optuna.py を作って。
既存ファイルは壊さず、設定は configs/search_space.yaml に分離して。
実行方法と出力ファイルも README に追記して。
```

## 注意

- API キーは `docker-compose.yml` に直書きしない。
- 大容量データや `.pt`, `.pkl`, `.csv`, `.ipynb` 出力は GitHub に入れない。
- 実験結果は `runs/` に保存し、最終的なコードと設定だけ GitHub に commit する。
