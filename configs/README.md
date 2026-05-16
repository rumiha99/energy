# configs

実験や推論で変える値は、できるだけここに置く。

目的:

- Python ファイル内のパス直書きを減らす
- Codex がパラメータ探索しやすくする
- 実験ごとの設定を `runs/` に保存しやすくする

現在の入口:

- `preprocess_yolo.json`: YOLO セグメンテーション前処理用
- `cbam_full_from_images.json`: CBAM/ConvNeXt 完全版。画像から h0-h7, size_count, R を再計算して学習・評価する
