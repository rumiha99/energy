# CBAM Tuning Review 2026-05-17

対象:

```text
src/FF/AFF/gakusyu/cbam_full_from_images.py
```

## Broad Search Summary

初期探索の出力:

```text
runs/cbam_tuning_20260518_0800/
```

11 trial 完了時点の上位:

| Trial | Test Macro F1 | Test Acc | Fusion | Pretrained | Optimizer | Loss | Image Size | Aug |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| 0005 | 0.9028 | 0.9028 | gate | true | adamw | ce | 256 | strong |
| 0000 | 0.9018 | 0.9028 | concat | true | sgd | ce | 192 | strong |
| 0010 | 0.8873 | 0.8889 | gate | true | adamw | ce | 256 | strong |

観察:

- `feature_only` は 0.56-0.58 程度で弱い。
- `image_only` は初期探索から外す対象にした。
- pretrained ConvNeXt + 画像特徴 + handcrafted特徴の融合が強い。
- `gate` と `concat` が有望。
- `sum` は中位。
- A クラスはほぼ正解できており、主な誤分類は B/C 間。
- 上位は CrossEntropy, strong augmentation, learning rate 約 `5e-4`-`7e-4` に集中。

## Focused Search

第二段階の出力:

```text
runs/cbam_tuning_focused_20260518_0800/
```

切り替え内容:

- `fusion_type`: `gate`, `concat` に限定。
- `pretrained`: `true` 固定。
- `feature_only`, `image_only`, pretrainedなしを除外。
- learning rate と weight decay を上位 trial 周辺に寄せる。
- image size は `192`, `224`, `256`, `288` を探索。
- augmentation は `strong`, `light` のみに限定。
- B/C 誤分類の改善を狙って、dropout と fusion hidden の周辺を再探索。

起動コマンド:

```bash
python scripts/tune_cbam_full_from_images.py \
  --phase focused \
  --deadline 2026-05-18T08:00:00 \
  --output-dir /home/hiromu/energy/runs/cbam_tuning_focused_20260518_0800 \
  --study-name cbam_focused_after_review \
  --min-epochs 18 \
  --max-epochs 55
```

