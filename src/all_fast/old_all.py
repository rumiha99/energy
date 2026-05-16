import os 
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve
import re
import math
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score
import joblib
import time
from ultralytics import YOLO

import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
import multiprocessing
import time
import os
import sys

# ._internal_worker は、あなたが提示した pipeline_full.py の
# process_image_subdirectory 関数を import している想定です
from pipeline_full import process_image_subdirectory

def run(base_dir, model_path, target_folders, area_threshold=50000, max_workers=1):
    """
    画像処理パイプラインを実行します。
    
    Args:
        max_workers (int): 
            2以上: 指定したワーカー数で並列実行 (マルチプロセス)
            1以下: 逐次実行 (シングルプロセス・デバッグ/比較用)
    """
    
    print(f"処理を開始します... 対象: {target_folders}")
    overall_start_time = time.time()

    # --- ▼ ここからが変更点 ▼ ---

    # max_workers が 2 以上の場合のみマルチプロセス（Pool）を使用する
    if max_workers > 1:
        print(f"並列処理を開始します (最大ワーカー数: {max_workers})")
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("マルチプロセスの開始メソッドを 'spawn' に設定しました。")
        except RuntimeError:
            # 既に設定されている場合は何もしない
            pass 

        # Poolに渡すための引数リストを作成
        tasks = [
            (base_dir, folder, model_path, area_threshold) for folder in target_folders
        ]

        # マルチプロセスで実行
        with multiprocessing.Pool(processes=max_workers) as pool:
            pool.starmap(process_image_subdirectory, tasks)
    
    else:
        # max_workersが 1 以下の場合は、単純な for ループで逐次実行する
        print("逐次処理（シングルプロセス）を開始します。")
        for folder in target_folders:
            # process_image_subdirectory を直接呼び出す
            process_image_subdirectory(
                base_data_dir=base_dir, 
                sub_folder=folder, 
                model_path=model_path, 
                area_threshold=area_threshold
            )

    # --- ▲ ここまでが変更点 ▲ ---

    overall_end_time = time.time()
    print(f"\nすべての処理が完了しました。 (総所要時間: {overall_end_time - overall_start_time:.2f}秒)")
start = time.time()

# --- 入力と出力の設定 ---
# (変更なし)
# max_workers=1 (デフォルト) または 0 を指定
# [run 関数の定義部分は、元のコードのまま変更ありません]
# ...
# def run(base_dir, model_path, target_folders, area_threshold=50000, max_workers=1):
#     ... (中略) ...
#     print(f"\nすべての処理が完了しました。 (総所要時間: {overall_end_time - overall_start_time:.2f}秒)")
# -----------------------------------------------------------------


# === 全体時間の計測開始 ===
print("========== 全処理開始 ==========")
overall_start_time = time.time()
last_timestamp = overall_start_time # 各ステップの時間を測るための中間地点

# -----------------------------------------------------------------
# [1. 前処理 (pipeline_full)]
# -----------------------------------------------------------------
print("\n--- [1. 前処理 (YOLOセグメンテーション等)] 開始 ---")

# max_workers=1 (デフォルト) または 0 を指定
run(
    base_dir="/home/data/1104_test",
    model_path='/home/YOLO/0708_maesyori/datasets/train/weights/best.pt',
    target_folders=['A', 'B', 'C'],
    max_workers=1 
)

current_time = time.time()
print(f"--- [1. 前処理] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
last_timestamp = current_time
# -----------------------------------------------------------------


import hida
import keijo
import size_module

# -----------------------------------------------------------------
# [2. 特徴量抽出 (hida)]
# -----------------------------------------------------------------
print("\n--- [2. 特徴量抽出 (hida)] 開始 ---")
data = "1104_test"
hida_tappleA = hida.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="A",method="45rotate",n=9,T=0.4)
result_hidaA = hida_tappleA.run_all()
hida_tappleB = hida.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="B",method="45rotate",n=9,T=0.4)
result_hidaB = hida_tappleB.run_all()
hida_tappleC = hida.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="C",method="45rotate",n=9,T=0.4)
result_hidaC = hida_tappleC.run_all()

dfA = pd.DataFrame(result_hidaA, columns=["filename", "R"])
dfB = pd.DataFrame(result_hidaB, columns=["filename", "R"])
dfC = pd.DataFrame(result_hidaC, columns=["filename", "R"])
dfA ["Label"] = "0"
dfB ["Label"] = "1"
dfC ["Label"] = "2"
result_hida = pd.concat([dfA, dfB, dfC], axis=0,ignore_index=True)

current_time = time.time()
print(f"--- [2. 特徴量抽出 (hida)] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
last_timestamp = current_time
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# [3. 特徴量抽出 (size)]
# -----------------------------------------------------------------
print("\n--- [3. 特徴量抽出 (size)] 開始 ---")
size_tappleA = size_module.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="A")
result_sizeA = size_tappleA.run()
size_tappleB = size_module.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="B")
result_sizeB = size_tappleB.run()
size_tappleC = size_module.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="C")
result_sizeC = size_tappleC.run()

dfA = pd.DataFrame(result_sizeA, columns=["filename", "size_count"])
dfB = pd.DataFrame(result_sizeB, columns=["filename", "size_count"])
dfC = pd.DataFrame(result_sizeC, columns=["filename", "size_count"])
result_size = pd.concat([dfA, dfB, dfC], axis=0,ignore_index=True)

current_time = time.time()
print(f"--- [3. 特徴量抽出 (size)] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
last_timestamp = current_time
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# [4. 特徴量抽出 (keijo)]
# -----------------------------------------------------------------
print("\n--- [4. 特徴量抽出 (keijo)] 開始 ---")
keijo_tappleA = keijo.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="A")
result_keijoA = keijo_tappleA.run()
keijo_tappleB = keijo.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="B")
result_keijoB = keijo_tappleB.run()
keijo_tappleC = keijo.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="C")
result_keijoC = keijo_tappleC.run()

# --- ▼▼▼【修正提案 1】▼▼▼ ---
# 元のコードでは result_hidaA 等から "MSE" を取得していましたが、
# おそらく keijo の結果 (result_keijoA 等) を使う意図だと思われますので修正します。
#
# (元のコード)
# dfA = pd.DataFrame(result_hidaA, columns=["filename", "MSE"])
# dfB = pd.DataFrame(result_hidaB, columns=["filename", "MSE"])
# dfC = pd.DataFrame(result_hidaC, columns=["filename", "MSE"])
#
# (修正後のコード)
dfA = pd.DataFrame(result_keijoA, columns=["filename", "MSE"])
dfB = pd.DataFrame(result_keijoB, columns=["filename", "MSE"])
dfC = pd.DataFrame(result_keijoC, columns=["filename", "MSE"])
# --- ▲▲▲【修正提案 1】▲▲▲ ---

result_keijo = pd.concat([dfA, dfB, dfC], axis=0,ignore_index=True)

current_time = time.time()
print(f"--- [4. 特徴量抽出 (keijo)] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
last_timestamp = current_time
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# [5. 特徴量の結合・保存]
# -----------------------------------------------------------------
print("\n--- [5. 特徴量の結合・保存] 開始 ---")
df_merged = pd.merge(result_keijo, result_size, on="filename")
df_merged = pd.merge(df_merged, result_hida, on="filename")
df_merged.to_csv(f"/home/data/{data}/feature.csv", index=False)

current_time = time.time()
print(f"--- [5. 特徴量の結合・保存] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
last_timestamp = current_time
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# [6. SVMによる予測]
# -----------------------------------------------------------------
print("\n--- [6. SVMによる予測] 開始 ---")
# === モデルとスケーラーの読み込み ===
model = "_jikuari_tai"
model_path = "../svm_model.pkl"
scaler_path = "../scaler.pkl"
df_merged.to_csv(f'/home/data/{data}/feature{model}.csv', index=False)
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# === 新しいデータの読み込み ===
# df = pd.DataFrame(merged)
df = pd.read_csv(f'/home/data/{data}/feature{model}.csv')
df = pd.DataFrame(df)

# === 特徴量の抽出と標準化 ===
X_new = df[["MSE", "size_count", "R"]]# 学習時と同じ特徴量を使用
X_new = scaler.transform(X_new)# 標準化

# === 予測 ===
y_pred_new = svm_model.predict(X_new)

# 結果をDataFrameに追加
df["Predicted_Label"] = y_pred_new

# --- ▼▼▼【修正提案 2】▼▼▼ ---
# 予測結果の確認 (元の print([["MSE",...]]) ではヘッダー名しか表示されないため、
# DataFrameの内容を表示するように修正します)
print("予測結果（一部）:")
print(df[["filename", "MSE", "size_count", "R", "Predicted_Label"]].head())
# --- ▲▲▲【修正提案 2】▲▲▲ ---

# CSVとして保存（オプション）
df.to_csv(f"/home/data/{data}predicted{model}.csv", index=False)

current_time = time.time()
print(f"--- [6. SVMによる予測] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
# -----------------------------------------------------------------


# === 全体時間の計測終了 ===
overall_end_time = time.time()
print(f"\n========== 全処理完了 ==========")
print(f"(総所要時間: {overall_end_time - overall_start_time:.2f}秒)")

# (※元の start, end 変数での計測は不要なため削除しました)