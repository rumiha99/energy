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
import hida_fast
import keijo_fast
import size_module_fast
import maesyori_fast

# 1. pip install したライブラリをインポート

# (import文は元のまま)

# === 全体時間の計測開始 ===
print("========== 全処理開始 ==========")
overall_start_time = time.time()
last_timestamp = overall_start_time # 各ステップの時間を測るための中間地点

# 2. パラメータを定義
BASE_DIR = '/home/data/1104_test' 
MODEL_PATH = '/home/YOLO/0708_maesyori/datasets/train/weights/best.pt'
AREA_THRESHOLD = 50000
TARGET_FOLDERS = ['A', 'B', 'C'] 

# 3. 実行
# (if __name__ == "__main__": は引き続き必要)
if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # [1. 前処理 (maesyori_fast)]
    # -----------------------------------------------------------------
    print("\n--- [1. 前処理 (YOLOセグメンテーション等)] 開始 ---")
    maesyori_fast.run(
        base_dir=BASE_DIR,
        model_path=MODEL_PATH,
        target_folders=TARGET_FOLDERS,
        area_threshold=AREA_THRESHOLD,
        max_workers=2
    )
    
    current_time = time.time()
    print(f"--- [1. 前処理] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
    last_timestamp = current_time
    # -----------------------------------------------------------------

    #判別フェーズ
    #特徴量抽出
    data = "1021_fasttest"
    
    # -----------------------------------------------------------------
    # [2. 特徴量抽出 (hida_fast)]
    # -----------------------------------------------------------------
    print("\n--- [2. 特徴量抽出 (hida)] 開始 ---")
    hida_tappleA = hida_fast.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="A",method="45rotate",n=9,T=0.4)
    result_hidaA = hida_tappleA.run_all()
    hida_tappleB = hida_fast.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="B",method="45rotate",n=9,T=0.4)
    result_hidaB = hida_tappleB.run_all()
    hida_tappleC = hida_fast.Hida_folder_jikuari_img_xxxx(base_dir=f"/home/data/{data}",subfolder="C",method="45rotate",n=9,T=0.4)
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
    # [3. 特徴量抽出 (size_module_fast)]
    # -----------------------------------------------------------------
    print("\n--- [3. 特徴量抽出 (size)] 開始 ---")
    size_tappleA = size_module_fast.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="A")
    result_sizeA = size_tappleA.run()
    size_tappleB = size_module_fast.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="B")
    result_sizeB = size_tappleB.run()
    size_tappleC = size_module_fast.Size_folder_taikakusen(base_dir=f"/home/data/{data}",subfolder="C")
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
    # [4. 特徴量抽出 (keijo_fast)]
    # -----------------------------------------------------------------
    print("\n--- [4. 特徴量抽出 (keijo)] 開始 ---")
    keijo_tappleA = keijo_fast.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="A")
    result_keijoA = keijo_tappleA.run()
    keijo_tappleB = keijo_fast.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="B")
    result_keijoB = keijo_tappleB.run()
    keijo_tappleC = keijo_fast.Keijo_folder(base_dir=f"/home/data/{data}",subfolder="C")
    result_keijoC = keijo_tappleC.run()

    dfA = pd.DataFrame(result_keijoA, columns=["filename", "MSE"])
    dfB = pd.DataFrame(result_keijoB, columns=["filename", "MSE"])
    dfC = pd.DataFrame(result_keijoC, columns=["filename", "MSE"])
    result_keijo = pd.concat([dfA, dfB, dfC], axis=0,ignore_index=True)

    current_time = time.time()
    print(f"--- [4. 特徴量抽出 (keijo)] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
    last_timestamp = current_time
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # [5. 特徴量の結合]
    # -----------------------------------------------------------------
    print("\n--- [5. 特徴量の結合] 開始 ---")
    df_merged = pd.merge(result_keijo, result_size, on="filename")
    df_merged = pd.merge(df_merged, result_hida, on="filename")
    # df_merged.to_csv(f"/home/data/{data}/feature.csv", index=False) # コメントアウトされていたためそのまま

    current_time = time.time()
    print(f"--- [5. 特徴量の結合] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
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
    # df_merged.to_csv(f'/home/data/{data}/feature{model}.csv', index=False) # コメントアウトされていたためそのまま
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # === 新しいデータの読み込み ===
    # df = pd.DataFrame(merged)
    # df = pd.read_csv(f'/home/data/{data}/feature{model}.csv') # コメントアウトされていたためそのまま
    df = pd.DataFrame(df_merged)

    # === 特徴量の抽出と標準化 ===
    X_new = df[["MSE", "size_count", "R"]] 
    X_new = scaler.transform(X_new) 

    # === 予測 ===
    y_pred_new = svm_model.predict(X_new)

    # 結果をDataFrameに追加
    df["Predicted_Label"] = y_pred_new

    # CSVとして保存（オプション）
    df.to_csv(f"/home/data/{data}predicted{model}.csv", index=False)

    current_time = time.time()
    print(f"--- [6. SVMによる予測] 完了 (所要時間: {current_time - last_timestamp:.2f}秒) ---")
    # -----------------------------------------------------------------

# === 全体時間の計測終了 ===
overall_end_time = time.time()
print(f"\n========== 全処理完了 ==========")
print(f"(総所要時間: {overall_end_time - overall_start_time:.2f}秒)")

# (※元の start, end 変数での計測は、上記の計測に統合されました)