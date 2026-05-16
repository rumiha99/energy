import os 
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = "0203_energee_after"

inputfolder_lists = [
    f"/home/data/{data}/maskBB/A",
    f"/home/data/{data}/maskBB/B",
    f"/home/data/{data}/maskBB/C",
]
one_dimensional_data_dict = {}
evaluation_results = {}

for folder in inputfolder_lists:
    folder_name = os.path.basename(folder)
    image_paths = glob.glob(os.path.join(folder, '*.JPEG'))

    for img_path in image_paths:
        # 画像の読み込み
        mask = cv2.imread(img_path)

        # グレースケール画像に変換
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 二値化
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 輪郭を検出し、最大の輪郭を取得
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        max_contour = max(contours, key=cv2.contourArea)

        # 最小外接円を取得
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        radius = int(radius)

        # 重心を計算
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        center = (cX, cY)

        # 極座標変換
        h, w = gray.shape
        flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        linear_polar = cv2.warpPolar(gray, (w, h), center, radius, flags)

        # 行ごとの黒ピクセル数をカウント
        black_pixel_count = np.sum(linear_polar == 0, axis=1)
        file_name = os.path.basename(img_path)
        one_dimensional_data_dict[file_name] = black_pixel_count

        # 真円の場合の理想データ（黒ピクセル数が0）
        y_pseudo = np.zeros_like(black_pixel_count)

        # 評価指標の計算
        mae = mean_absolute_error(y_pseudo, black_pixel_count)
        mse = mean_squared_error(y_pseudo, black_pixel_count)

        evaluation_results[file_name] = {
            'MSE': mse, 
            'MAE': mae, 
            'Folder': folder_name
        }
mae_values = [metrics['MAE'] for metrics in evaluation_results.values()]
print(mae_values)
# print(evaluation_results.values())