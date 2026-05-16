import os
import glob
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

class KeijoAnalyzer:
    def __init__(self):
        self.results = {}
        self.data_vectors = {}

    def analyze_image(self, img_path):
        mask = cv2.imread(img_path)
        if mask is None:
            print(f"⚠️ 読み込み失敗: {img_path}")
            return None, None

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ 輪郭なし: {img_path}")
            return None, None

        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        radius = int(radius)

        M = cv2.moments(max_contour)
        cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
        center = (cX, cY)

        h, w = gray.shape
        flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        linear_polar = cv2.warpPolar(gray, (w, h), center, radius, flags)

        black_pixel_count = np.sum(linear_polar == 0, axis=1)

        # 評価指標の計算
        y_pseudo = np.zeros_like(black_pixel_count)
        mse = mean_squared_error(y_pseudo, black_pixel_count)

        return black_pixel_count, mse


class Keijo_file(KeijoAnalyzer):
    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

    def run(self):
        _, mse = self.analyze_image(self.img_path)
        return mse  # MSE のみ返す


class Keijo_folder(KeijoAnalyzer):
    def __init__(self, base_dir, mask_dir="mask", subfolder="collage_1", output_csv=None):
        super().__init__()
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.subfolder = subfolder
        self.folder_path = os.path.join(base_dir, mask_dir, subfolder)
        self.output_csv = output_csv or os.path.join(base_dir, "keijo_mse.csv")

    def run(self):
        image_paths = glob.glob(os.path.join(self.folder_path, "*.jpg")) + glob.glob(os.path.join(self.folder_path, "*.png"))
        print(f"📂 フォルダ: {self.folder_path} に画像 {len(image_paths)} 枚")
        mse_results = []

        for img_path in image_paths:
            file_name = os.path.basename(img_path)
            vec, mse = self.analyze_image(img_path)
            mse_results.append((file_name, mse))
        return mse_results

    def save_results(self):
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.index.name = 'filename'
        df.to_csv(self.output_csv)
        print(f"✅ 結果を保存: {self.output_csv}")


# import keijo
# analyzer = keijo.Keijo_file("/home/data/maesyori_img/mask/collage_1/mask_1_1.jpg")
# result = analyzer.run()
# print(result)
# import keijo
# analyzer = keijo.Keijo_folder(base_dir="/home/data/maesyori_img")
# result = analyzer.run()
# print(result)