import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt # visualizeメソッドで必要
import concurrent.futures
from tqdm import tqdm # 進捗表示のために追加（推奨）

# ===================================================================
# 1. 単一画像の処理クラス（変更なし）
# ===================================================================
class Size_taikakusen:
    def __init__(self, mask_path):
        self.mask_path = mask_path
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"画像が読み込めません: {mask_path}")
        _, self.mask_bin = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        self.center_of_mass = self._compute_center_of_mass()

    def _compute_center_of_mass(self):
        contours, _ = cv2.findContours(self.mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # print("輪郭が見つかりません")
            return np.array([0, 0])
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            # print("面積ゼロの輪郭")
            return np.array([0, 0])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.array([cx, cy])

    def compute_diameters(self):
        min_distance = float('inf')
        max_distance = 0
        min_points = None
        max_points = None

        for angle in range(0, 180):
            angle_rad = np.deg2rad(angle)
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)

            # 正方向の探索
            x, y = self.center_of_mass.copy()
            farthest_point = None
            while 0 <= int(x) < self.mask.shape[1] and 0 <= int(y) < self.mask.shape[0]:
                if self.mask_bin[int(y), int(x)] == 0: # 輪郭の外に出た瞬間
                    farthest_point = (x, y)
                    break
                x += dx
                y += dy
            
            # 負方向の探索
            x, y = self.center_of_mass.copy()
            nearest_point = None
            while 0 <= int(x) < self.mask.shape[1] and 0 <= int(y) < self.mask.shape[0]:
                if self.mask_bin[int(y), int(x)] == 0: # 輪郭の外に出た瞬間
                    nearest_point = (x, y)
                    break
                x -= dx
                y -= dy

            if farthest_point is not None and nearest_point is not None:
                dist = np.linalg.norm(np.array(farthest_point) - np.array(nearest_point))
                if dist < min_distance:
                    min_distance = dist
                    min_points = (farthest_point, nearest_point)
                if dist > max_distance:
                    max_distance = dist
                    max_points = (farthest_point, nearest_point)

        self.min_distance = min_distance if min_distance != float('inf') else 0
        self.max_distance = max_distance
        self.min_points = min_points
        self.max_points = max_points
        return (self.min_distance + self.max_distance) / 2 # 平均を返す
    
    # visualize メソッドは変更なし

# ===================================================================
# 2. 各プロセスで実行されるヘルパー関数
# ===================================================================
def process_single_image(file_path):
    """
    単一の画像ファイルを処理し、ファイル名と計算結果のタプルを返す。
    エラーが発生した場合は None を返す。
    """
    try:
        file_name = os.path.basename(file_path)
        size_calc = Size_taikakusen(file_path)
        avg_diameter = size_calc.compute_diameters()
        return (file_name, avg_diameter)
    except Exception as e:
        print(f"⚠️ エラー発生 ({os.path.basename(file_path)}): {e}")
        return None

# ===================================================================
# 3. フォルダ単位の処理クラス（マルチプロセス対応に修正）
# ===================================================================
class Size_folder_taikakusen:
    def __init__(self, base_dir, mask_dir="mask", subfolder="collage_1"):
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.subfolder = subfolder
        self.input_folder = os.path.join(base_dir, mask_dir, subfolder)
        self.data_tapple = []
        print("探索対象フォルダ:", self.input_folder)

    def run(self):
        if not os.path.isdir(self.input_folder):
            print(f"⚠️ フォルダが存在しません: {self.input_folder}")
            return []

        # 処理対象の画像ファイルパスのリストを作成
        image_paths = [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        if not image_paths:
            print(f"⚠️ フォルダ内に処理対象の画像がありません: {self.input_folder}")
            return []
        
        print(f"📂 {len(image_paths)} 枚の画像をマルチプロセスで処理します。")

        # ProcessPoolExecutorを使用して並列処理を実行
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # executor.mapを使い、各画像パスに対してprocess_single_imageを並列実行
            results_iterator = executor.map(process_single_image, image_paths)

            # tqdmを使って進捗バーを表示しつつ、Noneでない結果のみをリストに格納
            self.data_tapple = []
            desc = f"処理中 ({self.subfolder})"
            for result in tqdm(results_iterator, total=len(image_paths), desc=desc):
                if result is not None:
                    self.data_tapple.append(result)
        
        return self.data_tapple

# ===================================================================
# 4. 実行例
# ===================================================================
