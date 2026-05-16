import os
import cv2
import numpy as np

class Size_folder:
    def __init__(self, base_dir, mask_dir="mask", subfolder="collage_1"):
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.subfolder = subfolder

        # マスク画像があるフォルダのパスを組み立てる
        self.input_folder = os.path.join(base_dir, mask_dir, subfolder)
        self.data_tapple = []

        print("探索対象フォルダ:", self.input_folder)

    def run(self):
        if not os.path.isdir(self.input_folder):
            print(f"⚠️ フォルダが存在しません: {self.input_folder}")
            return self.data_tapple

        for file in os.listdir(self.input_folder):
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPEG')):
                file_path = os.path.join(self.input_folder, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"⚠️ 画像読み込み失敗: {file_path}")
                    continue

                # 二値化
                _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

                # 白ピクセルカウント
                white_pixel_count = np.sum(binary == 255)

                # 結果保存
                self.data_tapple.append((file, white_pixel_count))

        return self.data_tapple

class Size_file:
    def __init__(self, image_path):
        self.image_path = image_path

    def count_white_pixels(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"画像が読み込めませんでした: {self.image_path}")

        # 二値化
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

        # 白ピクセルのカウント
        white_pixel_count = np.sum(binary == 255)

        return white_pixel_count
    
# size1 = Size_file('/home/data/maesyori_img/mask/collage_1/mask_1_1.jpg')
# white_count = size1.count_white_pixels()
# print(white_count)

# size2 = Size_folder(['/home/data/maesyori_img/mask/collage_1'])
# white_count_list = size2.count_white_pixels()
# print(white_count_list)

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
            print("輪郭が見つかりません")
            return np.array([0, 0])
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            print("面積ゼロの輪郭")
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

            x, y = self.center_of_mass
            farthest_x, farthest_y = None, None
            while 0 <= int(x) < self.mask.shape[1] and 0 <= int(y) < self.mask.shape[0]:
                if self.mask_bin[int(y), int(x)] == 0:
                    farthest_x, farthest_y = x, y
                    break
                x += dx
                y += dy

            x, y = self.center_of_mass
            nearest_x, nearest_y = None, None
            while 0 <= int(x) < self.mask.shape[1] and 0 <= int(y) < self.mask.shape[0]:
                if self.mask_bin[int(y), int(x)] == 0:
                    nearest_x, nearest_y = x, y
                    break
                x -= dx
                y -= dy

            if farthest_x is not None and nearest_x is not None:
                dist = np.sqrt(((farthest_x - nearest_x) - 2)**2 + ((farthest_y - nearest_y) - 2)**2)
                if dist < min_distance:
                    min_distance = dist
                    min_points = ((farthest_x, farthest_y), (nearest_x, nearest_y))
                if dist > max_distance:
                    max_distance = dist
                    max_points = ((farthest_x, farthest_y), (nearest_x, nearest_y))

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_points = min_points
        self.max_points = max_points
        return (min_distance + max_distance) / 2  # 平均を返す

    def visualize(self):
        image = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        cx, cy = self.center_of_mass
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

        if self.min_points:
            cv2.line(image,
                     (int(self.min_points[0][0]), int(self.min_points[0][1])),
                     (int(self.min_points[1][0]), int(self.min_points[1][1])),
                     (0, 255, 255), 2)

        if self.max_points:
            cv2.line(image,
                     (int(self.max_points[0][0]), int(self.max_points[0][1])),
                     (int(self.max_points[1][0]), int(self.max_points[1][1])),
                     (0, 0, 255), 2)

        plt.figure(figsize=(8, 6))
        plt.imshow(image[..., ::-1])
        plt.title("Mask with Min (Yellow) and Max (Red) Diameters")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    
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
            return self.data_tapple

        for file in os.listdir(self.input_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(self.input_folder, file)

                try:
                    size_calc = Size_taikakusen(file_path)
                    avg_diameter = size_calc.compute_diameters()
                    self.data_tapple.append((file, avg_diameter))
                except Exception as e:
                    print(f"⚠️ エラー（{file}）: {e}")

        return self.data_tapple