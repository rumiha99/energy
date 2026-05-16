import os
import re
import numpy as np
import cv2
from scipy.ndimage import median_filter

# class Hida_file: #sdisvalの閾値処理結果を直接hxyに変換し、それをマスク画像に対してAND演算で絞り込んでいる

#     def __init__(self, img_path, mask_path, n=15, T=0.2):
#         self.img_path = img_path
#         self.mask_path = mask_path
#         self.n = n
#         self.T = T

#     def fmxy(self, absfxy, mxy):
#         return np.where(absfxy > mxy, 1, 0)

#     def Min(self, a, b):
#         return np.minimum(a, b)

#     def G12(self, theta1, theta2):
#         condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
#         result1 = np.abs(theta1 - theta2)
#         condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
#         result2 = theta2 - 2 * np.pi - theta1
#         condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
#         result3 = np.abs(theta1 - theta2)
#         condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
#         result4 = theta1 - theta2 - 2 * np.pi
#         result = np.where(condition1, result1, 
#                 np.where(condition2, result2, 
#                 np.where(condition3, result3, 
#                 np.where(condition4, result4, 0))))
#         return result

#     def SquareSum(self, I, x, y, h, w):
#         n = self.n
#         x1, y1 = x - n, y - n
#         x2, y2 = x + n, y + n
#         x1, x2 = max(x1, 0), min(x2, w - 2)
#         y1, y2 = max(y1, 0), min(y2, h - 2)
#         total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
#         return total

#     def sdis(self, Iruv, Imyv, x, y, h, w):
#         Tr = self.SquareSum(Iruv, x, y, h, w)
#         Tm = self.SquareSum(Imyv, x, y, h, w)
#         return Tr / Tm if Tm != 0 else 0

#     def calculate_R(self):
#         # 画像読み込み
#         img = cv2.imread(self.img_path)
#         mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
#         masked_img = cv2.imread(self.img_path)
#         h, w = img.shape[:2]

#         # 重心計算
#         coords = np.column_stack(np.where(mask_img == 255))
#         yc, xc = np.mean(coords, axis=0) if len(coords) > 0 else (0, 0)

#         # 勾配方向と大きさ
#         gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#         fdy, fdx = np.gradient(gray.astype(np.float32))
#         f0xy = np.arctan2(fdy, fdx)
#         absfxy = np.uint8(np.sqrt(fdx**2 + fdy**2))

#         # 中心からのベクトル角度
#         Y, X = np.indices((h, w))
#         dx, dy = X - xc, Y - yc
#         C0xy = np.arctan2(dy, dx)

#         # fdisxy
#         fdisxy = self.Min(
#             self.G12(C0xy + np.pi / 2, f0xy) ** 2,
#             self.G12(C0xy - np.pi / 2, f0xy) ** 2
#         )

#         # メディアンフィルタ
#         kernel_size = 2 * self.n + 1
#         mxy = np.uint8(cv2.medianBlur(absfxy, kernel_size))

#         # rdis
#         fmxy_result = self.fmxy(absfxy, mxy)
#         rdisxy = fmxy_result * fdisxy

#         # 積分画像
#         Iruv = cv2.integral(rdisxy)
#         Imyv = cv2.integral(fmxy_result.astype(np.uint8))

#         # sdis計算
#         sdisval = np.zeros((h, w), dtype=np.float32)
#         for y in range(h):
#             for x in range(w):
#                 sdisval[y, x] = self.sdis(Iruv, Imyv, x, y, h, w)
#         sdisval = np.nan_to_num(sdisval, nan=0.0, posinf=0.0, neginf=0.0)

#         # 閾値処理
#         hxy = np.where(sdisval < self.T, 1, 0).astype(np.uint8)
#         hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)

#         # R計算
#         count_mask = np.sum(mask_img == 255)
#         count_hida = np.sum(hxy2 == 1)
#         R = count_hida / count_mask if count_mask > 0 else 0

#         return R,count_hida,count_mask

class Hida_file:
    def __init__(self, img_path, mask_path, n=15, T=0.2):
        self.img_path = img_path
        self.mask_path = mask_path
        self.n = n
        self.T = T
    
    def fmxy(self, absfxy, mxy):
        return np.where(absfxy > mxy, 1, 0)

    def Min(self, a, b):
        return np.minimum(a, b)

    def G12(self, theta1, theta2):
        condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
        result1 = np.abs(theta1 - theta2)
        condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
        result2 = theta2 - 2 * np.pi - theta1
        condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
        result3 = np.abs(theta1 - theta2)
        condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
        result4 = theta1 - theta2 - 2 * np.pi
        result = np.where(condition1, result1, 
                 np.where(condition2, result2, 
                 np.where(condition3, result3, 
                 np.where(condition4, result4, 0))))
        return result

    def SquareSum(self, I, x, y, h, w, n):
        x1, y1 = x - n, y - n
        x2, y2 = x + n, y + n
        x1, x2 = max(x1, 0), min(x2, w - 2)
        y1, y2 = max(y1, 0), min(y2, h - 2)
        total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
        return total

    def sdis(self, Iruv, Imyv, x, y, h, w, n):
        Tr = self.SquareSum(Iruv, x, y, h, w, n)
        Tm = self.SquareSum(Imyv, x, y, h, w, n)
        return Tr / Tm
    
    def run(self):
        # 画像の読み込み
        img = cv2.imread(self.img_path)
        mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        masked_img = cv2.imread(self.img_path)
        h, w = img.shape[:2]

        # 重心を計算
        x_sum, y_sum, count = 0, 0, 0
        for i in range(h):
            for j in range(w):
                if mask_img[i][j] == 255:
                    x_sum += j
                    y_sum += i
                    count += 1
        xc, yc = (x_sum / count, y_sum / count) if count > 0 else (0, 0)

        # fθ(x,y)(勾配の方向), |f(x,y)| (2)(3)
        image = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        fdy, fdx = np.gradient(image)
        f0xy = np.arctan2(fdy, fdx)
        absfxy = np.uint8(np.sqrt(fdx**2 + fdy**2))

        # C0(x,y)：中心からのベクトルの角度
        height, width = image.shape
        C0xy = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                dx, dy = x - xc, y - yc
                C0xy[y, x] = np.arctan(dy / dx) if dx != 0 else 0

        # f(xy)の勾配ベクトルが中心から(x,y)へのベクトルへ垂直か評価する関数fdisxy(4)(5)
        fdisxy = self.Min(self.G12(C0xy + np.pi/2, f0xy)**2, self.G12(C0xy - np.pi/2, f0xy)**2)

        # mxy = |fxy|に対する2n+1×2n+1のメディアンフィルタリングの結果
        kernel_size = 2 * self.n + 1
        mxy = np.uint8(cv2.medianBlur(absfxy, kernel_size))

        # rdis (8)
        rdisxy = self.fmxy(absfxy, mxy) * fdisxy

        # Iruv, Imyv (9)(10)
        Iruv = cv2.integral(rdisxy)
        Imyv = cv2.integral(self.fmxy(absfxy, mxy).astype(np.uint8))

        # sdis計算
        sdisval = np.zeros((image.shape[0], image.shape[1]))
        for y in range(0, image.shape[0], 1):
            for x in range(0, image.shape[1], 1):
                sdisval[y, x] = self.sdis(Iruv, Imyv, x, y, h, w, self.n)
        sdisval = np.nan_to_num(sdisval, nan=0.0, posinf=0.0, neginf=0.0)

        # 閾値処理
        hxy = np.where(sdisval < self.T, 1, 0)
        hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)

        # シイタケ領域のPixel数を計算
        count_mask = np.sum(mask_img == 255)
        count_hida = np.sum(hxy2 == 1)
        R = count_hida / count_mask if count_mask > 0 else 0
        
        return R
    







class Hida_folder:
    def __init__(self, base_dir, img_dir="combined", mask_dir="mask", subfolder="collage_1", n=15, T=0.2):
        self.base_dir = base_dir
        self.subfolder = subfolder
        self.n = n
        self.T = T
        self.combined_path = os.path.join(base_dir, img_dir, subfolder)
        self.mask_path = os.path.join(base_dir, mask_dir, subfolder)

    def extract_index(self, name):
        match = re.search(r'(\d+_\d+)', name)
        return match.group(1) if match else None

    def run_all(self):
        img_files = os.listdir(self.combined_path)
        mask_files = os.listdir(self.mask_path)

        # 画像とマスクファイルを紐づけ
        self.img_dict = {self.extract_index(f): f for f in img_files if self.extract_index(f)}
        self.mask_dict = {self.extract_index(f): f for f in mask_files if self.extract_index(f)}
        
        # 共通のキー（インデックス）を取得
        common_keys = sorted(set(self.img_dict.keys()) & set(self.mask_dict.keys()))

        R_results = []
        for key in common_keys:
            try:
                img_file = os.path.join(self.combined_path, self.img_dict[key])
                mask_file = os.path.join(self.mask_path, self.mask_dict[key])

                # Hida_file を使って R を計算
                instance = Hida_file(img_file, mask_file, n=self.n, T=self.T)
                R = instance.run()
                R_results.append((self.mask_dict[key], R))
            except Exception as e:
                print(f"Error processing {key}: {e}")
        return R_results

class Hida_folder_img_xxxx:
    def __init__(self, base_dir, img_dir="combined", mask_dir="mask", subfolder="collage_1", n=15, T=0.2):
        self.base_dir = base_dir
        self.subfolder = subfolder
        self.n = n
        self.T = T
        self.combined_path = os.path.join(base_dir, img_dir, subfolder)
        self.mask_path = os.path.join(base_dir, mask_dir, subfolder)

    def extract_index(self, name):
        """
        ファイル名から共通のキーを抽出する方法を変更。
        "ファイル名本体_combined.png" や "ファイル名本体_mask.png" から、
        "ファイル名本体" の部分（例: "IMG_1829"）を抽出する。
        """
        # ファイル名から "_combined.png" または "_mask.png" を取り除く
        if name.endswith("_combined.png"):
            return name[:-13] # "_combined.png" の文字数
        elif name.endswith("_mask.png"):
            return name[:-9] # "_mask.png" の文字数
        else:
            # 念のため、予期しないファイル名の場合は拡張子のみ取り除く
            return os.path.splitext(name)[0]

    def run_all(self):
        img_files = os.listdir(self.combined_path)
        mask_files = os.listdir(self.mask_path)

        # 画像とマスクファイルを紐づけ
        self.img_dict = {self.extract_index(f): f for f in img_files if self.extract_index(f)}
        self.mask_dict = {self.extract_index(f): f for f in mask_files if self.extract_index(f)}
        
        # 共通のキー（インデックス）を取得
        common_keys = sorted(set(self.img_dict.keys()) & set(self.mask_dict.keys()))

        R_results = []
        for key in common_keys:
            try:
                img_file = os.path.join(self.combined_path, self.img_dict[key])
                mask_file = os.path.join(self.mask_path, self.mask_dict[key])

                # Hida_file を使って R を計算
                instance = Hida_file(img_file, mask_file, n=self.n, T=self.T)
                R = instance.run()
                R_results.append((self.mask_dict[key], R))
            except Exception as e:
                print(f"Error processing {key}: {e}")
        return R_results

# class Hida_folder:
#     def __init__(self, base_dir, img_dir = "combined",mask_dir = "mask",subfolder='collage_1', n=15, T=0.2):
#         self.base_dir = base_dir
#         self.subfolder = subfolder
#         self.n = n
#         self.T = T
#         self.combined_path = os.path.join(base_dir, img_dir, subfolder)
#         self.mask_path = os.path.join(base_dir, mask_dir, subfolder)

#     def fmxy(self, absfxy, mxy):
#         return np.where(absfxy > mxy, 1, 0)

#     def Min(self, a, b):
#         return np.minimum(a, b)

#     def G12(self, theta1, theta2):
#         condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
#         result1 = np.abs(theta1 - theta2)
#         condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
#         result2 = theta2 - 2 * np.pi - theta1
#         condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
#         result3 = np.abs(theta1 - theta2)
#         condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
#         result4 = theta1 - theta2 - 2 * np.pi
#         result = np.where(condition1, result1, 
#                 np.where(condition2, result2, 
#                 np.where(condition3, result3, 
#                 np.where(condition4, result4, 0))))
#         return result

#     def SquareSum(self, I, x, y, h, w):
#         n = self.n
#         x1, y1 = x - n, y - n
#         x2, y2 = x + n, y + n
#         x1, x2 = max(x1, 0), min(x2, w - 2)
#         y1, y2 = max(y1, 0), min(y2, h - 2)
#         total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
#         return total

#     def sdis(self, Iruv, Imyv, x, y, h, w):
#         Tr = self.SquareSum(Iruv, x, y, h, w)
#         Tm = self.SquareSum(Imyv, x, y, h, w)
#         return Tr / Tm if Tm != 0 else 0

#     def extract_index(self, name):
#         match = re.search(r'(\d+_\d+)', name)
#         return match.group(1) if match else None

#     def calculate_R(self, img_path, mask_path):
#         img = cv2.imread(img_path)
#         mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         masked_img = cv2.imread(img_path)
#         h, w = img.shape[:2]

#         coords = np.column_stack(np.where(mask_img == 255))
#         yc, xc = np.mean(coords, axis=0) if len(coords) > 0 else (0, 0)

#         gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#         fdy, fdx = np.gradient(gray.astype(np.float32))
#         f0xy = np.arctan2(fdy, fdx)
#         absfxy = np.uint8(np.sqrt(fdx**2 + fdy**2))

#         Y, X = np.indices((h, w))
#         dx, dy = X - xc, Y - yc
#         C0xy = np.arctan2(dy, dx)

#         fdisxy = self.Min(
#             self.G12(C0xy + np.pi / 2, f0xy) ** 2,
#             self.G12(C0xy - np.pi / 2, f0xy) ** 2
#         )

#         kernel_size = 2 * self.n + 1
#         mxy = np.uint8(cv2.medianBlur(absfxy, kernel_size))

#         fmxy_result = self.fmxy(absfxy, mxy)
#         rdisxy = fmxy_result * fdisxy

#         Iruv = cv2.integral(rdisxy)
#         Imyv = cv2.integral(fmxy_result.astype(np.uint8))

#         sdisval = np.zeros((h, w), dtype=np.float32)
#         for y in range(h):
#             for x in range(w):
#                 sdisval[y, x] = self.sdis(Iruv, Imyv, x, y, h, w)
#         sdisval = np.nan_to_num(sdisval, nan=0.0, posinf=0.0, neginf=0.0)

#         hxy = np.where(sdisval < self.T, 1, 0).astype(np.uint8)
#         hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)

#         count_mask = np.sum(mask_img == 255)
#         count_hida = np.sum(hxy2 == 1)
#         R = count_hida / count_mask if count_mask > 0 else 0

#         return R

#     def run_all(self):
#         img_files = os.listdir(self.combined_path)
#         mask_files = os.listdir(self.mask_path)

#         img_dict = {self.extract_index(f): f for f in img_files if self.extract_index(f)}
#         mask_dict = {self.extract_index(f): f for f in mask_files if self.extract_index(f)}
#         common_keys = sorted(set(img_dict.keys()) & set(mask_dict.keys()))

#         R_results = []
#         for key in common_keys:
#             img_file = os.path.join(self.combined_path, img_dict[key])
#             mask_file = os.path.join(self.mask_path, mask_dict[key])
#             R = self.calculate_R(img_file, mask_file)
#             R_results.append((mask_dict[key], R))
        
#         return R_results

import cv2
import numpy as np

import numpy as np

class Hida_syuho:
    def __init__(self, mask_img, hxy2, xc, yc):
        self.mask_img = mask_img
        self.hxy2 = hxy2
        self.xc = xc
        self.yc = yc
        self.xc_int = int(round(xc))
        self.yc_int = int(round(yc))
        self.h, self.w = mask_img.shape

    def calculate_top2_average(self):#4つに分けて上位2つの平均を取る
        mask_ul = self.mask_img[0:self.yc_int, 0:self.xc_int]
        hida_ul = self.hxy2[0:self.yc_int, 0:self.xc_int]
        count_mask_ul = np.sum(mask_ul == 255)
        count_hida_ul = np.sum(hida_ul == 1)
        R_ul = count_hida_ul / count_mask_ul if count_mask_ul > 0 else 0

        mask_ur = self.mask_img[0:self.yc_int, self.xc_int:self.w]
        hida_ur = self.hxy2[0:self.yc_int, self.xc_int:self.w]
        count_mask_ur = np.sum(mask_ur == 255)
        count_hida_ur = np.sum(hida_ur == 1)
        R_ur = count_hida_ur / count_mask_ur if count_mask_ur > 0 else 0

        mask_ll = self.mask_img[self.yc_int:self.h, 0:self.xc_int]
        hida_ll = self.hxy2[self.yc_int:self.h, 0:self.xc_int]
        count_mask_ll = np.sum(mask_ll == 255)
        count_hida_ll = np.sum(hida_ll == 1)
        R_ll = count_hida_ll / count_mask_ll if count_mask_ll > 0 else 0

        mask_lr = self.mask_img[self.yc_int:self.h, self.xc_int:self.w]
        hida_lr = self.hxy2[self.yc_int:self.h, self.xc_int:self.w]
        count_mask_lr = np.sum(mask_lr == 255)
        count_hida_lr = np.sum(hida_lr == 1)
        R_lr = count_hida_lr / count_mask_lr if count_mask_lr > 0 else 0

        R_values = [R_ul, R_ur, R_ll, R_lr]
        average_top2 = sum(sorted(R_values, reverse=True)[:2]) / 2

        return average_top2
    
    def calculate_45rotate(self):#論文の手法，45°回転して2つに分けて最大値を取る
        def calc_R(mask, hida):
            count_mask = np.sum(mask == 255)
            count_hida = np.sum(hida == 1)
            return count_hida / count_mask if count_mask > 0 else 0

        def rotate(img, angle):
            M = cv2.getRotationMatrix2D((self.xc, self.yc), angle, 1.0)
            return cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_NEAREST)

        R_values = []

        for angle in [0, 45, 90]:
            mask_rot = rotate(self.mask_img, angle) if angle != 0 else self.mask_img
            hida_rot = rotate(self.hxy2, angle) if angle != 0 else self.hxy2

            if angle % 180 == 0:
                # 左右分割
                left_mask = mask_rot[:, :self.xc]
                right_mask = mask_rot[:, self.xc:]
                left_hida = hida_rot[:, :self.xc]
                right_hida = hida_rot[:, self.xc:]
                R_values.append(calc_R(left_mask, left_hida))
                R_values.append(calc_R(right_mask, right_hida))
            else:
                # 上下分割
                top_mask = mask_rot[:self.yc, :]
                bottom_mask = mask_rot[self.yc:, :]
                top_hida = hida_rot[:self.yc, :]
                bottom_hida = hida_rot[self.yc:, :]
                R_values.append(calc_R(top_mask, top_hida))
                R_values.append(calc_R(bottom_mask, bottom_hida))

        return max(R_values)


class Hida_file_jikuari:
    def __init__(self, img_path, mask_path, n=15, T=0.2,method='top2average'):
        self.img_path = img_path
        self.mask_path = mask_path
        self.n = n
        self.T = T
        self.img = cv2.imread(img_path)
        self.mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = self.img.shape[:2]
        self.method = method

    def fmxy(self, absfxy, mxy):
        return np.where(absfxy > mxy, 1, 0)

    def Min(self, a, b):
        return np.minimum(a, b)

    def G12(self, theta1, theta2):
        condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
        result1 = np.abs(theta1 - theta2)
        condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
        result2 = theta2 - 2 * np.pi - theta1
        condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
        result3 = np.abs(theta1 - theta2)
        condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
        result4 = theta1 - theta2 - 2 * np.pi
        result = np.where(condition1, result1,
                          np.where(condition2, result2,
                                   np.where(condition3, result3,
                                            np.where(condition4, result4, 0))))
        return result

    def SquareSum(self, I, x, y, h, w, n):
        x1, y1 = x - n, y - n
        x2, y2 = x + n, y + n
        x1, x2 = max(x1, 0), min(x2, w - 2)
        y1, y2 = max(y1, 0), min(y2, h - 2)
        total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
        return total

    def sdis(self, Iruv, Imyv, x, y, h, w, n):
        Tr = self.SquareSum(Iruv, x, y, h, w, n)
        Tm = self.SquareSum(Imyv, x, y, h, w, n)
        return Tr / Tm

    def main(self):
        img = self.img
        mask_img = self.mask_img
        masked_img = img.copy()
        h, w = self.h, self.w
        n = self.n
        T = self.T

        x_sum, y_sum, count = 0, 0, 0
        for i in range(h):
            for j in range(w):
                if mask_img[i][j] == 255:
                    x_sum += j
                    y_sum += i
                    count += 1
        xc, yc = (x_sum / count, y_sum / count) if count > 0 else (0, 0)
        xc_int, yc_int = int(round(xc)), int(round(yc))

        image = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        fdy, fdx = np.gradient(image)
        f0xy = np.arctan2(fdy, fdx)
        absfxy = np.uint8(np.sqrt(fdx ** 2 + fdy ** 2))

        height, width = image.shape
        C0xy = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                dx, dy = x - xc, y - yc
                C0xy[y, x] = np.arctan(dy / dx) if dx != 0 else 0

        fdisxy = self.Min(self.G12(C0xy + np.pi / 2, f0xy) ** 2,
                          self.G12(C0xy - np.pi / 2, f0xy) ** 2)

        kernel_size = 2 * n + 1
        mxy = np.uint8(cv2.medianBlur(absfxy, kernel_size))

        rdisxy = self.fmxy(absfxy, mxy) * fdisxy
        Iruv = cv2.integral(rdisxy)
        Imyv = cv2.integral(self.fmxy(absfxy, mxy).astype(np.uint8))

        sdisval = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                sdisval[y, x] = self.sdis(Iruv, Imyv, x, y, h, w, n)
        sdisval = np.nan_to_num(sdisval, nan=0.0, posinf=0.0, neginf=0.0)

        hxy = np.where(sdisval < T, 1, 0)
        hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)

        method_instance = Hida_syuho(self.mask_img, hxy2, int(xc), int(yc))
        if self.method == '45rotate':
            R = method_instance.calculate_45rotate()
        elif self.method == 'top2average':
            R = method_instance.calculate_top2_average()
        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return R


import os
import cv2
import numpy as np
import re

import os
import cv2
import numpy as np
import re

class Hida_folder_jikuari:
    def __init__(self, base_dir, img_dir="combined", mask_dir="mask", subfolder="collage_1", n=15, T=0.2, method='top2average'):
        self.base_dir = base_dir
        self.subfolder = subfolder
        self.n = n
        self.T = T
        self.combined_path = os.path.join(base_dir, img_dir, subfolder)
        self.mask_path = os.path.join(base_dir, mask_dir, subfolder)
        self.method = method

    def fmxy(self, absfxy, mxy):
        return np.where(absfxy > mxy, 1, 0)

    def Min(self, a, b):
        return np.minimum(a, b)

    def G12(self, theta1, theta2):
        condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
        result1 = np.abs(theta1 - theta2)
        condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
        result2 = theta2 - 2 * np.pi - theta1
        condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
        result3 = np.abs(theta1 - theta2)
        condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
        result4 = theta1 - theta2 - 2 * np.pi
        result = np.where(condition1, result1,
                 np.where(condition2, result2,
                 np.where(condition3, result3,
                 np.where(condition4, result4, 0))))
        return result

    def SquareSum(self, I, x, y, h, w):
        n = self.n
        x1, y1 = x - n, y - n
        x2, y2 = x + n, y + n
        x1, x2 = max(x1, 0), min(x2, w - 2)
        y1, y2 = max(y1, 0), min(y2, h - 2)
        total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
        return total

    def sdis(self, Iruv, Imyv, x, y, h, w):
        Tr = self.SquareSum(Iruv, x, y, h, w)
        Tm = self.SquareSum(Imyv, x, y, h, w)
        return Tr / Tm if Tm != 0 else 0

    def extract_index(self, name):
        match = re.search(r'(\d+_\d+)', name)
        return match.group(1) if match else None

    def calculate_R(self, img_path, mask_path):
        instance = Hida_file_jikuari(img_path, mask_path, n=self.n, T=self.T, method=self.method)
        return instance.main()

    def run_all(self):
        img_files = os.listdir(self.combined_path)
        mask_files = os.listdir(self.mask_path)

        self.img_dict = {self.extract_index(f): f for f in img_files if self.extract_index(f)}
        self.mask_dict = {self.extract_index(f): f for f in mask_files if self.extract_index(f)}
        common_keys = sorted(set(self.img_dict.keys()) & set(self.mask_dict.keys()))

        R_results = []
        for key in common_keys:
            try:
                img_file = os.path.join(self.combined_path, self.img_dict[key])
                mask_file = os.path.join(self.mask_path, self.mask_dict[key])
                R = self.calculate_R(img_file, mask_file)
                R_results.append((self.mask_dict[key], R))
            except Exception as e:
                print(f"Error processing {key}: {e}")
        return R_results



class Hida_folder_jikuari_img_xxxx:
    def __init__(self, base_dir, img_dir="combined", mask_dir="mask", subfolder="collage_1", n=15, T=0.2, method='top2average'):
        self.base_dir = base_dir
        self.subfolder = subfolder
        self.n = n
        self.T = T
        self.combined_path = os.path.join(base_dir, img_dir, subfolder)
        self.mask_path = os.path.join(base_dir, mask_dir, subfolder)
        self.method = method

    def fmxy(self, absfxy, mxy):
        return np.where(absfxy > mxy, 1, 0)

    def Min(self, a, b):
        return np.minimum(a, b)

    def G12(self, theta1, theta2):
        condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
        result1 = np.abs(theta1 - theta2)
        condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
        result2 = theta2 - 2 * np.pi - theta1
        condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
        result3 = np.abs(theta1 - theta2)
        condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
        result4 = theta1 - theta2 - 2 * np.pi
        result = np.where(condition1, result1,
                 np.where(condition2, result2,
                 np.where(condition3, result3,
                 np.where(condition4, result4, 0))))
        return result

    def SquareSum(self, I, x, y, h, w):
        n = self.n
        x1, y1 = x - n, y - n
        x2, y2 = x + n, y + n
        x1, x2 = max(x1, 0), min(x2, w - 2)
        y1, y2 = max(y1, 0), min(y2, h - 2)
        total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
        return total

    def sdis(self, Iruv, Imyv, x, y, h, w):
        Tr = self.SquareSum(Iruv, x, y, h, w)
        Tm = self.SquareSum(Imyv, x, y, h, w)
        return Tr / Tm if Tm != 0 else 0

    def extract_index(self, name):
        """
        ファイル名から共通のキーを抽出する方法を変更。
        "ファイル名本体_combined.png" や "ファイル名本体_mask.png" から、
        "ファイル名本体" の部分（例: "IMG_1829"）を抽出する。
        """
        # ファイル名から "_combined.png" または "_mask.png" を取り除く
        if name.endswith("_combined.png"):
            return name[:-13] # "_combined.png" の文字数
        elif name.endswith("_mask.png"):
            return name[:-9] # "_mask.png" の文字数
        else:
            # 念のため、予期しないファイル名の場合は拡張子のみ取り除く
            return os.path.splitext(name)[0]

    def calculate_R(self, img_path, mask_path):
        instance = Hida_file_jikuari(img_path, mask_path, n=self.n, T=self.T, method=self.method)
        return instance.main()

    def run_all(self):
        img_files = os.listdir(self.combined_path)
        mask_files = os.listdir(self.mask_path)

        self.img_dict = {self.extract_index(f): f for f in img_files if self.extract_index(f)}
        self.mask_dict = {self.extract_index(f): f for f in mask_files if self.extract_index(f)}
        common_keys = sorted(set(self.img_dict.keys()) & set(self.mask_dict.keys()))

        R_results = []
        for key in common_keys:
            try:
                img_file = os.path.join(self.combined_path, self.img_dict[key])
                mask_file = os.path.join(self.mask_path, self.mask_dict[key])
                R = self.calculate_R(img_file, mask_file)
                R_results.append((self.mask_dict[key], R))
            except Exception as e:
                print(f"Error processing {key}: {e}")
        return R_results


# import hida
# a = hida.Hida_file(img_path = "/home/data/maesyori_img/combined/collage_1/combined_1_1.jpg", mask_path = "/home/data/maesyori_img/mask/collage_1/mask_1_1.jpg")
# result = a.calculate_R()
# print(result)
# a = hida.Hida_folder(base_dir="/home/data/maesyori_img")
# result = a.run_all_watch()
