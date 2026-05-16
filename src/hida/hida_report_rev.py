import os
import re
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from icecream import ic

# fmxy (7)
def fmxy(absfxy, mxy):
    return np.where(absfxy > mxy, 1, 0)

def Min(a, b):
    return np.minimum(a, b)

def G12(theta1, theta2):
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

# 平方面積の合計 (11)(12) Tr Tm
def SquareSum(I, x, y, h, w, n):
    x1, y1 = x - n, y - n
    x2, y2 = x + n, y + n
    x1, x2 = max(x1, 0), min(x2, w - 2)
    y1, y2 = max(y1, 0), min(y2, h - 2)
    total = I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]
    return total

# sdis = Tr/Tm (13)
def sdis(Iruv, Imyv, x, y, h, w, n):
    Tr = SquareSum(Iruv, x, y, h, w, n)
    Tm = SquareSum(Imyv, x, y, h, w, n)
    return Tr / Tm

# パラメータ
n = 15
date = "0203_energee_after"
data_list = []

img_folders = [
    f"/home/data/{date}/maskedBB/A",
    f"/home/data/{date}/maskedBB/B",
    f"/home/data/{date}/maskedBB/C",
]
mask_folders = [
    f"/home/data/{date}/maskBB/A",
    f"/home/data/{date}/maskBB/B",
    f"/home/data/{date}/maskBB/C",
]

for img_folder, mask_folder in zip(img_folders, mask_folders):
    folder_name = os.path.basename(img_folder)
    img_files = os.listdir(img_folder)
    mask_files = os.listdir(mask_folder)
    
    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(img_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # 重心 (1)
        img = cv2.imread(img_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masked_img = cv2.imread(img_path)
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
        fdisxy = Min(G12(C0xy + np.pi/2, f0xy)**2, G12(C0xy - np.pi/2, f0xy)**2)
        
        # mxy = |fxy|に対する2n+1×2n+1のメディアンフィルタリングの結果
        kernel_size = 2 * n + 1
        mxy = np.uint8(cv2.medianBlur(absfxy, kernel_size))
        
        # rdis (8)
        rdisxy = fmxy(absfxy, mxy) * fdisxy
        
        # Iruv, Imyv (9)(10)
        Iruv = cv2.integral(rdisxy)
        Imyv = cv2.integral(fmxy(absfxy, mxy).astype(np.uint8))
        
        # sdis計算
        sdisval = np.zeros((image.shape[0], image.shape[1]))
        for y in range(0, image.shape[0], 1):
            for x in range(0, image.shape[1], 1):
                sdisval[y, x] = sdis(Iruv, Imyv, x, y, h, w, n)
        sdisval = np.nan_to_num(sdisval, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 閾値処理
        T = 0.2
        hxy = np.where(sdisval < T, 1, 0)
        hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)
        
        # シイタケ領域のPixel数を計算
        count_mask = np.sum(mask_img == 255)
        count_hida = np.sum(hxy2 == 1)
        R = count_hida / count_mask if count_mask > 0 else 0
        data_list.append((img_file, folder_name, R))
        
output_csv = f"/home/data/{date}/R_values.csv"
df = pd.DataFrame(data_list, columns=["filename", "R", "folder"])
df.to_csv(output_csv, index=False)
