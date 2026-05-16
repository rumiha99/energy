import os
import cv2
import numpy as np
import concurrent.futures
import time

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
            # 回転中心はクラスのxc, ycを使用
            M = cv2.getRotationMatrix2D((self.xc, self.yc), angle, 1.0)
            return cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_NEAREST)

        R_values = []
        xc_int, yc_int = self.xc_int, self.yc_int # 整数値の重心

        for angle in [0, 45, 90, 135]: # 論文に合わせて135°も追加する場合
            mask_rot = rotate(self.mask_img, angle)
            hida_rot = rotate(self.hxy2, angle)

            # 分割処理を修正 (重心で分割)
            # 0, 180度方向 -> 左右分割
            left_mask = mask_rot[:, :xc_int]
            right_mask = mask_rot[:, xc_int:]
            left_hida = hida_rot[:, :xc_int]
            right_hida = hida_rot[:, xc_int:]
            R_values.append(calc_R(left_mask, left_hida))
            R_values.append(calc_R(right_mask, right_hida))

            # 90, 270度方向 -> 上下分割
            top_mask = mask_rot[:yc_int, :]
            bottom_mask = mask_rot[yc_int:, :]
            top_hida = hida_rot[:yc_int, :]
            bottom_hida = hida_rot[yc_int:, :]
            R_values.append(calc_R(top_mask, top_hida))
            R_values.append(calc_R(bottom_mask, bottom_hida))

        return max(R_values) if R_values else 0

# ワーカー関数（変更なし）
def process_image_pair(args):
    img_path, mask_path, n, T, method, key = args
    try:
        instance = Hida_file_jikuari(img_path, mask_path, n=n, T=T, method=method)
        R = instance.main()
        # print(f"✅ Processed: {os.path.basename(mask_path)}")
        return (os.path.basename(mask_path), R)
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(mask_path)}: {e}")
        return (os.path.basename(mask_path), None)

class Hida_file_jikuari:
    def __init__(self, img_path, mask_path, n=15, T=0.2, method='top2average'):
        self.img_path = img_path
        self.mask_path = mask_path
        self.n = n
        self.T = T
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        self.mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_img is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        self.h, self.w = self.img.shape[:2]
        self.method = method

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


    def main(self):
        img, mask_img, h, w, n, T = self.img, self.mask_img, self.h, self.w, self.n, self.T
        
        ys, xs = np.where(mask_img == 255)
        xc, yc = (np.mean(xs), np.mean(ys)) if ys.size > 0 else (w / 2, h / 2)

        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fdy, fdx = np.gradient(image.astype(float))
        f0xy = np.arctan2(fdy, fdx)
        absfxy = np.sqrt(fdx**2 + fdy**2).astype(np.uint8)
        

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        C0xy_dx, C0xy_dy = xx - xc, yy - yc
        C0xy = np.arctan2(C0xy_dy, C0xy_dx) # arctan2版
        non_zero_dx_mask = C0xy_dx != 0    # arctan版
        C0xy = np.zeros_like(C0xy_dx, dtype=float)
        C0xy[non_zero_dx_mask] = np.arctan(C0xy_dy[non_zero_dx_mask] / C0xy_dx[non_zero_dx_mask])

        fdisxy = np.minimum(self.G12(C0xy + np.pi / 2, f0xy)**2,
                            self.G12(C0xy - np.pi / 2, f0xy)**2)
        
        mxy = cv2.medianBlur(absfxy, 2 * n + 1)
        fmxy_val = np.where(absfxy > mxy, 1, 0)
        
        rdisxy = fmxy_val * fdisxy
        Iruv = cv2.integral(rdisxy)
        Imyv = cv2.integral(fmxy_val.astype(np.uint8))
        
        sdisval = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                x1, y1 = max(0, x - n), max(0, y - n)
                x2, y2 = min(w - 1, x + n), min(h - 1, y + n)
                
                Tr = Iruv[y2+1, x2+1] - Iruv[y1, x2+1] - Iruv[y2+1, x1] + Iruv[y1, x1]
                Tm = Imyv[y2+1, x2+1] - Imyv[y1, x2+1] - Imyv[y2+1, x1] + Imyv[y1, x1]
                if Tm > 0:
                    sdisval[y, x] = Tr / Tm
        
        hxy = np.where(sdisval < T, 1, 0).astype(np.uint8)
        hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask_img)

        method_instance = Hida_syuho(self.mask_img, hxy2, xc, yc)
        if self.method == '45rotate':
            R = method_instance.calculate_45rotate()
        elif self.method == 'top2average':
            R = method_instance.calculate_top2_average()
        else:
            raise ValueError(f"Unknown method '{self.method}'")
        return R

# フォルダ全体を管理するクラス（変更なし）
class Hida_folder_jikuari_img_xxxx:
    def __init__(self, base_dir, img_dir="combined", mask_dir="mask", subfolder="collage_1", n=15, T=0.2, method='top2average'):
        self.base_dir = base_dir
        self.subfolder = subfolder
        self.n = n
        self.T = T
        self.combined_path = os.path.join(base_dir, img_dir, subfolder)
        self.mask_path = os.path.join(base_dir, mask_dir, subfolder)
        self.method = method

    def extract_index(self, name):
        if name.endswith("_combined.png"):
            return name[:-13]
        elif name.endswith("_mask.png"):
            return name[:-9]
        else:
            return os.path.splitext(name)[0]

    def run_all(self):
        img_files = os.listdir(self.combined_path)
        mask_files = os.listdir(self.mask_path)

        img_dict = {self.extract_index(f): f for f in img_files}
        mask_dict = {self.extract_index(f): f for f in mask_files}
        common_keys = sorted(set(img_dict.keys()) & set(mask_dict.keys()))

        tasks = []
        for key in common_keys:
            img_file = os.path.join(self.combined_path, img_dict[key])
            mask_file = os.path.join(self.mask_path, mask_dict[key])
            tasks.append((img_file, mask_file, self.n, self.T, self.method, key))

        R_results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_iterator = executor.map(process_image_pair, tasks)
            R_results = [result for result in results_iterator if result and result[1] is not None]
            
        return R_results

