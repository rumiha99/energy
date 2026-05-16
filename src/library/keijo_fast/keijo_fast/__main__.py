import os
import glob
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import time

# ===================================================================
# 1. 各プロセスから呼び出されるヘルパー関数
#    - クラスの外に定義することで、プロセスの親子関係での問題を回避します
# ===================================================================
def analyze_image_for_process(img_path):
    """
    単一の画像ファイルを処理し、(ファイル名, MSE値) のタプルを返す。
    エラーが発生した場合は None を返す。
    """
    try:
        # KeijoAnalyzerの画像解析ロジックをここに直接記述
        mask = cv2.imread(img_path)
        if mask is None:
            # print(f"⚠️ 読み込み失敗: {img_path}") # tqdmと一緒だと表示が崩れるためコメントアウト
            return None

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        radius = int(radius)

        M = cv2.moments(max_contour)
        if M["m00"] == 0:
            return None
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        center = (cX, cY)

        h, w = gray.shape
        flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        linear_polar = cv2.warpPolar(gray, (w, h), center, radius, flags)

        black_pixel_count = np.sum(linear_polar == 0, axis=1)
        y_pseudo = np.zeros_like(black_pixel_count)
        mse = mean_squared_error(y_pseudo, black_pixel_count)

        file_name = os.path.basename(img_path)
        return (file_name, mse)
        
    except Exception as e:
        # print(f"⚠️ エラー発生 ({os.path.basename(img_path)}): {e}")
        return None

# ===================================================================
# 2. クラス定義（runメソッドを修正）
# ===================================================================
class KeijoAnalyzer:
    """KeijoAnalyzerの基本クラス。画像解析のロジックはヘルパー関数に移動。"""
    def __init__(self):
        # runメソッドが結果を返すため、インスタンス変数としてのresultsは必須ではなくなる
        pass

class Keijo_file(KeijoAnalyzer):
    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

    def run(self):
        # 単一ファイルの場合はヘルパー関数を直接呼び出す
        result = analyze_image_for_process(self.img_path)
        return result[1] if result else None # MSE のみ返す

class Keijo_folder(KeijoAnalyzer):
    def __init__(self, base_dir, mask_dir="mask", subfolder="collage_1", output_csv=None):
        super().__init__()
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.subfolder = subfolder
        self.folder_path = os.path.join(base_dir, mask_dir, subfolder)
        self.output_csv = output_csv or os.path.join(base_dir, f"keijo_mse_{subfolder}.csv")

    def run(self):
        """
        フォルダ内の画像を並列処理し、結果をリストとして返す。
        [('ファイル名1', MSE1), ('ファイル名2', MSE2), ...]
        """
        # 元のプログラムに合わせて.pngを検索
        image_paths = glob.glob(os.path.join(self.folder_path, "*.png"))
        if not image_paths:
            print(f"⚠️ フォルダ内にPNG画像がありません: {self.folder_path}")
            return []

        print(f"📂 {self.folder_path} の画像 {len(image_paths)} 枚をマルチプロセスで処理します。")

        mse_results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 各画像パスに対してヘルパー関数を並列実行
            results_iterator = executor.map(analyze_image_for_process, image_paths)

            # tqdmを使って進捗を表示しつつ、Noneでない結果のみをリストに追加
            for result in tqdm(results_iterator, total=len(image_paths), desc=f"処理中 ({self.subfolder})"):
                if result is not None:
                    mse_results.append(result)

        print(f"✅ {len(mse_results)} 枚の画像の処理が完了しました。")
        return mse_results

    def save_results_from_list(self, results_list):
        """
        run()から返されたリストを受け取り、CSVに保存するメソッド。
        """
        if not results_list:
            print("⚠️ 保存する結果がありません。")
            return
            
        # リストからDataFrameを作成
        df = pd.DataFrame(results_list, columns=['filename', 'mse'])
        df = df.set_index('filename')
        
        df.to_csv(self.output_csv)
        print(f"✅ 結果を保存: {self.output_csv}")


