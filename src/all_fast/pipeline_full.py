# pipeline_full.py
import torch
from ultralytics import YOLO
from PIL import Image
import sys
import os
import cv2
import numpy as np
import glob
import shutil
import time

# このファイルは Jupyter Notebook からインポートされて使用されます。
# このファイル自体を直接実行するものではありません。

# -----------------------------------------------------------------
# [ステップ1] セグメンテーションとクロップ
# -----------------------------------------------------------------
def segment_and_crop(input_image_dir: str, model_path: str, crop_dir: str, mask_dir: str):
    print(f"[{os.getpid()}] ステップ1 開始: {input_image_dir}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"モデルの読み込みに失敗しました: {model_path}. 詳細: {e}")

    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    allowed_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for ext in allowed_extensions:
        image_paths.extend(glob.glob(os.path.join(input_image_dir, ext)))

    if not image_paths:
        return

    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            continue

        results = model(img, retina_masks=True)
        result = results[0]
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        if result.masks is not None and result.boxes is not None:
            num_objects = len(result.masks.data)
            original_img_np = np.array(img)

            for i in range(num_objects):
                try:
                    bbox_tensor = result.boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, bbox_tensor)
                    
                    cropped_object_rgb = original_img_np[y1:y2, x1:x2]
                    cropped_object_bgr = cv2.cvtColor(cropped_object_rgb, cv2.COLOR_RGB2BGR)
                    crop_filename = f"{base_filename}_object_{i}.png"
                    crop_path = os.path.join(crop_dir, crop_filename)
                    cv2.imwrite(crop_path, cropped_object_bgr)

                    full_mask_canvas = np.zeros(original_img_np.shape[:2], dtype=np.uint8)
                    mask_tensor = result.masks.data[i]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask_np, (original_img_np.shape[1], original_img_np.shape[0]))
                    full_mask_canvas[mask_resized > 0] = 255
                    
                    cropped_mask = full_mask_canvas[y1:y2, x1:x2]
                    
                    mask_filename = f"{base_filename}_mask_{i}.png"
                    mask_path = os.path.join(mask_dir, mask_filename)
                    cv2.imwrite(mask_path, cropped_mask)
                except Exception as e:
                    continue
# -----------------------------------------------------------------
# [ステップ1] セグメンテーションとクロップ (★バッチ処理 高速化版)
# -----------------------------------------------------------------
# def segment_and_crop(input_image_dir: str, model_path: str, crop_dir: str, mask_dir: str):
#     print(f"[{os.getpid()}] ステップ1 開始: {input_image_dir}")
#     try:
#         model = YOLO(model_path)
#         # GPUが使えるなら明示的にGPUに送る
#         if torch.cuda.is_available():
#             model.to('cuda')
#         print(f"[{os.getpid()}] モデルを {model.device} にロードしました。")
#     except Exception as e:
#         raise RuntimeError(f"モデルの読み込みに失敗しました: {model_path}. 詳細: {e}")

#     os.makedirs(crop_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     allowed_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
#     image_paths = []
#     for ext in allowed_extensions:
#         image_paths.extend(glob.glob(os.path.join(input_image_dir, ext)))

#     if not image_paths:
#         print(f"[{os.getpid()}] 画像が見つかりません: {input_image_dir}")
#         return

#     # --- ▼▼▼ ここからが変更点 ▼▼▼ ---

#     # 1. 画像パスのリストをYOLOにまとめて渡す (バッチ推論)
#     #    stream=False で全結果をリストとして受け取る
#     print(f"[{os.getpid()}] {len(image_paths)} 枚の画像のバッチ推論を開始...")
#     try:
#         # batch引数でバッチサイズを明示的に指定できます（例: batch=16）
#         # VRAMが許す範囲で設定してください。None (auto) でも構いません。
#         results_list = model(
#             image_paths, 
#             retina_masks=True, 
#             verbose=False, 
#             stream=False,
#             batch = 2
#             # batch=None # Noneで自動、または 8, 16, 32 などを試す
#         )
#     except Exception as e:
#         print(f"[{os.getpid()}] バッチ推論中にエラー: {e}")
#         # VRAM不足(OOM)の場合、batchサイズを小さくする必要がある
#         return

#     print(f"[{os.getpid()}] バッチ推論完了。後処理を開始...")

#     # 2. 推論結果のリストをループ処理する
#     #    (i番目のresultが、i番目のimage_pathに対応する)
#     for i, result in enumerate(results_list):
#         image_path = image_paths[i] # 対応する画像パス
#         base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
#         # 後処理のために元画像をNumPy配列として読み込む
#         try:
#             # 元のコードではPIL→NumPyだったが、OpenCVで直接読む
#             original_img_np_bgr = cv2.imread(image_path)
#             if original_img_np_bgr is None:
#                 raise IOError("画像が読み込めません")
#             original_img_np = cv2.cvtColor(original_img_np_bgr, cv2.COLOR_BGR2RGB)
#         except Exception as e:
#             print(f"[{os.getpid()}] 画像の読み込み失敗 (スキップ): {image_path}, {e}")
#             continue
            
#         if result.masks is not None and result.boxes is not None:
#             num_objects = len(result.masks.data)

#             for j in range(num_objects):
#                 try:
#                     # --- 元のコードの処理 (ほぼそのまま) ---
#                     bbox_tensor = result.boxes.xyxy[j]
#                     x1, y1, x2, y2 = map(int, bbox_tensor)
                    
#                     cropped_object_rgb = original_img_np[y1:y2, x1:x2]
#                     cropped_object_bgr = cv2.cvtColor(cropped_object_rgb, cv2.COLOR_RGB2BGR)
#                     crop_filename = f"{base_filename}_object_{j}.png"
#                     crop_path = os.path.join(crop_dir, crop_filename)
#                     cv2.imwrite(crop_path, cropped_object_bgr)

#                     full_mask_canvas = np.zeros(original_img_np.shape[:2], dtype=np.uint8)
#                     mask_tensor = result.masks.data[j]
#                     mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    
#                     # マスクを元画像サイズにリサイズ
#                     if mask_np.shape != original_img_np.shape[:2]:
#                          mask_resized = cv2.resize(mask_np, (original_img_np.shape[1], original_img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
#                     else:
#                          mask_resized = mask_np
                    
#                     full_mask_canvas[mask_resized > 0] = 255
#                     cropped_mask = full_mask_canvas[y1:y2, x1:x2]
                    
#                     mask_filename = f"{base_filename}_mask_{j}.png"
#                     mask_path = os.path.join(mask_dir, mask_filename)
#                     cv2.imwrite(mask_path, cropped_mask)
#                     # --- 元のコードの処理ここまで ---
#                 except Exception as e:
#                     print(f"[{os.getpid()}] オブジェクト処理失敗 (スキップ): {image_path} (object {j}), {e}")
#                     continue
#     # --- ▲▲▲ 変更点ここまで ▲▲▲ ---

# -----------------------------------------------------------------
# [ステップ2] マスクのクリーンアップ（最大連結成分）
# -----------------------------------------------------------------
def _keep_largest_component(image: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    if num_labels <= 2:
        return image
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    cleaned_mask = np.zeros_like(image)
    cleaned_mask[labels == largest_component_label] = 255
    return cleaned_mask

def clean_masks_keep_largest(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")
    if not image_files:
        return
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            continue
        cleaned_image = _keep_largest_component(mask_image)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cleaned_image)


# -----------------------------------------------------------------
# [ステップ3] 面積によるフィルタリングとクロップ削除
# -----------------------------------------------------------------
def filter_masks_by_area(
    source_mask_dir: str, 
    large_mask_dir: str, 
    small_mask_dir: str, 
    crop_dir: str, 
    area_threshold: int
):
    os.makedirs(large_mask_dir, exist_ok=True)
    os.makedirs(small_mask_dir, exist_ok=True)
    try:
        filenames = [
            f for f in os.listdir(source_mask_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"元フォルダが見つかりません: {source_mask_dir}")
    if not filenames:
        return
    for filename in filenames:
        source_path = os.path.join(source_mask_dir, filename)
        try:
            img = cv2.imread(source_path)
            if img is None:
                continue
            height, width, _ = img.shape
            area = height * width
            
            if area < area_threshold:
                shutil.copy(source_path, os.path.join(small_mask_dir, filename))
                crop_filename = filename.replace('_mask_', '_object_')
                crop_path_to_delete = os.path.join(crop_dir, crop_filename)
                if os.path.exists(crop_path_to_delete):
                    os.remove(crop_path_to_delete)
            else:
                shutil.copy(source_path, os.path.join(large_mask_dir, filename))
        except Exception as e:
            continue


# -----------------------------------------------------------------
# [ステップ4] マスクの穴埋め
# -----------------------------------------------------------------
def _fill_true_holes(image: np.ndarray) -> np.ndarray:
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    output_image = binary_image.copy()
    inverted_image = cv2.bitwise_not(binary_image)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary_image.shape[:2]
    
    for cnt in contours:
        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
        is_on_border = (x == 0) or (y == 0) or (x + rect_w == w) or (y + rect_h == h)
        
        if not is_on_border:
            cv2.drawContours(output_image, [cnt], -1, color=255, thickness=cv2.FILLED)
    return output_image

def fill_mask_holes(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")
    if not image_files:
        return
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            continue
        filled_image = _fill_true_holes(mask_image)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, filled_image)


# -----------------------------------------------------------------
# [ステップ5] マスクの適用
# -----------------------------------------------------------------
def apply_mask_to_crop(crop_image_dir: str, mask_image_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        crop_files = [
            f for f in os.listdir(crop_image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"クロップ画像フォルダが見つかりません: {crop_image_dir}")
    if not crop_files:
        return
    for crop_filename in crop_files:
        mask_filename = crop_filename.replace('_object_', '_mask_')
        crop_path = os.path.join(crop_image_dir, crop_filename)
        mask_path = os.path.join(mask_image_dir, mask_filename)
        if not os.path.exists(mask_path):
            # --- ▼▼▼ デバッグプリント追加 ▼▼▼ ---
            print(f"[{os.getpid()}] ステップ5 スキップ: {crop_filename} に対応するマスクが見つかりません。")
            print(f"    -> 検索したパス: {mask_path}")
            # --- ▲▲▲ ここまで ▲▲▲ ---
            continue
            
        try:
            crop_image = cv2.imread(crop_path)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if crop_image is None or mask_image is None:
                continue

            _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            masked_image = cv2.bitwise_and(crop_image, mask_3ch)

            output_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(output_path, masked_image)
            
        except Exception as e:
            continue


# -----------------------------------------------------------------
# [ステップ6] ファイル名の変更
# -----------------------------------------------------------------
def _rename_files_in_folder(target_dir: str, suffix: str):
    try:
        filenames = [
            f for f in os.listdir(target_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    except FileNotFoundError:
        return
    if not filenames:
        return
    for filename in filenames:
        try:
            base, ext = os.path.splitext(filename)
            parts = base.split('_')
            
            original_num_str = parts[1] # '2642'
            index_str = parts[-1]       # '0'
            
            xxx = original_num_str[-3:] # '642'
            y = index_str               # '0'
            
            new_filename = f"IMG_{xxx}{y}_{suffix}{ext}" # 'IMG_6420_crop.png'
            
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
        except IndexError:
            print(f"[{os.getpid()}] ステップ6 リネーム失敗 (IndexError): ファイル名 '{filename}' の形式が不正です。")
            continue 
        except Exception as e:
            print(f"[{os.getpid()}] ステップ6 リネーム失敗 (その他): {filename}, エラー: {e}")
            continue

def rename_processed_files(crop_dir: str, mask_dir: str, combined_dir: str):
    _rename_files_in_folder(crop_dir, 'crop')
    _rename_files_in_folder(mask_dir, 'mask')
    _rename_files_in_folder(combined_dir, 'combined')


# -----------------------------------------------------------------
# [高レベル関数] パイプライン実行 (★これがワーカーになる)
# -----------------------------------------------------------------
def process_image_subdirectory(
    base_data_dir: str, 
    sub_folder: str, 
    model_path: str, 
    area_threshold: int = 50000
):
    """
    指定されたサブフォルダ（例: 'A'）に対して、
    セグメンテーションからリネームまでの一連の処理をすべて実行する。
    """
    
    print(f"--- [{os.getpid()}] 処理開始: {sub_folder} ---")
    start_time = time.time()
    
    # パス定義
    path_org = os.path.join(base_data_dir, 'org', sub_folder)
    path_crop = os.path.join(base_data_dir, 'crop', sub_folder)
    path_mask_org = os.path.join(base_data_dir, 'mask_org', sub_folder)
    path_mask_to1 = os.path.join(base_data_dir, 'mask_to1', sub_folder)
    path_mask_large = os.path.join(base_data_dir, 'mask_large', sub_folder)
    path_mask_small = os.path.join(base_data_dir, 'mask_small', sub_folder)
    path_mask_final = os.path.join(base_data_dir, 'mask', sub_folder)
    path_combined = os.path.join(base_data_dir, 'combined', sub_folder)

    # ステップ1: セグメンテーション
    segment_and_crop(
        input_image_dir=path_org,
        model_path=model_path,
        crop_dir=path_crop,
        mask_dir=path_mask_org
    )

    # ステップ2: 最大連結成分
    clean_masks_keep_largest(
        input_dir=path_mask_org,
        output_dir=path_mask_to1
    )

    # ステップ3: 面積フィルタリング
    filter_masks_by_area(
        source_mask_dir=path_mask_to1,
        large_mask_dir=path_mask_large,
        small_mask_dir=path_mask_small,
        crop_dir=path_crop,
        area_threshold=area_threshold
    )

    # ステップ4: 穴埋め
    fill_mask_holes(
        input_dir=path_mask_large,
        output_dir=path_mask_final
    )

    # ステップ5: マスク適用
    apply_mask_to_crop(
        crop_image_dir=path_crop,
        mask_image_dir=path_mask_final,
        output_dir=path_combined
    )

    # ステップ6: リネーム
    rename_processed_files(
        crop_dir=path_crop,
        mask_dir=path_mask_final,
        combined_dir=path_combined
    )
    
    end_time = time.time()
    print(f"--- [{os.getpid()}] 処理完了: {sub_folder} (所要時間: {end_time - start_time:.2f}秒) ---")