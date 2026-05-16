# workers.py
import os
import cv2
import numpy as np
import shutil
import functools
import sys

# -----------------------------------------------------------------
# [共通] 並列処理のためのエラーハンドリングラッパー
# -----------------------------------------------------------------
def _log_error(func):
    """
    ワーカー関数で発生した例外をキャッチし、
    Noneを返して他のプロセスの実行を継続させるデコレータ。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"ワーカー処理エラー ({func.__name__}): {e} | 引数: {args}", file=sys.stderr)
            return None
    return wrapper

# -----------------------------------------------------------------
# [ステップ2] ワーカーとヘルパー
# -----------------------------------------------------------------
def _keep_largest_component(image: np.ndarray) -> np.ndarray:
    """
    二値画像を受け取り、最大の連結成分のみを残した画像を返す。(内部ヘルパー関数)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    if num_labels <= 2: # 背景(0)と1つの成分(1)のみ
        return image
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    cleaned_mask = np.zeros_like(image)
    cleaned_mask[labels == largest_component_label] = 255
    return cleaned_mask

@_log_error
def _clean_mask_worker(args):
    """ [ワーカー] 1枚のマスク画像をクリーンアップする """
    input_path, output_path = args
    
    mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        return

    cleaned_image = _keep_largest_component(mask_image)
    cv2.imwrite(output_path, cleaned_image)

# -----------------------------------------------------------------
# [ステップ3] ワーカー
# -----------------------------------------------------------------
@_log_error
def _filter_mask_worker(args):
    """ [ワーカー] 1枚のマスクを面積でフィルタリングし、対応するクロップを処理する """
    source_path, large_mask_dir, small_mask_dir, crop_dir, area_threshold, filename = args

    img = cv2.imread(source_path)
    if img is None:
        return
        
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

# -----------------------------------------------------------------
# [ステップ4] ワーカーとヘルパー
# -----------------------------------------------------------------
def _fill_true_holes(image: np.ndarray) -> np.ndarray:
    """
    画像内の「完全に白で囲まれた黒い領域（真の穴）」のみを塗りつぶす。(内部ヘルパー関数)
    """
    if image.ndim != 2:
        raise ValueError("入力画像はグレースケールである必要があります")

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    output_image = binary_image.copy()
    
    inverted_image = cv2.bitwise_not(binary_image)
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return output_image

    h, w = binary_image.shape[:2]

    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            x, y, rect_w, rect_h = cv2.boundingRect(contours[i])
            is_on_border = (x == 0) or (y == 0) or (x + rect_w >= w-1) or (y + rect_h >= h-1)
            
            if not is_on_border:
                cv2.drawContours(output_image, [contours[i]], -1, color=255, thickness=cv2.FILLED)
                
    return output_image

@_log_error
def _fill_mask_hole_worker(args):
    """ [ワーカー] 1枚のマスク画像の穴を埋める """
    input_path, output_path = args
    
    mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        return

    filled_image = _fill_true_holes(mask_image)
    cv2.imwrite(output_path, filled_image)

# -----------------------------------------------------------------
# [ステップ5] ワーカー
# -----------------------------------------------------------------
@_log_error
def _apply_mask_worker(args):
    """ [ワーカー] 1枚のクロップ画像にマスクを適用する """
    crop_path, mask_image_dir, output_dir, crop_filename = args
    
    mask_filename = crop_filename.replace('_object_', '_mask_')
    mask_path = os.path.join(mask_image_dir, mask_filename)
    
    if not os.path.exists(mask_path):
        return
        
    crop_image = cv2.imread(crop_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if crop_image is None or mask_image is None:
        return

    if crop_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image, (crop_image.shape[1], crop_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(crop_image, mask_3ch)

    output_path = os.path.join(output_dir, crop_filename)
    cv2.imwrite(output_path, masked_image)

# -----------------------------------------------------------------
# [ステップ6] ワーカー
# -----------------------------------------------------------------
@_log_error
def _rename_worker(args):
    """ [ワーカー] 1つのファイルをリネームする """
    target_dir, filename, suffix = args
    
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    
    try:
        original_num_str = parts[1] # '2642'
        index_str = parts[-1]       # '0'
        
        xxx = original_num_str[-3:] # '642'
        y = index_str               # '0'
        
        new_filename = f"IMG_{xxx}{y}_{suffix}{ext}" # 'IMG_6420_crop.png'
        
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_filename)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
    except (IndexError, Exception) as e:
        print(f"リネーム失敗: {filename}, {e}", file=sys.stderr)