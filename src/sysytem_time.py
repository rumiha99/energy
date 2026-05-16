import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import joblib

import hida
import keijo
import size_module

# === 椎茸画像1枚を処理する関数 ===
def process_image(image_path):
    results = detection_model.predict(image_path, verbose=False)
    orig_img = results[0].orig_img
    img_h, img_w, _ = orig_img.shape
    rows, cols = 4, 6
    cell_h, cell_w = img_h // rows, img_w // cols
    cell_bboxes = [[[] for _ in range(cols)] for _ in range(rows)]

    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            start_x, start_y, end_x, end_y = map(int, box)
            center_x, center_y = (start_x + end_x) // 2, (start_y + end_y) // 2
            row_idx = min(center_y // cell_h, rows - 1)
            col_idx = min(center_x // cell_w, cols - 1)
            cell_bboxes[row_idx][col_idx].append((start_x, start_y, end_x, end_y))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_folder = os.path.join(crop_output_folder, base_name)
    mask_output_folder_specific = os.path.join(mask_output_folder, base_name)
    combined_output_folder_specific = os.path.join(combined_output_folder, base_name)
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder_specific, exist_ok=True)
    os.makedirs(combined_output_folder_specific, exist_ok=True)

    for row in range(rows):
        for col in range(cols):
            if not cell_bboxes[row][col]:
                continue
            for idx, (sx, sy, ex, ey) in enumerate(cell_bboxes[row][col]):
                clip_img = orig_img[sy:ey, sx:ex]
                clip_filename = os.path.join(image_output_folder, f"clip_{row+1}_{col+1}.jpg")
                cv2.imwrite(clip_filename, clip_img)

                mask_results = segmentation_model.predict(clip_img, verbose=False)
                if mask_results and mask_results[0].masks is not None:
                    mask = mask_results[0].masks.data[0].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (clip_img.shape[1], clip_img.shape[0]))
                    mask_filename = os.path.join(mask_output_folder_specific, f"mask_{row+1}_{col+1}.jpg")
                    cv2.imwrite(mask_filename, mask_resized)

                    mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                    combined_img = cv2.bitwise_and(clip_img, mask_rgb)
                    combined_filename = os.path.join(combined_output_folder_specific, f"combined_{row+1}_{col+1}.jpg")
                    cv2.imwrite(combined_filename, combined_img)

# === 初期設定 ===
start = time.time()
data = "maesyori_img"
input_base = "/home/data/maesyori_img"
input_files = [os.path.join(input_base, f"collage_{i}.jpg") for i in range(1, 11)]

mask_output_folder = f"/home/data/{data}/mask/"
crop_output_folder = f"/home/data/{data}/crop/"
combined_output_folder = f"/home/data/{data}/combined/"
os.makedirs(mask_output_folder, exist_ok=True)
os.makedirs(crop_output_folder, exist_ok=True)
os.makedirs(combined_output_folder, exist_ok=True)

# YOLOモデル読み込み
detection_model = YOLO('/home/YOLO/hukusuu_train/datasets/train7/weights/best.pt')
segmentation_model = YOLO("/home/YOLO/-327_seg/datasets/train2/weights/best.pt")

# === collage画像をすべて処理 ===
for img_path in input_files:
    print(f"Processing: {img_path}")
    process_image(img_path)

# === 特徴量抽出・推論 ===
svm_model = joblib.load("/home/src/svm_model_jikuari.pkl")
scaler = joblib.load("/home/src/scaler_jikuari.pkl")

for i in range(1, 11):
    subfolder = f"collage_{i}"

    hida_tapple = hida.Hida_folder_jikuari(base_dir=f"/home/data/{data}", subfolder=subfolder,method="45rotate")
    result_hida = hida_tapple.run_all()

    size_tapple = size_module.Size_folder(base_dir=f"/home/data/{data}", subfolder=subfolder)
    result_size = size_tapple.count_white_pixels()

    keijo_tapple = keijo.Keijo_folder(base_dir=f"/home/data/{data}", subfolder=subfolder)
    result_keijo = keijo_tapple.run()

    dict_hida = dict(result_hida)
    dict_size = dict(result_size)
    dict_keijo = dict(result_keijo)

    merged = []
    for filename in dict_hida.keys() & dict_size.keys() & dict_keijo.keys():
        merged.append({
            'filename': filename,
            'MSE': dict_keijo[filename],
            'size_count': dict_size[filename],
            'R': dict_hida[filename]
        })

    df = pd.DataFrame(merged)
    X_new = df[["MSE", "size_count", "R"]]
    X_new = scaler.transform(X_new)
    y_pred_new = svm_model.predict(X_new)
    df["Predicted_Label"] = y_pred_new

    # 保存
    df.to_csv(f"/home/data/{data}/predicted_results_collage_{i}.csv", index=False)
    print(df[["filename", "MSE", "size_count", "R", "Predicted_Label"]])

end = time.time()
print(f"Total time: {end - start:.2f} sec")
