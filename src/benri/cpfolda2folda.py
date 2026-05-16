import os
import random
import shutil

#2つフォルダを指定しファイルをコピーする

def copy_random_files(src_folder, dest_folder, num_files):
    # フォルダ内のファイルリストを取得
    all_files = os.listdir(src_folder)
    
    # 指定した数のファイルをランダムに選択
    selected_files = random.sample(all_files, num_files)
    
    # ファイルをコピー
    for file_name in selected_files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy2(src_path, dest_path)
    
    print(f"{num_files}個のファイルを{src_folder}から{dest_folder}にコピーしました。")

# 使用例
src_folder = '/home/data/0707_sam/data/cropped_results_with_padding'#コピー元
dest_folder = '/home/coco-annotator/datasets/0728_segmet'#コピー先
num_files = 300  # コピーするファイルの数を指定

copy_random_files(src_folder, dest_folder, num_files)
