import os
import shutil
import random

# 4つのフォルダを指定し、3つのフォルダからファイルをランダムに選択して1つのフォルダにコピーする

# ソースフォルダとターゲットフォルダのパスを指定
source_folders = [
                '/home/data/shiitake_maybeuse/A', 
                '/home/data/shiitake_maybeuse/B', 
                '/home/data/shiitake_maybeuse/CD'
                ]
target_folder = '/home/coco-annotator/datasets/bbox_train2'
num_files_to_copy = 14  # ここにコピーするファイルの数を指定

# 各ソースフォルダからファイルをランダムに選択し、ターゲットフォルダにコピー
for folder in source_folders:
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    selected_files = random.sample(files, min(len(files), num_files_to_copy))
    for file in selected_files:
        shutil.copy(os.path.join(folder, file), target_folder)