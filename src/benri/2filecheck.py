import os

# フォルダのパスを指定
folder1 = "/home/coco-annotator/datasets/aaa"
folder2 = '/home/coco-annotator/datasets/bbb'

# フォルダ内のファイルリストを取得
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# 共通しないファイルを取得
unique_files1 = files1 - files2
unique_files2 = files2 - files1

# ファイルに書き込む
with open('gazou.txt', 'w') as f:
    for file in unique_files1:
        f.write(f"{file}\n")
    for file in unique_files2:
        f.write(f"{file}\n")

print("gazou.txtに共通しないファイルを書き込みました。")
