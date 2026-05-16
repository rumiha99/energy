# import os
# import shutil
# from random import sample

# #3つのフォルダを指定して、1つ目のフォルダ内のファイルを2つのフォルダに移動する

# # フォルダのパスを指定
# source_folder = '/home/YOLO/0930_bbox/datasets/images/0930_bbox'
# destination_folder1 = '/home/YOLO/0930_bbox/datasets/images/test'
# destination_folder2 = '/home/YOLO/0930_bbox/datasets/images/val'

# # 移動するファイルの数を指定
# num_files_to_move = 4

# # 移動元フォルダからファイルのリストを取得
# files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# # 指定された数だけファイルをランダムに選択
# files_to_move = sample(files, min(num_files_to_move, len(files)))

# # 選択したファイルを移動先フォルダに移動
# for i, file in enumerate(files_to_move):
#     src_path = os.path.join(source_folder, file)
#     if i % 2 == 0:
#         dst_path = os.path.join(destination_folder1, file)
#     else:
#         dst_path = os.path.join(destination_folder2, file)
#     shutil.move(src_path, dst_path)

# print(f'{len(files_to_move)} files have been moved.')


import os
import shutil
from random import sample

def move_files(source_folder, destination_folder1, destination_folder2, num_files_to_move):
    # 移動元フォルダからファイルのリストを取得
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # 指定された数だけファイルをランダムに選択
    files_to_move = sample(files, min(num_files_to_move, len(files)))

    # 選択したファイルを移動先フォルダに移動
    for i, file in enumerate(files_to_move):
        src_path = os.path.join(source_folder, file)
        if i % 2 == 0:
            dst_path = os.path.join(destination_folder1, file)
        else:
            dst_path = os.path.join(destination_folder2, file)
        shutil.move(src_path, dst_path)

    print(f'{len(files_to_move)} files have been moved.')

if __name__ == "__main__":
    import sys
    source_folder = sys.argv[1]
    destination_folder1 = sys.argv[2]
    destination_folder2 = sys.argv[3]
    num_files_to_move = int(sys.argv[4])
    move_files(source_folder, destination_folder1, destination_folder2, num_files_to_move)