# import os
# import shutil

# # 3つのフォルダを指定して、1つ目のフォルダ内のファイルと2つ目のフォルダ内のファイルの名前が一致するファイルを3つ目のフォルダに移動する

# # フォルダのパスを指定
# source_folder = '/home/YOLO/0930_bbox/datasets/images/val'  # このフォルダにあるファイルと同じ名前のファイルを移動
# compare_folder = '/home/YOLO/0930_bbox/datasets/labels/0930_bbox'  # 移動元
# destination_folder = '/home/YOLO/0930_bbox/datasets/labels/val'  # 移動先

# # 1つ目のフォルダからファイルのリストを取得
# source_files = os.listdir(source_folder)

# # 2つ目のフォルダからファイルのリストを取得
# compare_files = os.listdir(compare_folder)

# # 1つ目のフォルダ内の各ファイルに対して処理
# for file in source_files:
#     # ファイル名（拡張子なし）を取得
#     file_name_without_extension = os.path.splitext(file)[0]
    
#     # 2つ目のフォルダ内のファイルと名前が一致するか確認
#     for compare_file in compare_files:
#         compare_file_name_without_extension = os.path.splitext(compare_file)[0]
#         if file_name_without_extension == compare_file_name_without_extension:
#             # 一致するファイルを3つ目のフォルダに移動
#             src_path = os.path.join(compare_folder, compare_file)
#             dst_path = os.path.join(destination_folder, compare_file)
#             shutil.move(src_path, dst_path)
#             print(f'Moved: {compare_file}')


import os
import shutil

def move_matching_files(source_folder, compare_folder, destination_folder):
    # 1つ目のフォルダからファイルのリストを取得
    source_files = os.listdir(source_folder)

    # 2つ目のフォルダからファイルのリストを取得
    compare_files = os.listdir(compare_folder)

    # 1つ目のフォルダ内の各ファイルに対して処理
    for file in source_files:
        # ファイル名（拡張子なし）を取得
        file_name_without_extension = os.path.splitext(file)[0]
        
        # 2つ目のフォルダ内のファイルと名前が一致するか確認
        for compare_file in compare_files:
            compare_file_name_without_extension = os.path.splitext(compare_file)[0]
            if file_name_without_extension == compare_file_name_without_extension:
                # 一致するファイルを3つ目のフォルダに移動
                src_path = os.path.join(compare_folder, compare_file)
                dst_path = os.path.join(destination_folder, compare_file)
                shutil.move(src_path, dst_path)
                print(f'Moved: {compare_file}')

if __name__ == "__main__":
    import sys
    source_folder = sys.argv[1]
    compare_folder = sys.argv[2]
    destination_folder = sys.argv[3]
    move_matching_files(source_folder, compare_folder, destination_folder)